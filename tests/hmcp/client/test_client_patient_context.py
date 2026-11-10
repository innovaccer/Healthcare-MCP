"""Client-side patient context and guardrail-report verification tests.

These tests serve a plain ``mcp.server.mcpserver.MCPServer`` with the real
:class:`PatientContextExtension` and a fake :class:`TokenVerifier` (a token
whose ``patient`` claim matches the requested patient) over
``streamable_http_app()`` with an in-process ``httpx2.ASGITransport`` passed as
the connector's ``transport`` seam. A small second extension stamps a guardrail
report so the client-side verification can be exercised. All values are
synthetic.
"""

from __future__ import annotations

import contextlib
import copy
import json
from typing import Any

import httpx2
import mcp.types as types
import pytest
from mcp.client.extension import ClientExtension
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import Context, MCPServer
from mcp.shared.exceptions import MCPError

from hmcp.client.client_connector import HMCPClientConnector
from hmcp.client.patient_context import (
    GuardrailReportMissing,
    PatientContextGetResult,
    PatientContextHandle,
    PatientContextOpenParams,
)
from hmcp.server.patient_context import (
    PatientContextConfig,
    PatientContextExtension,
)
from hmcp.shared.constants import (
    GUARDRAILS_EXTENSION,
    METHOD_PATIENT_CONTEXT_OPEN,
    PATIENT_CONTEXT_EXTENSION,
)

BASE = "http://127.0.0.1:8000"
MCP_URL = f"{BASE}/mcp"

TOKEN = "tok-1"
AUTH_HEADERS = {"Authorization": f"Bearer {TOKEN}"}
PATIENT_REFERENCE = "Patient/123"

REQUIRED_ID = "com.example/x"
REQUIRED_ID_V1 = "com.example/x@1"


class _Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        if token != TOKEN:
            return None
        return AccessToken(
            token=token,
            client_id="app-1",
            scopes=["user/*.rs"],
            resource=MCP_URL,
            subject="practitioner-1",
            claims={"patient": "123"},
        )


class _ReportExtension(Extension):
    """Stamps a guardrail report onto every result except ``failing_tool``."""

    identifier = GUARDRAILS_EXTENSION

    def settings(self) -> dict[str, Any]:
        return {"revision": "1"}

    async def intercept_tool_call(self, params, ctx, call_next):
        result = await call_next(ctx)
        if params.name == "failing_tool":
            return result
        if isinstance(result, types.CallToolResult):
            meta = dict(result.meta or {})
            meta[GUARDRAILS_EXTENSION] = {
                "report": [{"id": REQUIRED_ID, "version": "1.0.0", "verdict": "pass"}]
            }
            return result.model_copy(update={"meta": meta})
        return result


def _scope_for(name: str) -> str:
    return "patient" if name == "read_chart" else "none"


def build_server() -> MCPServer:
    config = PatientContextConfig()
    patient_ext = PatientContextExtension(config, scope_for=_scope_for)
    server = MCPServer(
        "k1-test",
        extensions=[patient_ext, _ReportExtension()],
        token_verifier=_Verifier(),
        auth=AuthSettings(
            issuer_url="https://auth.example.com",
            resource_server_url=MCP_URL,
            validate_token_resource=True,
        ),
    )

    @server.tool()
    async def read_chart(patient_note: str = "ok") -> str:
        return f"chart:{patient_note}"

    @server.tool()
    async def see_meta(ctx: Context) -> str:
        return json.dumps(dict(ctx.request_context.meta or {}))

    @server.tool()
    async def failing_tool() -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="boom")], is_error=True
        )

    return server


@contextlib.asynccontextmanager
async def running(server: MCPServer):
    app = server.streamable_http_app()
    transport = httpx2.ASGITransport(app=app)
    async with server.session_manager.run():
        yield transport


def connector_for(transport: httpx2.ASGITransport) -> HMCPClientConnector:
    return HMCPClientConnector(
        url=MCP_URL, transport=transport, headers=dict(AUTH_HEADERS)
    )


# ---------------------------------------------------------------------------
# Patient context lifecycle
# ---------------------------------------------------------------------------


async def test_open_returns_handle_and_get_returns_patient():
    server = build_server()
    async with running(server) as transport:
        async with connector_for(transport) as connector:
            handle = await connector.open_patient_context(PATIENT_REFERENCE)
            assert isinstance(handle, PatientContextHandle)
            assert handle.context_id.startswith("pc_")
            assert handle.status == "active"
            assert handle.expires_at.tzinfo is not None

            fetched = await connector.get_patient_context(handle)
            assert isinstance(fetched, PatientContextGetResult)
            assert fetched.patient is not None
            assert fetched.patient.reference == PATIENT_REFERENCE
            assert fetched.context_id == handle.context_id
            assert fetched.status == "active"


async def test_patient_scoped_tool_requires_context():
    server = build_server()
    async with running(server) as transport:
        async with connector_for(transport) as connector:
            handle = await connector.open_patient_context(PATIENT_REFERENCE)

            result = await connector.call_tool(
                "read_chart", {"patient_note": "hi"}, patient_context=handle
            )
            assert result.content[0].text == "chart:hi"

            with pytest.raises(MCPError) as excinfo:
                await connector.call_tool("read_chart", {"patient_note": "hi"})
            assert excinfo.value.code == -31001


async def test_closed_context_is_invalid():
    server = build_server()
    async with running(server) as transport:
        async with connector_for(transport) as connector:
            handle = await connector.open_patient_context(PATIENT_REFERENCE)
            await connector.close_patient_context(handle)

            with pytest.raises(MCPError) as excinfo:
                await connector.call_tool(
                    "read_chart", {"patient_note": "hi"}, patient_context=handle
                )
            assert excinfo.value.code == -31002


# ---------------------------------------------------------------------------
# meta merge and guardrail verification
# ---------------------------------------------------------------------------


async def test_meta_merge_preserves_caller_keys():
    server = build_server()
    async with running(server) as transport:
        async with connector_for(transport) as connector:
            handle = await connector.open_patient_context(PATIENT_REFERENCE)

            caller_meta = {
                PATIENT_CONTEXT_EXTENSION: {"contextId": "CALLER", "extra": 1},
                GUARDRAILS_EXTENSION: {"require": ["OLD"], "keep": 2},
                "custom": "keep",
            }
            snapshot = copy.deepcopy(caller_meta)

            result = await connector.call_tool(
                "see_meta",
                meta=caller_meta,
                patient_context=handle,
                require_guardrails=[REQUIRED_ID_V1],
            )
            meta = json.loads(result.content[0].text)
            # Explicit arguments win, other keys (top-level and nested) survive.
            assert meta["custom"] == "keep"
            assert meta[PATIENT_CONTEXT_EXTENSION]["contextId"] == handle.context_id
            assert meta[PATIENT_CONTEXT_EXTENSION]["extra"] == 1
            assert meta[GUARDRAILS_EXTENSION]["require"] == [REQUIRED_ID_V1]
            assert meta[GUARDRAILS_EXTENSION]["keep"] == 2
            # The caller's own dicts are not mutated.
            assert caller_meta == snapshot


async def test_required_guardrail_present_does_not_raise():
    server = build_server()
    async with running(server) as transport:
        async with connector_for(transport) as connector:
            handle = await connector.open_patient_context(PATIENT_REFERENCE)
            result = await connector.call_tool(
                "read_chart",
                {"patient_note": "hi"},
                patient_context=handle,
                require_guardrails=[REQUIRED_ID_V1],
            )
            assert result.content[0].text == "chart:hi"
            report = result.meta[GUARDRAILS_EXTENSION]["report"]
            assert report[0]["id"] == REQUIRED_ID


async def test_missing_guardrail_raises_naming_only_the_id():
    server = build_server()
    async with running(server) as transport:
        async with connector_for(transport) as connector:
            handle = await connector.open_patient_context(PATIENT_REFERENCE)
            with pytest.raises(GuardrailReportMissing) as excinfo:
                await connector.call_tool(
                    "read_chart",
                    {"patient_note": "hi"},
                    patient_context=handle,
                    require_guardrails=["com.example/y"],
                )
            assert excinfo.value.guardrail_id == "com.example/y"
            message = str(excinfo.value)
            assert "com.example/y" in message
            assert PATIENT_REFERENCE not in message


async def test_error_result_still_requires_guardrail_report():
    server = build_server()
    async with running(server) as transport:
        async with connector_for(transport) as connector:
            with pytest.raises(GuardrailReportMissing):
                await connector.call_tool(
                    "failing_tool", require_guardrails=[REQUIRED_ID]
                )


# ---------------------------------------------------------------------------
# Default extension declarations
# ---------------------------------------------------------------------------


def test_default_extensions_declared_without_duplicates():
    class CallerPatientContextExtension(ClientExtension):
        identifier = PATIENT_CONTEXT_EXTENSION

        def settings(self) -> dict[str, Any]:
            return {"caller": True}

    connector = HMCPClientConnector(
        url=MCP_URL, extensions=[CallerPatientContextExtension()]
    )
    identifiers = [ext.identifier for ext in connector._extensions]
    assert identifiers.count(PATIENT_CONTEXT_EXTENSION) == 1
    assert identifiers.count(GUARDRAILS_EXTENSION) == 1
    assert any(
        isinstance(ext, CallerPatientContextExtension) for ext in connector._extensions
    )


# ---------------------------------------------------------------------------
# Guardrail-report verification (malformed meta must never satisfy a require)
# ---------------------------------------------------------------------------


def _call_result(meta: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(content=[], meta=meta)


@pytest.mark.parametrize(
    "guardrails",
    [
        "oops",  # _meta[GUARDRAILS_EXTENSION] is a str
        [{"id": REQUIRED_ID}],  # ... a list
        {"report": 7},  # report is an int
        {"report": [{"id": 123}]},  # id is not a str
        None,
    ],
)
def test_verify_rejects_malformed_meta(guardrails):
    result = _call_result({GUARDRAILS_EXTENSION: guardrails})
    with pytest.raises(GuardrailReportMissing):
        HMCPClientConnector._verify_guardrail_reports(result, [REQUIRED_ID])


@pytest.mark.parametrize(
    "report",
    [
        [{"id": REQUIRED_ID + "x"}],
        [{"id": "a" + REQUIRED_ID}],
        ["bogus", {"verdict": "pass"}],
    ],
)
def test_verify_requires_an_exact_str_id(report):
    result = _call_result({GUARDRAILS_EXTENSION: {"report": report}})
    with pytest.raises(GuardrailReportMissing):
        HMCPClientConnector._verify_guardrail_reports(result, [REQUIRED_ID])


def test_verify_pinned_requirement_accepts_matching_major():
    result = _call_result(
        {
            GUARDRAILS_EXTENSION: {
                "report": [{"id": REQUIRED_ID, "version": "1.4.2", "verdict": "pass"}]
            }
        }
    )
    HMCPClientConnector._verify_guardrail_reports(result, [REQUIRED_ID_V1])


def test_verify_pinned_requirement_accepts_bare_integer_version():
    result = _call_result(
        {GUARDRAILS_EXTENSION: {"report": [{"id": REQUIRED_ID, "version": "1"}]}}
    )
    HMCPClientConnector._verify_guardrail_reports(result, [REQUIRED_ID_V1])


@pytest.mark.parametrize(
    "entry",
    [
        {"id": REQUIRED_ID, "version": "2.0.0"},  # wrong major
        {"id": REQUIRED_ID},  # missing version
        {"id": REQUIRED_ID, "version": 1},  # non-string version
        {"id": REQUIRED_ID, "version": None},  # non-string version
        {"id": REQUIRED_ID, "version": "v1.0.0"},  # no leading integer
    ],
)
def test_verify_pinned_requirement_rejects_non_matching(entry):
    result = _call_result({GUARDRAILS_EXTENSION: {"report": [entry]}})
    with pytest.raises(GuardrailReportMissing) as excinfo:
        HMCPClientConnector._verify_guardrail_reports(result, [REQUIRED_ID_V1])
    # The full requirement string (including the pin) is named.
    assert excinfo.value.guardrail_id == REQUIRED_ID_V1


def test_verify_unpinned_requirement_accepts_any_version():
    result = _call_result(
        {GUARDRAILS_EXTENSION: {"report": [{"id": REQUIRED_ID, "version": "9.9.9"}]}}
    )
    HMCPClientConnector._verify_guardrail_reports(result, [REQUIRED_ID])


# ---------------------------------------------------------------------------
# Naive expiry rejection
# ---------------------------------------------------------------------------


class _NaiveExpiryExtension(Extension):
    """Serves patientContext/open with a naive ``expiresAt`` (no offset)."""

    identifier = "com.example/naive-expiry"

    def settings(self) -> dict[str, Any]:
        return {"revision": "1"}

    def methods(self):
        async def handler(ctx, params):
            return {
                "contextId": "pc_naive",
                "status": "active",
                "expiresAt": "2026-01-01T00:00:00",
            }

        return [
            MethodBinding(
                method=METHOD_PATIENT_CONTEXT_OPEN,
                params_type=PatientContextOpenParams,
                handler=handler,
            )
        ]


async def test_naive_expiry_from_server_is_rejected():
    server = MCPServer("naive-expiry", extensions=[_NaiveExpiryExtension()])
    async with running(server) as transport:
        async with HMCPClientConnector(url=MCP_URL, transport=transport) as connector:
            with pytest.raises(ValueError, match="UTC offset"):
                await connector.open_patient_context(PATIENT_REFERENCE)
