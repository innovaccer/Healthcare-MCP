"""Tests for HMCPServer (HMCP 0.1.0, mcp 2.3.0 / MCP spec 2026-07-28).

All tokens, ids and resources here are synthetic.
"""

from __future__ import annotations

import logging

import httpx2
import mcp.types as types
import pytest
from mcp import MCPDeprecationWarning
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from hmcp.server.guardrails import GuardrailsConfig, GuardrailsExtension
from hmcp.server.hmcp_server import HMCPServer
from hmcp.server.patient_context import PatientContextConfig
from hmcp.server.patient_context.extension import (
    PatientContextIdParams,
    PatientContextOpenParams,
)
from hmcp.shared.constants import (
    METHOD_PATIENT_CONTEXT_CLOSE,
    METHOD_PATIENT_CONTEXT_GET,
    METHOD_PATIENT_CONTEXT_OPEN,
    PATIENT_CONTEXT_EXTENSION,
)
from hmcp.shared.errors import HMCPErrorCode
from hmcp.shared.guardrails import redaction_placeholder_writeback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Verifier:
    """Token verifier over a fixed, synthetic token table."""

    def __init__(self, tokens: dict[str, AccessToken] | None = None) -> None:
        self._tokens = tokens or {}

    async def verify_token(self, token: str) -> AccessToken | None:
        return self._tokens.get(token)


def _echo_server() -> HMCPServer:
    server = HMCPServer("t")

    @server.tool()
    async def echo(text: str) -> str:
        return "echo:" + text

    return server


# ---------------------------------------------------------------------------
# In-memory client: subclassing, tool round-trip, capability advertisement
# ---------------------------------------------------------------------------


async def test_hmcp_server_is_mcpserver_and_tool_round_trips() -> None:
    server = HMCPServer("t")
    assert isinstance(server, MCPServer)

    @server.tool()
    async def echo(text: str) -> str:
        return "echo:" + text

    async with Client(server) as client:
        result = await client.call_tool("echo", {"text": "hi"})

    assert result.content[0].text == "echo:hi"


async def test_server_advertises_no_experimental_hmcp_capability() -> None:
    server = HMCPServer("t")
    async with Client(server) as client:
        capabilities = client.server_capabilities

    experimental = capabilities.experimental
    assert experimental is None or "hmcp" not in experimental


# ---------------------------------------------------------------------------
# Sampling is gone
# ---------------------------------------------------------------------------


async def test_sampling_is_removed_and_refused() -> None:
    server = HMCPServer("t")

    # Main HMCP-specific check: the sampling API is gone.
    assert not hasattr(server, "sampling")
    assert not hasattr(server, "_samplingCallback")

    # Secondary: the SDK still sends a raw sampling request; the server refuses it.
    async with Client(server) as client:
        request = types.CreateMessageRequest(
            method="sampling/createMessage",
            params=types.CreateMessageRequestParams(messages=[], maxTokens=10),
        )
        with pytest.raises(MCPError) as excinfo:
            await client.session.send_request(request, types.CreateMessageResult)

    assert excinfo.value.code == -32601


# ---------------------------------------------------------------------------
# Transport restriction
# ---------------------------------------------------------------------------


def test_run_rejects_sse_transport() -> None:
    server = HMCPServer("t")
    with pytest.raises(ValueError) as excinfo:
        server.run(transport="sse")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "HTTP+SSE" in message
    assert "2026-07-28" in message


def test_run_rejects_unknown_transport_with_supported_list() -> None:
    server = HMCPServer("t")
    with pytest.raises(ValueError) as excinfo:
        server.run(transport="foo")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "foo" in message
    assert "stdio" in message
    assert "streamable-http" in message
    # The HTTP+SSE deprecation message is only for "sse".
    assert "2026-07-28" not in message


def test_sse_app_rejected() -> None:
    server = HMCPServer("t")
    with pytest.raises(ValueError) as excinfo:
        server.sse_app()
    assert "2026-07-28" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Authentication defaults
# ---------------------------------------------------------------------------


def test_warns_without_authentication(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="hmcp.server.hmcp_server"):
        HMCPServer("t")
    assert any(
        "running without authentication" in record.message for record in caplog.records
    )


def test_no_warning_with_token_verifier(caplog: pytest.LogCaptureFixture) -> None:
    # MCPServer requires auth settings whenever a token_verifier is supplied.
    auth = AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url="http://127.0.0.1:9999/mcp",
        validate_token_resource=True,
    )
    with caplog.at_level(logging.WARNING, logger="hmcp.server.hmcp_server"):
        HMCPServer("t", token_verifier=_Verifier(), auth=auth)
    assert not any(
        "running without authentication" in record.message for record in caplog.records
    )


def test_validate_token_resource_defaults_to_true() -> None:
    # Constructing AuthSettings without the field warns; the warning is expected.
    with pytest.warns(MCPDeprecationWarning):
        auth = AuthSettings(
            issuer_url="https://auth.example.com",
            resource_server_url="http://127.0.0.1:9999/mcp",
        )

    server = HMCPServer("t", token_verifier=_Verifier(), auth=auth)

    assert server.settings.auth.validate_token_resource is True


def test_validate_token_resource_explicit_false_is_kept() -> None:
    auth = AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url="http://127.0.0.1:9999/mcp",
        validate_token_resource=False,
    )

    server = HMCPServer("t", token_verifier=_Verifier(), auth=auth)

    assert server.settings.auth.validate_token_resource is False


def test_missing_resource_server_url_rejected() -> None:
    # resource_server_url unset, validate_token_resource unset: binding cannot
    # work, so construction must fail rather than silently claim it is on.
    auth = AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url=None,
    )

    with pytest.raises(ValueError) as excinfo:
        HMCPServer("t", token_verifier=_Verifier(), auth=auth)

    message = str(excinfo.value)
    assert "resource_server_url" in message
    assert "RFC 8707" in message


def test_missing_resource_server_url_with_explicit_false_accepted() -> None:
    # Explicit False is the documented escape hatch: the verifier checks audience.
    auth = AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url=None,
        validate_token_resource=False,
    )

    server = HMCPServer("t", token_verifier=_Verifier(), auth=auth)

    assert server.settings.auth.validate_token_resource is False


# ---------------------------------------------------------------------------
# Authenticated streamable-http over httpx2.ASGITransport
# ---------------------------------------------------------------------------

_BASE = "http://127.0.0.1:8000"
_RESOURCE = f"{_BASE}/mcp"

_TOKENS = {
    "tok-valid": AccessToken(
        token="tok-valid",
        client_id="app-1",
        scopes=["patient/*.rs"],
        subject="alice",
        resource=_RESOURCE,
    ),
    # Issued for a different resource server: audience binding must reject it.
    "tok-other-audience": AccessToken(
        token="tok-other-audience",
        client_id="app-1",
        scopes=["patient/*.rs"],
        subject="bob",
        resource="http://127.0.0.1:9999/mcp",
    ),
}


def _auth_server() -> HMCPServer:
    server = HMCPServer(
        "auth-test",
        token_verifier=_Verifier(_TOKENS),
        auth=AuthSettings(
            issuer_url="https://auth.example.com",
            resource_server_url=_RESOURCE,
            validate_token_resource=True,
        ),
    )

    @server.tool()
    async def ping() -> str:
        return "pong"

    return server


async def _call_over_http(app, token: str | None) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    http_client = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app),
        base_url=_BASE,
        headers=headers,
    )
    async with Client(
        streamable_http_client(f"{_BASE}/mcp", http_client=http_client)
    ) as client:
        result = await client.call_tool("ping", {})
    return result.content[0].text


async def test_streamable_http_valid_token_calls_tool() -> None:
    server = _auth_server()
    app = server.streamable_http_app()
    async with server.session_manager.run():
        assert await _call_over_http(app, "tok-valid") == "pong"


async def test_streamable_http_missing_token_rejected() -> None:
    server = _auth_server()
    app = server.streamable_http_app()
    async with server.session_manager.run():
        with pytest.raises(Exception):
            await _call_over_http(app, None)


async def test_streamable_http_wrong_audience_token_rejected() -> None:
    server = _auth_server()
    app = server.streamable_http_app()
    async with server.session_manager.run():
        with pytest.raises(Exception):
            await _call_over_http(app, "tok-other-audience")


# ---------------------------------------------------------------------------
# Guardrails wiring, direct add_tool, and the SSE entry point
# ---------------------------------------------------------------------------


def test_guardrails_property() -> None:
    assert HMCPServer("t").guardrails is None

    config = GuardrailsConfig(definitions=[redaction_placeholder_writeback()])
    server = HMCPServer("t", guardrails=config)

    assert isinstance(server.guardrails, GuardrailsExtension)


async def test_add_tool_direct_none_scope_advertised_and_enforced() -> None:
    server = HMCPServer("t", patient_context=PatientContextConfig())

    async def plain() -> str:
        return "plain-ran"

    # Registered directly through add_tool (not the decorator).
    server.add_tool(plain, patient_scope="none")

    async with Client(server) as client:
        tools = await client.list_tools()
        meta = {tool.name: (tool.meta or {}) for tool in tools.tools}
        assert meta["plain"][PATIENT_CONTEXT_EXTENSION] == {"scope": "none"}

        result = await client.call_tool("plain", {})
        assert result.content[0].text == "plain-ran"


async def test_run_sse_async_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        await HMCPServer("t").run_sse_async()
    assert "2026-07-28" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Patient context on a server with no authentication fails closed
# ---------------------------------------------------------------------------

_NOT_AUTHENTICATED_MESSAGE = "Request is not authenticated"
_NO_AUTH_WARNING_MARKER = "every patient-context request"


class _RawResult(BaseModel):
    model_config = {"extra": "allow"}


def _assert_not_authenticated(error: MCPError) -> None:
    assert error.code == int(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)
    assert error.message == _NOT_AUTHENTICATED_MESSAGE


async def test_no_auth_patient_context_requests_are_not_authenticated() -> None:
    server = HMCPServer("t", patient_context=PatientContextConfig())
    called = {"ran": False}

    @server.tool(patient_scope="patient")
    async def patient_tool() -> str:
        called["ran"] = True
        return "ran"

    async with Client(server) as client:
        open_params = PatientContextOpenParams(
            patient={"reference": "Patient/test-0001"}, purpose_of_use="TREAT"
        )
        open_request = types.Request[PatientContextOpenParams, str](
            method=METHOD_PATIENT_CONTEXT_OPEN, params=open_params
        )
        with pytest.raises(MCPError) as excinfo:
            await client.session.send_request(open_request, _RawResult)
        _assert_not_authenticated(excinfo.value)

        for method in (METHOD_PATIENT_CONTEXT_GET, METHOD_PATIENT_CONTEXT_CLOSE):
            id_request = types.Request[PatientContextIdParams, str](
                method=method, params=PatientContextIdParams(context_id="pc_x")
            )
            with pytest.raises(MCPError) as excinfo:
                await client.session.send_request(id_request, _RawResult)
            _assert_not_authenticated(excinfo.value)

        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("patient_tool", {})
        _assert_not_authenticated(excinfo.value)

    assert called["ran"] is False


def test_no_auth_patient_context_warns_once_at_construction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="hmcp.server.hmcp_server"):
        HMCPServer("t", patient_context=PatientContextConfig())

    matching = [
        record.getMessage()
        for record in caplog.records
        if _NO_AUTH_WARNING_MARKER in record.getMessage()
    ]
    assert len(matching) == 1
    assert "allow_unauthenticated" in matching[0]
    assert "token_verifier" in matching[0]


def test_no_auth_patient_context_warning_suppressed_by_auth(
    caplog: pytest.LogCaptureFixture,
) -> None:
    auth = AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url=_RESOURCE,
        validate_token_resource=True,
    )
    with caplog.at_level(logging.WARNING, logger="hmcp.server.hmcp_server"):
        HMCPServer(
            "t",
            patient_context=PatientContextConfig(),
            token_verifier=_Verifier(),
            auth=auth,
        )

    assert not any(
        _NO_AUTH_WARNING_MARKER in record.getMessage() for record in caplog.records
    )


def test_no_auth_patient_context_warning_suppressed_by_allow_unauthenticated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="hmcp.server.hmcp_server"):
        HMCPServer(
            "t", patient_context=PatientContextConfig(allow_unauthenticated=True)
        )

    assert not any(
        _NO_AUTH_WARNING_MARKER in record.getMessage() for record in caplog.records
    )
