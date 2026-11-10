"""Patient context integrated into HMCPServer.

Synthetic patients/ids only. Uses the in-memory ``Client`` with
``allow_unauthenticated=True`` and a permissive stub policy so a context can be
opened without a real token (the default SMART policy requires a patient-launch
token, which the in-memory transport cannot carry).
"""

from __future__ import annotations

from typing import Any

import mcp.types as types
import pytest
from mcp.client import Client
from mcp.server.extension import Extension
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from hmcp.server.hmcp_server import HMCPServer
from hmcp.server.patient_context import PatientContextConfig
from hmcp.server.patient_context.extension import PatientContextOpenParams
from hmcp.server.patient_context.policy import AccessDecision
from hmcp.shared.constants import (
    META_CONTEXT_ID,
    METHOD_PATIENT_CONTEXT_OPEN,
    PATIENT_CONTEXT_EXTENSION,
)
from hmcp.shared.errors import HMCPErrorCode
from hmcp.shared.types import current_patient_context

_PATIENT_REF = "Patient/test-0001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AllowAllPolicy:
    """Permissive policy so anonymous dev-mode callers may open a context."""

    async def authorize_open(self, *, principal, token, patient, purpose_of_use):
        return AccessDecision(allowed=True, reason="ok")

    async def authorize_population(self, *, principal, token, tool_name):
        return AccessDecision(allowed=True, reason="ok")


class _OpenResult(BaseModel):
    model_config = {"extra": "allow"}

    contextId: str
    status: str


def _config() -> PatientContextConfig:
    return PatientContextConfig(policy=_AllowAllPolicy(), allow_unauthenticated=True)


async def _open(client: Client, reference: str = _PATIENT_REF) -> _OpenResult:
    params = PatientContextOpenParams(
        patient={"reference": reference}, purpose_of_use="TREAT"
    )
    request = types.Request[PatientContextOpenParams, str](
        method=METHOD_PATIENT_CONTEXT_OPEN, params=params
    )
    return await client.session.send_request(request, _OpenResult)


def _context_meta(context_id: str) -> dict[str, Any]:
    return {PATIENT_CONTEXT_EXTENSION: {META_CONTEXT_ID: context_id}}


async def _expect_error(awaitable: Any) -> MCPError:
    with pytest.raises(MCPError) as excinfo:
        await awaitable
    return excinfo.value


async def _listed_meta(client: Client) -> dict[str, dict[str, Any]]:
    tools = await client.list_tools()
    return {tool.name: (tool.meta or {}) for tool in tools.tools}


# ---------------------------------------------------------------------------
# Advertising + enforcement
# ---------------------------------------------------------------------------


async def test_patient_scope_advertised_and_enforced() -> None:
    server = HMCPServer("t", patient_context=_config())

    @server.tool(patient_scope="patient")
    async def get_meds() -> str:
        context = current_patient_context()
        return context.context_id if context else "none"

    async with Client(server) as client:
        meta = await _listed_meta(client)
        assert meta["get_meds"][PATIENT_CONTEXT_EXTENSION] == {"scope": "patient"}

        error = await _expect_error(client.call_tool("get_meds", {}))
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)

        opened = await _open(client)
        result = await client.call_tool(
            "get_meds", {}, meta=_context_meta(opened.contextId)
        )
        assert result.content[0].text == opened.contextId


async def test_default_scope_is_patient_advertised_and_enforced() -> None:
    server = HMCPServer("t", patient_context=_config())

    @server.tool()
    async def defaulted() -> str:
        return "ran"

    async with Client(server) as client:
        meta = await _listed_meta(client)
        assert meta["defaulted"][PATIENT_CONTEXT_EXTENSION] == {"scope": "patient"}

        error = await _expect_error(client.call_tool("defaulted", {}))
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)


async def test_none_scope_runs_without_context() -> None:
    server = HMCPServer("t", patient_context=_config())

    @server.tool(patient_scope="none")
    async def plain() -> str:
        return "plain-ran"

    async with Client(server) as client:
        result = await client.call_tool("plain", {})
        assert result.content[0].text == "plain-ran"


async def test_patient_scope_without_patient_context_raises() -> None:
    server = HMCPServer("t")  # no patient_context configured

    with pytest.raises(ValueError):

        @server.tool(patient_scope="patient")
        async def get_meds() -> str:
            return "nope"


async def test_invalid_patient_scope_raises() -> None:
    server = HMCPServer("t", patient_context=_config())

    with pytest.raises(ValueError):

        @server.tool(patient_scope="patiant")  # type: ignore[arg-type]
        async def get_meds() -> str:
            return "nope"


def test_tools_kwarg_with_patient_context_raises() -> None:
    from mcp.server.mcpserver.tools import Tool

    def some_tool() -> str:
        return "x"

    with pytest.raises(ValueError):
        HMCPServer(
            "t",
            patient_context=_config(),
            tools=[Tool.from_function(some_tool)],
        )


async def test_advertised_scope_matches_enforced_scope() -> None:
    server = HMCPServer("t", patient_context=_config())

    @server.tool(patient_scope="patient")
    async def patient_tool() -> str:
        return "p"

    @server.tool(patient_scope="population")
    async def population_tool() -> str:
        return "pop"

    @server.tool(patient_scope="none")
    async def none_tool() -> str:
        return "n"

    async with Client(server) as client:
        meta = await _listed_meta(client)

    for name in ("patient_tool", "population_tool", "none_tool"):
        advertised = meta[name][PATIENT_CONTEXT_EXTENSION]["scope"]
        assert advertised == server._scope_for(name)


async def test_caller_meta_keys_survive_merge() -> None:
    server = HMCPServer("t", patient_context=_config())

    @server.tool(meta={"caller": "kept", "nested": {"a": 1}}, patient_scope="none")
    async def annotated() -> str:
        return "ok"

    async with Client(server) as client:
        meta = await _listed_meta(client)

    assert meta["annotated"]["caller"] == "kept"
    assert meta["annotated"]["nested"] == {"a": 1}
    assert meta["annotated"][PATIENT_CONTEXT_EXTENSION] == {"scope": "none"}


# ---------------------------------------------------------------------------
# Lifecycle: revoke
# ---------------------------------------------------------------------------


async def test_revoke_makes_next_call_fail() -> None:
    server = HMCPServer("t", patient_context=_config())

    @server.tool(patient_scope="patient")
    async def get_meds() -> str:
        return "ran"

    async with Client(server) as client:
        opened = await _open(client)
        meta = _context_meta(opened.contextId)
        assert (await client.call_tool("get_meds", {}, meta=meta)).content

        revoked = await server.patient_contexts.revoke(patient_reference=_PATIENT_REF)
        assert revoked == 1

        error = await _expect_error(client.call_tool("get_meds", {}, meta=meta))
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REVOKED)


# ---------------------------------------------------------------------------
# Extension ordering
# ---------------------------------------------------------------------------


class _ObservingExtension(Extension):
    """Inner extension that records whether it can see the patient context."""

    identifier = "com.example/hmcp-observing"

    def __init__(self) -> None:
        super().__init__()
        self.seen_context_id: str | None = None

    def settings(self) -> dict[str, Any]:
        return {"revision": "1"}

    async def intercept_tool_call(self, params, ctx, call_next):
        context = current_patient_context()
        self.seen_context_id = context.context_id if context else None
        return await call_next(ctx)


async def test_extra_extension_runs_inside_patient_context() -> None:
    observer = _ObservingExtension()
    server = HMCPServer("t", patient_context=_config(), extensions=[observer])

    @server.tool(patient_scope="patient")
    async def get_meds() -> str:
        return "ran"

    async with Client(server) as client:
        opened = await _open(client)
        await client.call_tool("get_meds", {}, meta=_context_meta(opened.contextId))

    assert observer.seen_context_id == opened.contextId
