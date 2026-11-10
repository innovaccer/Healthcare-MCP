"""Tests for the HMCP patient-context extension.

Covers the extension methods, the ``tools/call`` interceptor, the lifecycle
API and the audit trail. Synthetic values only.
"""

from __future__ import annotations

import json
import secrets
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx2
import mcp.types as types
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.extension import Extension
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from hmcp.server.patient_context.audit import AuditEvent
from hmcp.server.patient_context.extension import (
    PatientContextConfig,
    PatientContextExtension,
    PatientContextIdParams,
    PatientContextOpenParams,
)
from hmcp.server.patient_context.policy import AccessDecision
from hmcp.server.patient_context.store import InMemoryContextStore
from hmcp.shared.constants import (
    CONTEXT_ID_PREFIX,
    GUARDRAILS_EXTENSION,
    META_CONTEXT_ID,
    METHOD_PATIENT_CONTEXT_CLOSE,
    METHOD_PATIENT_CONTEXT_GET,
    METHOD_PATIENT_CONTEXT_OPEN,
    PATIENT_CONTEXT_EXTENSION,
    PATIENT_CONTEXT_REVISION,
)
from hmcp.shared.errors import HMCPErrorCode
from hmcp.shared.types import (
    ContextStatus,
    Identifier,
    PatientContext,
    PatientRef,
    Principal,
    current_patient_context,
)

PATIENT_REF = "Patient/test-0001"
ID_VALUE = "id-value-abc987"
GUARDRAIL_ID = "com.innovaccer.hmcp.guardrails/phi-redaction"
SENTINEL = "PHI-SECRET-patient-test-0001"

UNVERIFIED = Identifier(system="http://example.org/mrn", value=ID_VALUE)
VERIFIED = Identifier(system="http://verified.example.org", value="verified-1")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class RecordingAuditSink:
    """In-memory AuditSink that can be told to fail on one outcome."""

    def __init__(
        self, *, raise_on: str | None = None, raise_message: str = "audit sink down"
    ) -> None:
        self.events: list[AuditEvent] = []
        self._raise_on = raise_on
        self._raise_message = raise_message

    async def emit(self, event: AuditEvent) -> None:
        if self._raise_on is not None and event.outcome == self._raise_on:
            raise RuntimeError(self._raise_message)
        self.events.append(event)

    def tool_events(self) -> list[AuditEvent]:
        return [e for e in self.events if e.action == "tool.call"]


class FailingStore:
    """ContextStore wrapper whose chosen method raises a sentinel error."""

    def __init__(self, inner: InMemoryContextStore, *, fail_on: str) -> None:
        self._inner = inner
        self._fail_on = fail_on

    async def put(self, ctx: PatientContext) -> None:
        if self._fail_on == "put":
            raise RuntimeError(f"store backend saw {SENTINEL}")
        await self._inner.put(ctx)

    async def get(self, context_id: str) -> PatientContext | None:
        if self._fail_on == "get":
            raise RuntimeError(f"store backend saw {SENTINEL}")
        return await self._inner.get(context_id)

    async def set_status(
        self, context_id: str, status: ContextStatus
    ) -> PatientContext | None:
        if self._fail_on == "set_status":
            raise RuntimeError(f"store backend saw {SENTINEL}")
        return await self._inner.set_status(context_id, status)

    async def find(self, **kwargs: Any) -> list[PatientContext]:
        return await self._inner.find(**kwargs)

    async def purge_expired(self, now: datetime) -> int:
        return await self._inner.purge_expired(now)


class StubPolicy:
    """Configurable AccessPolicy stub."""

    def __init__(
        self,
        *,
        open_decision: AccessDecision | None = None,
        population_decision: AccessDecision | None = None,
        open_error: Exception | None = None,
        population_error: Exception | None = None,
    ) -> None:
        self.open_decision = open_decision or AccessDecision(allowed=True, reason="ok")
        self.population_decision = population_decision or AccessDecision(
            allowed=True, reason="ok"
        )
        self.open_error = open_error
        self.population_error = population_error
        self.open_calls = 0

    async def authorize_open(self, *, principal, token, patient, purpose_of_use):
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error
        return self.open_decision

    async def authorize_population(self, *, principal, token, tool_name):
        if self.population_error is not None:
            raise self.population_error
        return self.population_decision


class StampingExtension(Extension):
    """Inner extension that stamps a guardrail report onto the result _meta."""

    identifier = GUARDRAILS_EXTENSION

    def settings(self) -> dict[str, Any]:
        return {"revision": "1"}

    async def intercept_tool_call(self, params, ctx, call_next):
        result = await call_next(ctx)
        if isinstance(result, types.CallToolResult):
            meta = dict(result.meta or {})
            meta[GUARDRAILS_EXTENSION] = {
                "report": [{"id": GUARDRAIL_ID, "verdict": "redacted"}]
            }
            return result.model_copy(update={"meta": meta})
        return result


class OpenResult(BaseModel):
    model_config = {"extra": "allow"}

    contextId: str
    status: str
    expiresAt: str


class GetResult(BaseModel):
    model_config = {"extra": "allow"}

    contextId: str
    status: str
    expiresAt: str
    patient: dict[str, Any]
    encounter: str | None = None
    purposeOfUse: str


class CloseResult(BaseModel):
    model_config = {"extra": "allow"}

    contextId: str
    status: str


SCOPES = {
    "get_meds": "patient",
    "get_meds_sync": "patient",
    "pop_tool": "population",
    "plain_tool": "none",
}


def make_config(
    *,
    policy: Any | None = None,
    audit: RecordingAuditSink | None = None,
    ttl: int = 3600,
    allow_unauthenticated: bool = True,
) -> PatientContextConfig:
    return PatientContextConfig(
        policy=policy or StubPolicy(),
        audit=audit or RecordingAuditSink(),
        context_ttl_seconds=ttl,
        allow_unauthenticated=allow_unauthenticated,
    )


def build_server(config: PatientContextConfig, extra_extensions: tuple = ()):
    extension = PatientContextExtension(
        config, scope_for=lambda name: SCOPES.get(name, "none")
    )
    server = MCPServer("t", extensions=[extension, *extra_extensions])
    called = {"get_meds": 0, "sync": 0, "pop": 0, "plain": 0}

    @server.tool()
    async def get_meds() -> str:
        called["get_meds"] += 1
        context = current_patient_context()
        return context.context_id if context else "none"

    @server.tool()
    def get_meds_sync() -> str:
        called["sync"] += 1
        context = current_patient_context()
        return context.context_id if context else "none"

    @server.tool()
    async def pop_tool() -> str:
        called["pop"] += 1
        return "pop"

    @server.tool()
    async def plain_tool() -> str:
        called["plain"] += 1
        return "plain"

    return server, extension, called


def ctx_meta(context_id: str) -> dict[str, Any]:
    return {PATIENT_CONTEXT_EXTENSION: {META_CONTEXT_ID: context_id}}


async def inject_context(
    store: Any,
    *,
    reference: str = PATIENT_REF,
    status: ContextStatus = ContextStatus.ACTIVE,
    principal: Principal | None = None,
    expires_at: datetime | None = None,
) -> PatientContext:
    now = datetime.now(timezone.utc)
    context = PatientContext(
        context_id=CONTEXT_ID_PREFIX + secrets.token_urlsafe(32),
        patient=PatientRef(reference=reference),
        purpose_of_use="TREAT",
        principal=principal or Principal.anonymous(),
        status=status,
        created_at=now,
        expires_at=expires_at or now + timedelta(seconds=3600),
    )
    await store.put(context)
    return context


def _assert_clean(error: MCPError) -> None:
    message = str(error.message)
    data_text = json.dumps(error.data, default=str)
    for secret in (PATIENT_REF, ID_VALUE):
        assert secret not in message
        assert secret not in data_text


async def expect_error(awaitable) -> MCPError:
    with pytest.raises(MCPError) as excinfo:
        await awaitable
    _assert_clean(excinfo.value)
    return excinfo.value


def assert_clean_chain(error: MCPError) -> None:
    """The mapped error must not chain the backend exception that caused it."""
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(traceback.format_exception(error))
    assert SENTINEL not in rendered


async def open_context(
    client: Client,
    *,
    reference: str = PATIENT_REF,
    identifiers: tuple[Identifier, ...] = (UNVERIFIED,),
    purpose: str = "TREAT",
) -> OpenResult:
    params = PatientContextOpenParams(
        patient=PatientRef(reference=reference, identifiers=identifiers),
        purpose_of_use=purpose,
    )
    request = types.Request[PatientContextOpenParams, str](
        method=METHOD_PATIENT_CONTEXT_OPEN, params=params
    )
    return await client.session.send_request(request, OpenResult)


async def open_raw(client: Client, payload: dict[str, Any]) -> OpenResult:
    """Send open params as a raw dict so client-side validation is bypassed."""
    request = types.Request[dict[str, Any], str](
        method=METHOD_PATIENT_CONTEXT_OPEN, params=payload
    )
    return await client.session.send_request(request, OpenResult)


async def send_get(client: Client, context_id: str) -> GetResult:
    params = PatientContextIdParams(context_id=context_id)
    request = types.Request[PatientContextIdParams, str](
        method=METHOD_PATIENT_CONTEXT_GET, params=params
    )
    return await client.session.send_request(request, GetResult)


async def send_close(client: Client, context_id: str) -> CloseResult:
    params = PatientContextIdParams(context_id=context_id)
    request = types.Request[PatientContextIdParams, str](
        method=METHOD_PATIENT_CONTEXT_CLOSE, params=params
    )
    return await client.session.send_request(request, CloseResult)


# --------------------------------------------------------------------------- #
# open / get / close methods
# --------------------------------------------------------------------------- #


async def test_open_result_shape_and_identifier_drop() -> None:
    policy = StubPolicy(
        open_decision=AccessDecision(
            allowed=True, reason="ok", verified_identifiers=(VERIFIED,)
        )
    )
    config = make_config(policy=policy)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        assert opened.contextId.startswith(CONTEXT_ID_PREFIX)
        assert opened.status == ContextStatus.ACTIVE.value
        assert datetime.fromisoformat(opened.expiresAt).tzinfo is not None

        fetched = await send_get(client, opened.contextId)
        assert fetched.contextId == opened.contextId
        assert fetched.patient["reference"] == PATIENT_REF
        assert fetched.patient["identifiers"] == [
            {"system": VERIFIED.system, "value": VERIFIED.value}
        ]
        assert ID_VALUE not in json.dumps(fetched.patient)


async def test_open_policy_deny_is_31005_and_tool_not_involved() -> None:
    policy = StubPolicy(
        open_decision=AccessDecision(allowed=False, reason="token_patient_mismatch")
    )
    config = make_config(policy=policy)
    server, _extension, called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(open_context(client))
        assert error.code == int(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)
    assert called == {"get_meds": 0, "sync": 0, "pop": 0, "plain": 0}


async def test_open_policy_raises_is_31006() -> None:
    policy = StubPolicy(open_error=RuntimeError("policy exploded"))
    config = make_config(policy=policy)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(open_context(client))
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)


async def test_close_method_returns_closed() -> None:
    config = make_config()
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        closed = await send_close(client, opened.contextId)
        assert closed.contextId == opened.contextId
        assert closed.status == ContextStatus.CLOSED.value


# --------------------------------------------------------------------------- #
# patient-scoped tool calls
# --------------------------------------------------------------------------- #


async def test_patient_tool_requires_context_and_unknown_id() -> None:
    config = make_config()
    server, _extension, called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(client.call_tool("get_meds", {}))
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)
        assert called["get_meds"] == 0

        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta("pc_unknown"))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)
        assert called["get_meds"] == 0


async def test_patient_tool_binds_context_async_and_sync() -> None:
    config = make_config()
    server, _extension, called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)

        result = await client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))
        assert result.content[0].text == opened.contextId
        assert called["get_meds"] == 1

        sync_result = await client.call_tool(
            "get_meds_sync", {}, meta=ctx_meta(opened.contextId)
        )
        assert sync_result.content[0].text == opened.contextId
        assert called["sync"] == 1


async def test_patient_tool_suspended_revoked_closed_expired() -> None:
    config = make_config()
    store = config.store
    server, extension, _called = build_server(config)

    async with Client(server) as client:
        # suspended
        suspended = await open_context(client)
        assert await extension.suspend(context_id=suspended.contextId) == 1
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(suspended.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_SUSPENDED)

        # revoked
        revoked = await open_context(client)
        assert await extension.revoke(context_id=revoked.contextId) == 1
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(revoked.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REVOKED)

        # closed
        closed = await open_context(client)
        await send_close(client, closed.contextId)
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(closed.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)

        # expired (ttl elapsed, set directly in the store)
        expired = await open_context(client)
        stored = await store.get(expired.contextId)
        assert stored is not None
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        await store.put(stored.model_copy(update={"expires_at": past}))
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(expired.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)
        after = await store.get(expired.contextId)
        assert after is not None
        assert after.status is ContextStatus.EXPIRED


async def test_suspend_resume_and_resume_does_not_reactivate_revoked() -> None:
    config = make_config()
    server, extension, called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        assert await extension.suspend(context_id=opened.contextId) == 1
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_SUSPENDED)

        assert await extension.resume(context_id=opened.contextId) == 1
        result = await client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))
        assert result.content[0].text == opened.contextId

        revoked = await open_context(client)
        assert await extension.revoke(context_id=revoked.contextId) == 1
        assert await extension.resume(context_id=revoked.contextId) == 0
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(revoked.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REVOKED)
        assert called["get_meds"] == 1


async def test_revoke_by_patient_reference_counts() -> None:
    config = make_config()
    server, extension, _called = build_server(config)

    async with Client(server) as client:
        first = await open_context(client)
        second = await open_context(client)
        assert await extension.revoke(patient_reference=PATIENT_REF) == 2
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(first.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REVOKED)
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(second.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REVOKED)


# --------------------------------------------------------------------------- #
# scope none / population
# --------------------------------------------------------------------------- #


async def test_scope_none_runs_without_context_or_audit() -> None:
    sink = RecordingAuditSink()
    config = make_config(audit=sink, allow_unauthenticated=False)
    server, _extension, called = build_server(config)

    async with Client(server) as client:
        result = await client.call_tool("plain_tool", {})
        assert result.content[0].text == "plain"
        assert called["plain"] == 1
    assert sink.events == []


async def test_population_allowed_and_denied() -> None:
    allowed_sink = RecordingAuditSink()
    allowed_config = make_config(
        policy=StubPolicy(
            population_decision=AccessDecision(allowed=True, reason="population_scope")
        ),
        audit=allowed_sink,
    )
    allowed_server, _extension, allowed_called = build_server(allowed_config)

    async with Client(allowed_server) as client:
        result = await client.call_tool("pop_tool", {})
        assert result.content[0].text == "pop"
        assert allowed_called["pop"] == 1
    assert [e.outcome for e in allowed_sink.tool_events()] == ["attempt", "success"]

    denied_sink = RecordingAuditSink()
    denied_config = make_config(
        policy=StubPolicy(
            population_decision=AccessDecision(
                allowed=False, reason="single_patient_token"
            )
        ),
        audit=denied_sink,
    )
    denied_server, _extension, denied_called = build_server(denied_config)

    async with Client(denied_server) as client:
        error = await expect_error(client.call_tool("pop_tool", {}))
        assert error.code == int(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)
        assert denied_called["pop"] == 0
    denied_events = denied_sink.tool_events()
    assert denied_events[-1].outcome == "denied"
    assert denied_events[-1].error_code == int(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #


async def test_audit_success_emits_attempt_then_success() -> None:
    sink = RecordingAuditSink()
    config = make_config(audit=sink)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        await client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))

    assert [e.outcome for e in sink.tool_events()] == ["attempt", "success"]


async def test_audit_denial_has_error_code_before_error() -> None:
    sink = RecordingAuditSink()
    config = make_config(audit=sink)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta("pc_unknown"))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)

    events = sink.tool_events()
    assert events[-1].outcome == "denied"
    assert events[-1].error_code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)


async def test_audit_sink_raises_on_tool_attempt_is_31006_and_tool_not_called() -> None:
    sink = RecordingAuditSink(raise_on="attempt")
    config = make_config(audit=sink)
    server, _extension, called = build_server(config)
    context = await inject_context(config.store)

    async with Client(server) as client:
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(context.context_id))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert called["get_meds"] == 0


async def test_guardrails_parsed_from_result_meta() -> None:
    sink = RecordingAuditSink()
    config = make_config(audit=sink)
    server, _extension, _called = build_server(
        config, extra_extensions=(StampingExtension(),)
    )

    async with Client(server) as client:
        opened = await open_context(client)
        await client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))

    success = [e for e in sink.tool_events() if e.outcome == "success"]
    assert len(success) == 1
    assert success[0].guardrails == ((GUARDRAIL_ID, "redacted"),)


@pytest.mark.parametrize(
    "bad_guardrails",
    [
        "not-a-dict",
        {"report": "not-a-list"},
        {"report": [123, "not-an-entry"]},
    ],
)
async def test_malformed_guardrails_meta_does_not_crash_and_emits_success(
    bad_guardrails: Any,
) -> None:
    sink = RecordingAuditSink()
    config = make_config(audit=sink)
    extension = PatientContextExtension(config, scope_for=lambda name: "patient")
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def bad_meta() -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="tool-ran")],
            meta={GUARDRAILS_EXTENSION: bad_guardrails},
        )

    context = await inject_context(config.store)

    async with Client(server) as client:
        result = await client.call_tool(
            "bad_meta", {}, meta=ctx_meta(context.context_id)
        )

    assert result.content[0].text == "tool-ran"
    assert [e.outcome for e in sink.tool_events()] == ["attempt", "success"]


# --------------------------------------------------------------------------- #
# mapped errors must not chain the backend exception (PHI may ride in it)
# --------------------------------------------------------------------------- #


async def test_open_policy_failure_error_has_no_chained_exception() -> None:
    policy = StubPolicy(open_error=RuntimeError(f"policy backend saw {SENTINEL}"))
    config = make_config(policy=policy)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        with pytest.raises(MCPError) as excinfo:
            await open_context(client)

    assert excinfo.value.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert excinfo.value.message == "Patient context is unavailable"
    assert_clean_chain(excinfo.value)


async def test_open_store_put_failure_error_has_no_chained_exception() -> None:
    config = PatientContextConfig(
        store=FailingStore(InMemoryContextStore(), fail_on="put"),
        policy=StubPolicy(),
        audit=RecordingAuditSink(),
        allow_unauthenticated=True,
    )
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        with pytest.raises(MCPError) as excinfo:
            await open_context(client)

    assert excinfo.value.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert_clean_chain(excinfo.value)


async def test_get_store_failure_error_has_no_chained_exception() -> None:
    config = PatientContextConfig(
        store=FailingStore(InMemoryContextStore(), fail_on="get"),
        policy=StubPolicy(),
        audit=RecordingAuditSink(),
        allow_unauthenticated=True,
    )
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        with pytest.raises(MCPError) as excinfo:
            await send_get(client, "pc_unknown")

    assert excinfo.value.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert_clean_chain(excinfo.value)


async def test_close_store_failure_error_has_no_chained_exception() -> None:
    config = PatientContextConfig(
        store=FailingStore(InMemoryContextStore(), fail_on="set_status"),
        policy=StubPolicy(),
        audit=RecordingAuditSink(),
        allow_unauthenticated=True,
    )
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        with pytest.raises(MCPError) as excinfo:
            await send_close(client, opened.contextId)

    assert excinfo.value.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert_clean_chain(excinfo.value)


async def test_open_audit_failure_error_has_no_chained_exception() -> None:
    sink = RecordingAuditSink(
        raise_on="attempt", raise_message=f"audit sink saw {SENTINEL}"
    )
    config = make_config(audit=sink)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        with pytest.raises(MCPError) as excinfo:
            await open_context(client)

    assert excinfo.value.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert_clean_chain(excinfo.value)


async def test_tool_audit_failure_error_has_no_chained_exception() -> None:
    sink = RecordingAuditSink(
        raise_on="attempt", raise_message=f"audit sink saw {SENTINEL}"
    )
    config = make_config(audit=sink)
    server, _extension, _called = build_server(config)
    context = await inject_context(config.store)

    async with Client(server) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("get_meds", {}, meta=ctx_meta(context.context_id))

    assert excinfo.value.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert_clean_chain(excinfo.value)


async def test_population_policy_failure_error_has_no_chained_exception() -> None:
    policy = StubPolicy(population_error=RuntimeError(f"policy backend saw {SENTINEL}"))
    config = make_config(policy=policy)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("pop_tool", {})

    assert excinfo.value.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
    assert_clean_chain(excinfo.value)


# --------------------------------------------------------------------------- #
# principal binding over streamable-http
# --------------------------------------------------------------------------- #

BASE = "http://127.0.0.1:8000"
AUTH = AuthSettings(
    issuer_url="https://auth.example.com",
    resource_server_url=f"{BASE}/mcp",
    validate_token_resource=True,
)

TOKENS = {
    "tok-a": AccessToken(
        token="tok-a",
        client_id="app-1",
        scopes=["patient/*.rs"],
        subject="user-a",
        resource=f"{BASE}/mcp",
        claims={"iss": "https://auth.example.com", "patient": "test-0001"},
    ),
    "tok-b": AccessToken(
        token="tok-b",
        client_id="app-1",
        scopes=["patient/*.rs"],
        subject="user-b",
        resource=f"{BASE}/mcp",
        claims={"iss": "https://auth.example.com", "patient": "test-0002"},
    ),
}


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        return TOKENS.get(token)


def build_http_server(store: InMemoryContextStore):
    config = PatientContextConfig(store=store, allow_unauthenticated=False)
    extension = PatientContextExtension(
        config, scope_for=lambda name: SCOPES.get(name, "none")
    )
    server = MCPServer(
        "t",
        token_verifier=Verifier(),
        auth=AUTH,
        extensions=[extension],
    )

    @server.tool()
    async def get_meds() -> str:
        context = current_patient_context()
        return context.context_id if context else "none"

    return server, extension, config


async def _http_run(app, token: str, fn):
    headers = {"Authorization": f"Bearer {token}"}
    http_client = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url=BASE, headers=headers
    )
    async with Client(
        streamable_http_client(f"{BASE}/mcp", http_client=http_client)
    ) as client:
        return await fn(client)


async def test_http_principal_binding() -> None:
    store = InMemoryContextStore()
    server, _extension, _config = build_http_server(store)
    app = server.streamable_http_app()

    async with server.session_manager.run():
        opened = await _http_run(app, "tok-a", lambda c: open_context(c))
        assert opened.status == ContextStatus.ACTIVE.value

        error = await _http_run(
            app,
            "tok-b",
            lambda c: expect_error(
                c.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))
            ),
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)

        error = await _http_run(
            app,
            "tok-b",
            lambda c: expect_error(send_get(c, opened.contextId)),
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)


async def test_http_smart_claim_ceiling() -> None:
    store = InMemoryContextStore()
    server, _extension, _config = build_http_server(store)

    # Create the context directly (open would be denied by the SMART policy)
    # bound to token B's principal but for Patient/test-0001, while token B
    # carries the SMART patient claim "test-0002".
    principal_b = Principal.from_access_token(TOKENS["tok-b"])
    now = datetime.now(timezone.utc)
    context = PatientContext(
        context_id=CONTEXT_ID_PREFIX + secrets.token_urlsafe(32),
        patient=PatientRef(reference=PATIENT_REF),
        purpose_of_use="TREAT",
        principal=principal_b,
        status=ContextStatus.ACTIVE,
        created_at=now,
        expires_at=now + timedelta(seconds=3600),
    )
    await store.put(context)

    app = server.streamable_http_app()
    async with server.session_manager.run():
        error = await _http_run(
            app,
            "tok-b",
            lambda c: expect_error(
                c.call_tool("get_meds", {}, meta=ctx_meta(context.context_id))
            ),
        )
        assert error.code == int(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)


async def test_http_unknown_and_other_principal_are_indistinguishable() -> None:
    store = InMemoryContextStore()
    server, _extension, _config = build_http_server(store)
    app = server.streamable_http_app()

    async with server.session_manager.run():
        opened = await _http_run(app, "tok-a", lambda c: open_context(c))

        unknown_get = await _http_run(
            app, "tok-a", lambda c: expect_error(send_get(c, "pc_unknown"))
        )
        other_get = await _http_run(
            app, "tok-b", lambda c: expect_error(send_get(c, opened.contextId))
        )
        assert unknown_get.code == other_get.code
        assert unknown_get.message == other_get.message
        assert unknown_get.data == other_get.data

        unknown_close = await _http_run(
            app, "tok-a", lambda c: expect_error(send_close(c, "pc_unknown"))
        )
        other_close = await _http_run(
            app, "tok-b", lambda c: expect_error(send_close(c, opened.contextId))
        )
        assert unknown_close.code == other_close.code
        assert unknown_close.message == other_close.message
        assert unknown_close.data == other_close.data


# --------------------------------------------------------------------------- #
# lifecycle transition table regressions
# --------------------------------------------------------------------------- #


async def test_revoked_context_cannot_be_suspended_or_resumed() -> None:
    config = make_config()
    server, extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        assert await extension.revoke(context_id=opened.contextId) == 1
        assert await extension.suspend(context_id=opened.contextId) == 0
        assert await extension.resume(context_id=opened.contextId) == 0
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REVOKED)


async def test_closed_context_cannot_be_suspended_or_resumed() -> None:
    config = make_config()
    server, extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        await send_close(client, opened.contextId)
        assert await extension.suspend(context_id=opened.contextId) == 0
        assert await extension.resume(context_id=opened.contextId) == 0
        error = await expect_error(
            client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_INVALID)


async def test_suspend_expired_context_returns_zero() -> None:
    config = make_config()
    store = config.store
    server, extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        await store.set_status(opened.contextId, ContextStatus.EXPIRED)
        assert await extension.suspend(context_id=opened.contextId) == 0
        assert await extension.resume(context_id=opened.contextId) == 0


async def test_close_revoked_context_keeps_revoked() -> None:
    config = make_config()
    store = config.store
    server, extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        assert await extension.revoke(context_id=opened.contextId) == 1
        closed = await send_close(client, opened.contextId)
        assert closed.status == ContextStatus.REVOKED.value
        stored = await store.get(opened.contextId)
        assert stored is not None
        assert stored.status is ContextStatus.REVOKED


async def test_suspend_from_active_then_resume_still_works() -> None:
    config = make_config()
    server, extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        assert await extension.suspend(context_id=opened.contextId) == 1
        assert await extension.resume(context_id=opened.contextId) == 1
        result = await client.call_tool("get_meds", {}, meta=ctx_meta(opened.contextId))
        assert result.content[0].text == opened.contextId


# --------------------------------------------------------------------------- #
# input robustness: non-dict _meta
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_value", ["not-a-dict", ["not", "a", "dict"]])
async def test_non_dict_patient_context_meta_is_missing_context(bad_value) -> None:
    sink = RecordingAuditSink()
    config = make_config(audit=sink)
    server, _extension, called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(
            client.call_tool(
                "get_meds", {}, meta={PATIENT_CONTEXT_EXTENSION: bad_value}
            )
        )
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)
        assert called["get_meds"] == 0

    events = sink.tool_events()
    assert events[-1].outcome == "denied"
    assert events[-1].error_code == int(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)


# --------------------------------------------------------------------------- #
# audit emission rules: only the pre-tool / pre-put attempt is fail-closed
# --------------------------------------------------------------------------- #


async def test_audit_sink_raises_on_open_attempt_is_31006_and_nothing_stored() -> None:
    sink = RecordingAuditSink(raise_on="attempt")
    config = make_config(audit=sink)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(open_context(client))
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)

    assert await config.store.find(patient_reference=PATIENT_REF) == []


async def test_audit_sink_raises_on_denied_keeps_specific_code() -> None:
    sink = RecordingAuditSink(raise_on="denied")
    config = make_config(audit=sink)
    server, _extension, called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(client.call_tool("get_meds", {}))
        assert error.code == int(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)
    assert called["get_meds"] == 0


async def test_audit_sink_raises_on_success_keeps_result() -> None:
    sink = RecordingAuditSink(raise_on="success")
    config = make_config(audit=sink)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        opened = await open_context(client)
        assert opened.status == ContextStatus.ACTIVE.value
        fetched = await send_get(client, opened.contextId)
        assert fetched.contextId == opened.contextId


# --------------------------------------------------------------------------- #
# input robustness: open params patterns
# --------------------------------------------------------------------------- #


def _open_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "patient": {
            "reference": PATIENT_REF,
            "identifiers": [{"system": UNVERIFIED.system, "value": ID_VALUE}],
        },
        "purposeOfUse": "TREAT",
    }
    payload.update(overrides)
    return payload


async def test_open_rejects_bad_purpose_before_policy() -> None:
    policy = StubPolicy()
    sink = RecordingAuditSink()
    config = make_config(policy=policy, audit=sink)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(
            open_raw(client, _open_payload(purposeOfUse="treat"))
        )
        assert error.code == types.INVALID_PARAMS
        assert error.message == "Invalid request parameters"
        assert error.data == ""

    assert policy.open_calls == 0
    assert sink.events == []


async def test_open_rejects_bad_encounter_before_policy() -> None:
    policy = StubPolicy()
    sink = RecordingAuditSink()
    config = make_config(policy=policy, audit=sink)
    server, _extension, _called = build_server(config)

    async with Client(server) as client:
        error = await expect_error(
            open_raw(client, _open_payload(encounter="Encounter/<x>"))
        )
        assert error.code == types.INVALID_PARAMS
        assert error.message == "Invalid request parameters"
        assert error.data == ""

    assert policy.open_calls == 0
    assert sink.events == []


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #


def test_config_rejects_invalid_default_tool_scope() -> None:
    with pytest.raises(ValueError):
        PatientContextConfig(default_tool_scope="bogus")


@pytest.mark.parametrize("scope", ["patient", "population", "none"])
def test_config_accepts_valid_default_tool_scope(scope) -> None:
    config = PatientContextConfig(default_tool_scope=scope)
    assert config.default_tool_scope == scope


# --------------------------------------------------------------------------- #
# advertisement: settings as seen by a client
# --------------------------------------------------------------------------- #


async def test_settings_advertised_to_client_default_and_non_default() -> None:
    default_extension = PatientContextExtension(
        make_config(), scope_for=lambda name: "none"
    )
    async with Client(MCPServer("t", extensions=[default_extension])) as client:
        default_advertised = client.server_capabilities.extensions[
            PATIENT_CONTEXT_EXTENSION
        ]
    assert default_advertised == {
        "revision": PATIENT_CONTEXT_REVISION,
        "contextTtlSeconds": 3600,
        "defaultToolScope": "patient",
    }

    config = PatientContextConfig(
        policy=StubPolicy(),
        audit=RecordingAuditSink(),
        context_ttl_seconds=120,
        default_tool_scope="population",
        allow_unauthenticated=True,
    )
    non_default_extension = PatientContextExtension(
        config, scope_for=lambda name: "none"
    )
    async with Client(MCPServer("t", extensions=[non_default_extension])) as client:
        non_default_advertised = client.server_capabilities.extensions[
            PATIENT_CONTEXT_EXTENSION
        ]
    assert non_default_advertised == {
        "revision": PATIENT_CONTEXT_REVISION,
        "contextTtlSeconds": 120,
        "defaultToolScope": "population",
    }
