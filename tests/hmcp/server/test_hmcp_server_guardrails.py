"""I1b tests: guardrails integrated into HMCPServer.

Synthetic patients/ids only. Uses the in-memory ``Client`` with
``allow_unauthenticated=True`` and a permissive stub policy so a context can be
opened without a real token. The prompt evaluator LLM is a fake (no network).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp.types as types
import pytest
from mcp.client import Client
from mcp.server.extension import Extension
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from hmcp.server.guardrails import GuardrailsConfig
from hmcp.server.hmcp_server import HMCPServer
from hmcp.server.patient_context import PatientContextConfig
from hmcp.server.patient_context.extension import PatientContextOpenParams
from hmcp.server.patient_context.policy import AccessDecision
from hmcp.shared.constants import (
    GUARDRAILS_EXTENSION,
    META_CONTEXT_ID,
    META_REPORT,
    METHOD_PATIENT_CONTEXT_OPEN,
    PATIENT_CONTEXT_EXTENSION,
)
from hmcp.shared.errors import HMCPErrorCode
from hmcp.shared.guardrails import (
    PATIENT_CONTEXT_CONSISTENCY_ID,
    PHI_DEIDENTIFICATION_ID,
    PHI_REDACTION_ID,
    REDACTION_PLACEHOLDER_WRITEBACK_ID,
    patient_context_consistency,
    phi_deidentification,
    phi_redaction,
    redaction_placeholder_writeback,
)
from hmcp.shared.guardrails.definition import (
    ActionSpec,
    GuardrailDefinition,
    IdentifierType,
    PatternEvaluatorSpec,
    PhiCategory,
)
from hmcp.shared.types import current_patient_context

_REFERENCE = "Patient/test-0001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AllowAllPolicy:
    async def authorize_open(self, *, principal, token, patient, purpose_of_use):
        return AccessDecision(allowed=True, reason="ok")

    async def authorize_population(self, *, principal, token, tool_name):
        return AccessDecision(allowed=True, reason="ok")


class _RecordingPolicy:
    """Records whether ``authorize_population`` was consulted; verdict configurable."""

    def __init__(self, *, population_allowed: bool) -> None:
        self.population_allowed = population_allowed
        self.population_calls = 0

    async def authorize_open(self, *, principal, token, patient, purpose_of_use):
        return AccessDecision(allowed=True, reason="ok")

    async def authorize_population(self, *, principal, token, tool_name):
        self.population_calls += 1
        return AccessDecision(
            allowed=self.population_allowed,
            reason="ok" if self.population_allowed else "denied",
        )


class _ReportObserver(Extension):
    """Caller extension that records the ``_meta`` it gets back from ``call_next``.

    Ordering guard: guardrails run OUTSIDE caller extensions, so the result a
    caller extension receives from ``call_next`` must NOT yet carry the guardrails
    report. If the guardrails extension were ordered after the caller's, this
    observer would see the report and the test would fail.
    """

    identifier = "com.example/hmcp-report-observer"

    def __init__(self) -> None:
        super().__init__()
        self.saw_report_inner: bool | None = None

    def settings(self) -> dict[str, Any]:
        return {"revision": "1"}

    async def intercept_tool_call(self, params, ctx, call_next):
        result = await call_next(ctx)
        meta = getattr(result, "meta", None) or {}
        self.saw_report_inner = GUARDRAILS_EXTENSION in meta
        return result


class _FakeLLM:
    """Fake prompt-evaluator LLM: reports the synthetic name, records the context."""

    def __init__(self) -> None:
        self.seen_context_ids: list[str | None] = []

    async def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        context = current_patient_context()
        self.seen_context_ids.append(context.context_id if context else None)
        findings: list[dict[str, str]] = []
        if "Testpatient Alpha" in user:
            findings.append({"text": "Testpatient Alpha", "category": "NAME"})
        return json.dumps({"findings": findings})


class _SleepingMarkerVault:
    """A vault whose ``marker_for`` outlives any short guardrail timeout."""

    async def marker_for(self, context_id, expires_at, category, value):
        await asyncio.sleep(1.0)
        return "[NAME_1]"

    async def resolve(self, context_id, marker):
        return "Testpatient Alpha"


class _OpenResult(BaseModel):
    model_config = {"extra": "allow"}

    contextId: str
    status: str


def _build_server(
    *,
    policy: Any | None = None,
    extensions: tuple[Extension, ...] = (),
    guardrails: bool = True,
) -> tuple[HMCPServer, _FakeLLM, dict[str, Any]]:
    llm = _FakeLLM()
    config = (
        GuardrailsConfig(
            definitions=[
                phi_deidentification(),
                patient_context_consistency(),
                redaction_placeholder_writeback(),
            ],
            mandatory=[
                PHI_DEIDENTIFICATION_ID,
                PATIENT_CONTEXT_CONSISTENCY_ID,
                REDACTION_PLACEHOLDER_WRITEBACK_ID,
            ],
            llm=llm,
        )
        if guardrails
        else None
    )
    server = HMCPServer(
        "t",
        patient_context=PatientContextConfig(
            policy=policy or _AllowAllPolicy(), allow_unauthenticated=True
        ),
        guardrails=config,
        extensions=list(extensions) or None,
    )
    recorded: dict[str, Any] = {}

    @server.tool(patient_scope="patient")
    async def read() -> str:
        return "Testpatient Alpha, SSN 900-12-3456"

    @server.tool(patient_scope="patient")
    async def write(note: str) -> str:
        recorded["note"] = note
        recorded["write_calls"] = recorded.get("write_calls", 0) + 1
        return "written"

    @server.tool(patient_scope="patient")
    async def patient_tool(reference: str) -> str:
        return f"Patient/{reference}"

    @server.tool(patient_scope="population")
    async def population_tool() -> str:
        return "pop"

    return server, llm, recorded


async def _open(client: Client, reference: str = _REFERENCE) -> _OpenResult:
    params = PatientContextOpenParams(
        patient={"reference": reference}, purpose_of_use="TREAT"
    )
    request = types.Request[PatientContextOpenParams, str](
        method=METHOD_PATIENT_CONTEXT_OPEN, params=params
    )
    return await client.session.send_request(request, _OpenResult)


def _context_meta(context_id: str) -> dict[str, Any]:
    return {PATIENT_CONTEXT_EXTENSION: {META_CONTEXT_ID: context_id}}


def _verdicts(result: types.CallToolResult, guardrail_id: str) -> list[str]:
    report = (result.meta or {}).get(GUARDRAILS_EXTENSION, {}).get(META_REPORT) or []
    return [
        entry["verdict"]
        for entry in report
        if entry.get("id") == guardrail_id and "verdict" in entry
    ]


async def _expect_error(awaitable: Any) -> MCPError:
    with pytest.raises(MCPError) as excinfo:
        await awaitable
    return excinfo.value


# ---------------------------------------------------------------------------
# phi_deidentification (mandatory)
# ---------------------------------------------------------------------------


async def test_deidentification_pseudonymizes_read_and_reports() -> None:
    server, _deid, _recorded = _build_server()

    async with Client(server) as client:
        context = await _open(client)
        result = await client.call_tool(
            "read", {}, meta=_context_meta(context.contextId)
        )

    text = result.content[0].text
    assert "[NAME_1]" in text
    assert "[SSN_1]" in text
    assert "Testpatient Alpha" not in text
    assert "900-12-3456" not in text
    assert "pseudonymized" in _verdicts(result, PHI_DEIDENTIFICATION_ID)


async def test_deidentification_reidentifies_in_same_context() -> None:
    server, _deid, recorded = _build_server()

    async with Client(server) as client:
        context = await _open(client)
        await client.call_tool("read", {}, meta=_context_meta(context.contextId))
        result = await client.call_tool(
            "write",
            {"note": "note for [NAME_1]"},
            meta=_context_meta(context.contextId),
        )

    assert not result.is_error
    assert recorded["note"] == "note for Testpatient Alpha"


async def test_deidentification_reidentifies_bare_marker_in_same_context() -> None:
    server, _llm, recorded = _build_server()

    async with Client(server) as client:
        context = await _open(client)
        await client.call_tool("read", {}, meta=_context_meta(context.contextId))
        result = await client.call_tool(
            "write", {"note": "Hello NAME_1"}, meta=_context_meta(context.contextId)
        )

    assert not result.is_error
    assert recorded["note"] == "Hello Testpatient Alpha"
    assert "reidentified" in _verdicts(result, PHI_DEIDENTIFICATION_ID)


async def test_output_key_collision_fails_closed() -> None:
    server = HMCPServer(
        "t",
        patient_context=PatientContextConfig(
            policy=_AllowAllPolicy(), allow_unauthenticated=True
        ),
        guardrails=GuardrailsConfig(
            definitions=[phi_redaction()],
            mandatory=[PHI_REDACTION_ID],
            llm=_FakeLLM(),
        ),
    )

    @server.tool(patient_scope="patient")
    async def collide() -> str:
        return '{"900-12-3456": 1, "900-12-3457": 2}'

    async with Client(server) as client:
        context = await _open(client)
        result = await client.call_tool(
            "collide", {}, meta=_context_meta(context.contextId)
        )

    assert result.is_error
    text = "".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    assert "900-12-3456" not in text
    assert "900-12-3457" not in text


async def test_output_pseudonymize_timeout_fails_closed() -> None:
    definition = GuardrailDefinition(
        id="com.example.test/slow-pseudonymize",
        version="1.0.0",
        title="slow pseudonymize guardrail",
        stages=("output", "input"),
        evaluator=PatternEvaluatorSpec(
            kind="pattern",
            identifier_types=(
                IdentifierType(
                    name="digits",
                    display_name="Digits",
                    category=PhiCategory.OTHER_ID,
                    text_patterns=(r"(\d+)",),
                ),
            ),
        ),
        action=ActionSpec(on_finding="pseudonymize"),
        timeout_ms=100,
    )
    server = HMCPServer(
        "t",
        patient_context=PatientContextConfig(
            policy=_AllowAllPolicy(), allow_unauthenticated=True
        ),
        guardrails=GuardrailsConfig(
            definitions=[definition],
            mandatory=[definition.id],
            vault=_SleepingMarkerVault(),
        ),
    )

    @server.tool(patient_scope="patient")
    async def slow() -> str:
        return "123456"

    async with Client(server) as client:
        context = await _open(client)
        result = await client.call_tool(
            "slow", {}, meta=_context_meta(context.contextId)
        )

    assert result.is_error
    text = "".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    assert "123456" not in text


async def test_deidentification_unknown_marker_is_blocked() -> None:
    server, _deid, recorded = _build_server()

    async with Client(server) as client:
        context = await _open(client)
        await client.call_tool("read", {}, meta=_context_meta(context.contextId))
        result = await client.call_tool(
            "write",
            {"note": "note for [NAME_7]"},
            meta=_context_meta(context.contextId),
        )

    assert result.is_error
    assert "blocked" in _verdicts(result, PHI_DEIDENTIFICATION_ID)
    assert "write_calls" not in recorded


async def test_deidentification_marker_does_not_cross_contexts() -> None:
    server, _deid, recorded = _build_server()

    async with Client(server) as client:
        first = await _open(client, "Patient/test-0001")
        await client.call_tool("read", {}, meta=_context_meta(first.contextId))
        second = await _open(client, "Patient/test-0002")
        result = await client.call_tool(
            "write",
            {"note": "note for [NAME_1]"},
            meta=_context_meta(second.contextId),
        )

    assert result.is_error
    assert "write_calls" not in recorded


# ---------------------------------------------------------------------------
# patient_context_consistency (mandatory)
# ---------------------------------------------------------------------------


async def test_patient_context_consistency_blocks_mismatch() -> None:
    server, _deid, _recorded = _build_server()

    async with Client(server) as client:
        context = await _open(client, "Patient/test-0001")
        mismatch = await client.call_tool(
            "patient_tool",
            {"reference": "test-0002"},
            meta=_context_meta(context.contextId),
        )
        match = await client.call_tool(
            "patient_tool",
            {"reference": "test-0001"},
            meta=_context_meta(context.contextId),
        )

    assert mismatch.is_error
    assert "blocked" in _verdicts(mismatch, PATIENT_CONTEXT_CONSISTENCY_ID)
    assert not match.is_error
    assert match.content[0].text == "Patient/test-0001"


# ---------------------------------------------------------------------------
# redaction_placeholder_writeback (mandatory)
# ---------------------------------------------------------------------------


async def test_redaction_placeholder_writeback_blocks() -> None:
    server, _deid, recorded = _build_server()

    async with Client(server) as client:
        context = await _open(client)
        result = await client.call_tool(
            "write",
            {"note": "x [REDACTED:NAME]"},
            meta=_context_meta(context.contextId),
        )

    assert result.is_error
    assert "blocked" in _verdicts(result, REDACTION_PLACEHOLDER_WRITEBACK_ID)
    assert "write_calls" not in recorded


# ---------------------------------------------------------------------------
# Ordering: patient context is outermost, so guardrails see the context
# ---------------------------------------------------------------------------


async def test_guardrails_interceptor_sees_patient_context() -> None:
    server, llm, _recorded = _build_server()

    async with Client(server) as client:
        context = await _open(client)
        await client.call_tool("read", {}, meta=_context_meta(context.contextId))

    # The prompt evaluator runs inside the guardrails interceptor; if patient
    # context were not outermost, it would not see the bound context.
    assert llm.seen_context_ids
    assert llm.seen_context_ids[-1] == context.contextId


async def test_guardrails_report_is_added_outside_caller_extensions() -> None:
    observer = _ReportObserver()
    server, _deid, _recorded = _build_server(extensions=(observer,))

    async with Client(server) as client:
        context = await _open(client)
        result = await client.call_tool(
            "read", {}, meta=_context_meta(context.contextId)
        )

    # The caller extension is innermost: it must not yet see the guardrails report.
    assert observer.saw_report_inner is False
    # ...but the client-visible result does carry it.
    assert GUARDRAILS_EXTENSION in (result.meta or {})
    assert "pseudonymized" in _verdicts(result, PHI_DEIDENTIFICATION_ID)


# ---------------------------------------------------------------------------
# Population enforcement by behaviour
# ---------------------------------------------------------------------------


async def test_population_tool_allowed_without_context() -> None:
    policy = _RecordingPolicy(population_allowed=True)
    server, _deid, _recorded = _build_server(policy=policy, guardrails=False)

    async with Client(server) as client:
        result = await client.call_tool("population_tool", {})

    assert not result.is_error
    assert result.content[0].text == "pop"
    # The stub policy was consulted (population path, not patient enforcement).
    assert policy.population_calls == 1


async def test_population_tool_denied_is_31005() -> None:
    policy = _RecordingPolicy(population_allowed=False)
    server, _deid, _recorded = _build_server(policy=policy, guardrails=False)

    async with Client(server) as client:
        error = await _expect_error(client.call_tool("population_tool", {}))

    assert error.code == int(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)
    assert policy.population_calls == 1
