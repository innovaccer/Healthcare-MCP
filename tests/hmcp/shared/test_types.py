"""Tests for hmcp.shared.types."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from mcp.server.auth.provider import AccessToken
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from hmcp.shared.types import (
    ContextStatus,
    GuardrailReportEntry,
    GuardrailVerdict,
    PatientContext,
    PatientRef,
    Principal,
    bind_patient_context,
    current_patient_context,
    require_patient_context,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _patient_ref(reference: str = "Patient/test-0001") -> PatientRef:
    return PatientRef(reference=reference)


def _principal() -> Principal:
    return Principal(client_id="test-app")


def _context(
    *,
    status: ContextStatus = ContextStatus.ACTIVE,
    created_at: datetime = NOW,
    expires_at: datetime = NOW + timedelta(hours=1),
) -> PatientContext:
    return PatientContext(
        context_id="pc_synthetic",
        patient=_patient_ref(),
        purpose_of_use="TREAT",
        principal=_principal(),
        status=status,
        created_at=created_at,
        expires_at=expires_at,
    )


def _report(**overrides) -> GuardrailReportEntry:
    fields = dict(
        id="gr-test",
        version="1",
        digest="deadbeef",
        stage="input",
        verdict=GuardrailVerdict.PASS,
        evaluator="cascade(pattern,prompt)",
        findings=0,
        duration_ms=5,
    )
    fields.update(overrides)
    return GuardrailReportEntry(**fields)


# --- GuardrailReportEntry -------------------------------------------------


def test_guardrail_report_entry_requires_verdict():
    fields = dict(
        id="gr-test",
        version="1",
        digest="deadbeef",
        stage="input",
        evaluator="cascade(pattern,prompt)",
        findings=0,
        duration_ms=5,
    )
    with pytest.raises(ValidationError):
        GuardrailReportEntry(**fields)  # verdict omitted entirely
    with pytest.raises(ValidationError):
        _report(verdict=None)


def test_guardrail_report_entry_json_uses_camelcase_duration():
    dumped = _report(duration_ms=7).model_dump(by_alias=True, mode="json")
    assert dumped["durationMs"] == 7
    assert "duration_ms" not in dumped


def test_guardrail_report_entry_rejects_negative_findings():
    with pytest.raises(ValidationError):
        _report(findings=-1)


# --- PatientRef -----------------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    ["Patient/", "Observation/1", "Patient/" + "a" * 65],
)
def test_patient_ref_rejects_invalid(reference):
    with pytest.raises(ValidationError):
        PatientRef(reference=reference)


def test_patient_ref_accepts_valid():
    assert _patient_ref("Patient/abc-1.2").reference == "Patient/abc-1.2"


# --- PatientContext -------------------------------------------------------


def test_patient_context_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        _context(expires_at=datetime(2026, 1, 1, 1, 0, 0))


def test_is_usable_false_for_suspended():
    assert _context(status=ContextStatus.SUSPENDED).is_usable(now=NOW) is False


def test_is_usable_false_for_active_expired():
    ctx = _context(expires_at=NOW - timedelta(minutes=1))
    assert ctx.is_usable(now=NOW) is False


def test_is_usable_true_for_active_future():
    assert _context(expires_at=NOW + timedelta(hours=1)).is_usable(now=NOW) is True


def test_patient_context_is_frozen():
    ctx = _context()
    with pytest.raises(ValidationError):
        ctx.status = ContextStatus.CLOSED


def test_patient_context_forbids_extra_fields():
    fields = dict(
        context_id="pc_synthetic",
        patient=_patient_ref(),
        purpose_of_use="TREAT",
        principal=_principal(),
        status=ContextStatus.ACTIVE,
        created_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unexpected_field="synthetic",
    )
    with pytest.raises(ValidationError):
        PatientContext(**fields)


# --- Principal ------------------------------------------------------------


def test_principal_from_access_token():
    token = AccessToken(
        token="synthetic-token",
        client_id="test-app",
        scopes=["patient.read"],
        resource="https://res.example.com",
        subject="u1",
        claims={"iss": "https://auth.example.com"},
    )
    principal = Principal.from_access_token(token)
    assert principal.client_id == "test-app"
    assert principal.issuer == "https://auth.example.com"
    assert principal.subject == "u1"
    assert principal.resource == "https://res.example.com"


def test_principal_anonymous_client_id():
    assert Principal.anonymous().client_id == "anonymous"


# --- ContextVar accessors -------------------------------------------------


def test_current_patient_context_none_by_default():
    assert current_patient_context() is None


def test_bind_patient_context_sets_and_resets():
    ctx = _context()
    with bind_patient_context(ctx):
        assert current_patient_context() is ctx
    assert current_patient_context() is None


def test_bind_patient_context_resets_on_exception():
    ctx = _context()
    with pytest.raises(RuntimeError):
        with bind_patient_context(ctx):
            assert current_patient_context() is ctx
            raise RuntimeError("synthetic")
    assert current_patient_context() is None


def test_require_patient_context_outside_raises_mcp_error():
    with pytest.raises(MCPError) as excinfo:
        require_patient_context()
    assert excinfo.value.code == -31001


async def test_bind_patient_context_is_task_local():
    seen: dict[str, str] = {}

    async def worker(label: str) -> None:
        ctx = _context().model_copy(update={"context_id": label})
        with bind_patient_context(ctx):
            await asyncio.sleep(0)
            bound = current_patient_context()
            assert bound is not None
            seen[label] = bound.context_id

    await asyncio.gather(worker("pc_alpha"), worker("pc_beta"))

    assert seen == {"pc_alpha": "pc_alpha", "pc_beta": "pc_beta"}
    assert current_patient_context() is None
