"""Tests for builtin evaluators."""

import asyncio
from datetime import datetime, timezone

import pytest

from hmcp.shared.guardrails.definition import IdentifierType
from hmcp.shared.guardrails.evaluators.base import (
    EvaluationContext,
    EvaluationError,
    TextUnit,
)
from hmcp.shared.guardrails.evaluators.builtin import (
    PatientContextConsistencyEvaluator,
    RedactionPlaceholderEvaluator,
)
from hmcp.shared.types import (
    ContextStatus,
    Identifier,
    PatientContext,
    PatientRef,
    Principal,
)


def _ctx(patient_context=None, tool_name="get_patient"):
    return EvaluationContext(
        patient_context=patient_context, stage="input", tool_name=tool_name
    )


def _patient_context(reference="Patient/test-0001", identifiers=()):
    return PatientContext(
        context_id="ctx-test",
        patient=PatientRef(reference=reference, identifiers=tuple(identifiers)),
        purpose_of_use="TREAT",
        principal=Principal(client_id="client-test"),
        status=ContextStatus.ACTIVE,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        expires_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


def _mrn_type(is_access_key=True):
    return IdentifierType(
        name="mrn",
        display_name="MRN",
        category="MRN",
        text_patterns=(r"\bMRN[:#\s]*([A-Z0-9-]{4,20})\b",),
        is_access_key=is_access_key,
    )


def test_redaction_placeholder_finds_uppercase_only():
    evaluator = RedactionPlaceholderEvaluator()
    text = "a [REDACTED:NAME] b [redacted] c"
    result = asyncio.run(evaluator.find([TextUnit(text=text)], _ctx()))

    findings = result[0]
    assert len(findings) == 1
    assert findings[0].category == "REDACTION_PLACEHOLDER"
    assert findings[0].source == "builtin"
    assert (findings[0].start, findings[0].end) == (2, 17)


def test_consistency_requires_patient_context():
    evaluator = PatientContextConsistencyEvaluator(())
    with pytest.raises(EvaluationError):
        asyncio.run(evaluator.find([TextUnit(text="Patient/test-0002")], _ctx(None)))


def test_consistency_active_patient_has_no_finding():
    evaluator = PatientContextConsistencyEvaluator(())
    result = asyncio.run(
        evaluator.find([TextUnit(text="Patient/test-0001")], _ctx(_patient_context()))
    )
    assert result[0] == []


def test_consistency_other_patient_has_finding():
    evaluator = PatientContextConsistencyEvaluator(())
    result = asyncio.run(
        evaluator.find([TextUnit(text="Patient/test-0002")], _ctx(_patient_context()))
    )

    findings = result[0]
    assert len(findings) == 1
    assert findings[0].source == "builtin"
    assert findings[0].category == "OTHER_ID"
    assert findings[0].value == "test-0002"


def test_consistency_access_key_matches_context():
    pc = _patient_context(
        identifiers=(Identifier(system="urn:mrn", value="TEST-0001"),)
    )
    evaluator = PatientContextConsistencyEvaluator((_mrn_type(),))
    result = asyncio.run(evaluator.find([TextUnit(text="MRN: TEST-0001")], _ctx(pc)))
    assert result[0] == []


def test_consistency_access_key_mismatch():
    pc = _patient_context(
        identifiers=(Identifier(system="urn:mrn", value="TEST-0001"),)
    )
    evaluator = PatientContextConsistencyEvaluator((_mrn_type(),))
    result = asyncio.run(evaluator.find([TextUnit(text="MRN: TEST-9999")], _ctx(pc)))

    findings = result[0]
    assert len(findings) == 1
    assert findings[0].category == "MRN"
    assert findings[0].identifier_type == "mrn"
    assert findings[0].value == "TEST-9999"


def test_consistency_ignores_non_access_key_types():
    pc = _patient_context(
        identifiers=(Identifier(system="urn:mrn", value="TEST-0001"),)
    )
    evaluator = PatientContextConsistencyEvaluator((_mrn_type(is_access_key=False),))
    result = asyncio.run(evaluator.find([TextUnit(text="MRN: TEST-9999")], _ctx(pc)))
    assert result[0] == []


def test_consistency_skips_non_participating_optional_group():
    optional_type = IdentifierType(
        name="code",
        display_name="Code",
        category="OTHER_ID",
        text_patterns=(r"(MRN)?-code",),
        is_access_key=True,
    )
    pc = _patient_context()
    evaluator = PatientContextConsistencyEvaluator((optional_type,))
    result = asyncio.run(evaluator.find([TextUnit(text="hello -code")], _ctx(pc)))
    assert result[0] == []
