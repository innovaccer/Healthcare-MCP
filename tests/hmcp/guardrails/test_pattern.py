"""Tests for the pattern evaluator."""

import asyncio

from hmcp.shared.guardrails.definition import IdentifierType, PatternEvaluatorSpec
from hmcp.shared.guardrails.evaluators.base import EvaluationContext, TextUnit
from hmcp.shared.guardrails.evaluators.pattern import PatternEvaluator


def _ctx():
    return EvaluationContext(patient_context=None, stage="input", tool_name="t")


def _spec(identifier_types):
    return PatternEvaluatorSpec(kind="pattern", identifier_types=identifier_types)


def test_ssn_pattern_exact_span():
    evaluator = PatternEvaluator(
        _spec(
            [
                IdentifierType(
                    name="ssn",
                    display_name="SSN",
                    category="SSN",
                    text_patterns=(r"\b(\d{3}-\d{2}-\d{4})\b",),
                )
            ]
        )
    )
    text = "ssn 900-12-3456 end"
    result = asyncio.run(evaluator.find([TextUnit(text=text)], _ctx()))

    assert len(result) == 1
    findings = result[0]
    assert len(findings) == 1
    assert findings[0].start == 4
    assert findings[0].end == 15
    assert findings[0].category == "SSN"
    assert findings[0].identifier_type == "ssn"
    assert findings[0].source == "pattern"
    assert findings[0].value == "900-12-3456"


def test_json_key_match_yields_whole_string_finding():
    evaluator = PatternEvaluator(
        _spec(
            [
                IdentifierType(
                    name="mrn",
                    display_name="MRN",
                    category="MRN",
                    json_keys=("mrn",),
                )
            ]
        )
    )
    result = asyncio.run(
        evaluator.find([TextUnit(text="TEST-0001", key="MRN")], _ctx())
    )

    findings = result[0]
    assert len(findings) == 1
    assert findings[0].start == 0
    assert findings[0].end == len("TEST-0001")
    assert findings[0].category == "MRN"
    assert findings[0].identifier_type == "mrn"


def test_one_list_per_unit_in_order():
    evaluator = PatternEvaluator(
        _spec(
            [
                IdentifierType(
                    name="ssn",
                    display_name="SSN",
                    category="SSN",
                    text_patterns=(r"\b(\d{3}-\d{2}-\d{4})\b",),
                )
            ]
        )
    )
    result = asyncio.run(
        evaluator.find(
            [TextUnit(text="900-12-3456"), TextUnit(text="nothing here")], _ctx()
        )
    )

    assert len(result) == 2
    assert len(result[0]) == 1
    assert result[1] == []


def test_pattern_is_case_insensitive():
    evaluator = PatternEvaluator(
        _spec(
            [
                IdentifierType(
                    name="name",
                    display_name="Name",
                    category="NAME",
                    text_patterns=(r"(testpatient)",),
                )
            ]
        )
    )
    result = asyncio.run(evaluator.find([TextUnit(text="TESTPATIENT")], _ctx()))

    assert len(result[0]) == 1
    assert result[0][0].start == 0
    assert result[0][0].end == 11


def test_optional_group_that_does_not_participate_is_skipped():
    evaluator = PatternEvaluator(
        _spec(
            [
                IdentifierType(
                    name="code",
                    display_name="Code",
                    category="OTHER_ID",
                    text_patterns=(r"(MRN)?-code",),
                )
            ]
        )
    )

    result = asyncio.run(evaluator.find([TextUnit(text="hello -code")], _ctx()))
    assert result[0] == []

    positive = asyncio.run(evaluator.find([TextUnit(text="MRN-code")], _ctx()))
    assert len(positive[0]) == 1
    assert positive[0][0].value == "MRN"
