"""Tests for guardrail definitions."""

import pytest
from pydantic import ValidationError

from hmcp.shared.guardrails.definition import (
    ActionSpec,
    GuardrailDefinition,
    IdentifierType,
    PromptEvaluatorSpec,
)


def _identifier_type(**overrides):
    base = {
        "name": "mrn",
        "displayName": "MRN",
        "category": "MRN",
        "textPatterns": [r"\bMRN[:#\s]*([A-Z0-9-]{4,20})\b"],
        "jsonKeys": ["mrn"],
        "isAccessKey": True,
    }
    base.update(overrides)
    return base


def _definition(**overrides):
    base = {
        "id": "com.example/phi-guard",
        "version": "1.2.3",
        "title": "PHI guard",
        "description": "test definition",
        "stages": ["input", "output"],
        "evaluator": {
            "kind": "cascade",
            "steps": [
                {"kind": "pattern", "identifierTypes": [_identifier_type()]},
                {"kind": "deid", "categories": ["NAME"]},
            ],
        },
        "action": {"onFinding": "redact", "placeholder": "[REDACTED:{category}]"},
    }
    base.update(overrides)
    return GuardrailDefinition.model_validate(base)


def test_cascade_round_trips_with_camel_case_keys():
    definition = _definition()
    text = definition.to_json()

    assert '"failureMode"' in text
    assert '"appliesTo"' in text
    assert '"onFinding"' in text
    assert '"identifierTypes"' in text

    restored = GuardrailDefinition.from_json(text)
    assert restored == definition
    assert restored.digest == definition.digest


def test_digest_is_stable_and_changes_with_fields():
    definition = _definition()
    assert definition.digest.startswith("sha256:")
    assert (
        definition.digest == GuardrailDefinition.from_json(definition.to_json()).digest
    )

    assert _definition(version="1.2.4").digest != definition.digest

    changed_pattern = _definition(
        evaluator={
            "kind": "cascade",
            "steps": [
                {
                    "kind": "pattern",
                    "identifierTypes": [
                        _identifier_type(
                            textPatterns=[r"\bMRN[:#\s]*([A-Z0-9-]{5,20})\b"]
                        )
                    ],
                },
                {"kind": "deid", "categories": ["NAME"]},
            ],
        }
    )
    assert changed_pattern.digest != definition.digest


@pytest.mark.parametrize("bad_id", ["BadID", "com.example", "com.example/"])
def test_invalid_id_rejected(bad_id):
    with pytest.raises(ValidationError):
        _definition(id=bad_id)


@pytest.mark.parametrize("bad_version", ["1.2", "1.2.3.4", "v1.2.3"])
def test_invalid_version_rejected(bad_version):
    with pytest.raises(ValidationError):
        _definition(version=bad_version)


@pytest.mark.parametrize("pattern", [r"\d{3}-\d{2}-\d{4}", r"(\d)(\d)"])
def test_text_pattern_requires_exactly_one_group(pattern):
    with pytest.raises(ValidationError):
        IdentifierType(
            name="ssn", display_name="SSN", category="SSN", text_patterns=(pattern,)
        )


@pytest.mark.parametrize("pattern", [r"(", r"[a-"])
def test_text_pattern_syntactically_invalid_rejected(pattern):
    with pytest.raises(ValidationError):
        IdentifierType(
            name="ssn", display_name="SSN", category="SSN", text_patterns=(pattern,)
        )


def test_identifier_type_requires_pattern_or_key():
    with pytest.raises(ValidationError):
        IdentifierType(name="mrn", display_name="MRN", category="MRN")


@pytest.mark.parametrize(
    "placeholder",
    [
        "[REDACTED]",  # no {category} field
        "[REDACTED:{other}]",  # a field other than {category}
        "[REDACTED:{category}}]",  # an extra closing brace
        "[REDACTED:{{category}]",  # an extra opening brace
    ],
)
def test_placeholder_validator(placeholder):
    with pytest.raises(ValidationError):
        ActionSpec(on_finding="redact", placeholder=placeholder)


def test_prompt_spec_requires_placeholders():
    valid = PromptEvaluatorSpec(
        kind="prompt",
        system="sys {{nonce}}",
        user="nonce={{nonce}} body={{content}}",
        categories=("NAME",),
    )
    assert valid.system == "sys {{nonce}}"

    # system missing "{{nonce}}"
    with pytest.raises(ValidationError):
        PromptEvaluatorSpec(
            kind="prompt",
            system="no nonce",
            user="nonce={{nonce}} body={{content}}",
            categories=("NAME",),
        )
    # user missing "{{content}}"
    with pytest.raises(ValidationError):
        PromptEvaluatorSpec(
            kind="prompt",
            system="sys {{nonce}}",
            user="nonce={{nonce}} no content",
            categories=("NAME",),
        )
    # user missing "{{nonce}}"
    with pytest.raises(ValidationError):
        PromptEvaluatorSpec(
            kind="prompt",
            system="sys {{nonce}}",
            user="body={{content}} no nonce",
            categories=("NAME",),
        )


def test_builtin_requires_block_action():
    with pytest.raises(ValidationError):
        _definition(
            evaluator={"kind": "builtin", "name": "redaction_placeholder"},
            action={"onFinding": "redact"},
        )


def test_pseudonymize_requires_output_stage():
    with pytest.raises(ValidationError):
        _definition(stages=["input"], action={"onFinding": "pseudonymize"})

    allowed = _definition(stages=["output"], action={"onFinding": "pseudonymize"})
    assert allowed.action.on_finding == "pseudonymize"


def test_duplicate_stages_rejected():
    with pytest.raises(ValidationError):
        _definition(stages=["input", "input"])


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        _definition(unexpectedField="x")


def test_applies_filters_by_scope_and_tool():
    definition = _definition(
        appliesTo={"patientScopes": ["patient"], "tools": ["get_patient"]}
    )
    assert definition.applies("patient", "get_patient")
    assert not definition.applies("population", "get_patient")
    assert not definition.applies("patient", "search")

    default = _definition()
    assert default.applies("none", "anything")


def test_derived_members():
    definition = _definition()
    assert definition.major == 1
    assert definition.short_name == "phi-guard"
