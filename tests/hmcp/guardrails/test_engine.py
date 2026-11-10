"""Tests for the guardrail runtime engine. No network."""

import asyncio
import copy
import json
import logging
import types
from datetime import datetime, timezone

import pytest
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from hmcp.shared.guardrails.definition import (
    ActionSpec,
    BuiltinEvaluatorSpec,
    CascadeEvaluatorSpec,
    ClassifierEvaluatorSpec,
    DeidEvaluatorSpec,
    GuardrailDefinition,
    IdentifierType,
    PatternEvaluatorSpec,
    PhiCategory,
    PromptEvaluatorSpec,
)
from hmcp.shared.guardrails.engine import (
    CascadeEvaluator,
    GuardrailRuntime,
    build_evaluator,
)
from hmcp.shared.guardrails.evaluators.base import EvaluationContext, TextUnit
from hmcp.shared.guardrails.evaluators.classifier import ClassifierEvaluator
from hmcp.shared.guardrails.findings import Finding, locate_all, redact
from hmcp.shared.guardrails.reference import PHI_IDENTIFIER_TYPES, phi_deidentification
from hmcp.shared.guardrails.vault import InMemoryMarkerVault
from hmcp.shared.types import (
    ContextStatus,
    GuardrailVerdict,
    PatientContext,
    PatientRef,
    Principal,
)

_DEFINITION_ID = "com.example.test/pattern-check"


class SubstringDeid:
    """A fake de-identification model: reports the (value, label) pairs it is given."""

    def __init__(self, entities=()):
        self._entities = list(entities)

    async def process_text(self, request):
        results = []
        for text in request.text:
            entities = []
            for value, label in self._entities:
                for start, end in locate_all(text, value):
                    entities.append(
                        {
                            "text": value,
                            "best_label": label,
                            "location": {"stt_idx": start, "end_idx": end},
                        }
                    )
            results.append({"entities": entities})
        return results


class FakeLLM:
    """A fake prompt-evaluator LLM: reports the (value, category) pairs it is given.

    The prompt evaluator locates each reported text in the units itself, so the
    fake only echoes pairs whose value appears in the rendered user prompt.
    """

    def __init__(self, findings=()):
        self._findings = list(findings)

    async def complete(self, *, system, user, max_tokens):
        return json.dumps(
            {
                "findings": [
                    {"text": value, "category": category}
                    for value, category in self._findings
                    if value in user
                ]
            }
        )


class RaisingEvaluator:
    kind = "fake"

    def __init__(self, exc):
        self._exc = exc

    async def find(self, units, ctx):
        raise self._exc


class SlowEvaluator:
    kind = "fake"

    def __init__(self, delay):
        self._delay = delay

    async def find(self, units, ctx):
        await asyncio.sleep(self._delay)
        return [[] for _ in units]


class FixedEvaluator:
    """Returns the same findings for every unit (for cascade tests)."""

    kind = "fixed"

    def __init__(self, findings):
        self._findings = list(findings)

    async def find(self, units, ctx):
        return [list(self._findings) for _ in units]


class SucceedingVault:
    """A stub vault that would resolve/replace any marker, so it can never be
    the reason a guardrail reports ERROR."""

    async def marker_for(self, context_id, expires_at, category, value):
        return f"[{category}_1]"

    async def resolve(self, context_id, marker):
        return "Testpatient Alpha"


class SleepingVault:
    """A vault whose ``resolve`` outlives any short guardrail timeout."""

    async def marker_for(self, context_id, expires_at, category, value):
        return "[NAME_1]"

    async def resolve(self, context_id, marker):
        await asyncio.sleep(1.0)
        return "Testpatient Alpha"


class SleepingMarkerVault:
    """A vault whose ``marker_for`` outlives any short guardrail timeout."""

    async def marker_for(self, context_id, expires_at, category, value):
        await asyncio.sleep(1.0)
        return "[NAME_1]"

    async def resolve(self, context_id, marker):
        return "Testpatient Alpha"


def _ctx(patient_context=None, stage="input", tool_name="demo_tool"):
    return EvaluationContext(
        patient_context=patient_context, stage=stage, tool_name=tool_name
    )


def _patient_context(context_id="ctx-1", reference="Patient/test-0001"):
    return PatientContext(
        context_id=context_id,
        patient=PatientRef(reference=reference),
        purpose_of_use="TREAT",
        principal=Principal(client_id="client-test"),
        status=ContextStatus.ACTIVE,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )


def _pattern_definition(*, stages=("input",), action="redact", timeout_ms=5000):
    return GuardrailDefinition(
        id=_DEFINITION_ID,
        version="1.0.0",
        title="test pattern guardrail",
        stages=stages,
        evaluator=PatternEvaluatorSpec(
            kind="pattern", identifier_types=PHI_IDENTIFIER_TYPES
        ),
        action=ActionSpec(on_finding=action),
        timeout_ms=timeout_ms,
    )


def _digits_definition(*, stages=("output",), action="redact"):
    return GuardrailDefinition(
        id="com.example.test/digits-check",
        version="1.0.0",
        title="test digits guardrail",
        stages=stages,
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
        action=ActionSpec(on_finding=action),
    )


def _words_definition(*, stages=("input",), action="redact"):
    return GuardrailDefinition(
        id="com.example.test/words-check",
        version="1.0.0",
        title="test words guardrail",
        stages=stages,
        evaluator=PatternEvaluatorSpec(
            kind="pattern",
            identifier_types=(
                IdentifierType(
                    name="words",
                    display_name="Words",
                    category=PhiCategory.OTHER_ID,
                    text_patterns=(r"([A-Za-z]+)",),
                ),
            ),
        ),
        action=ActionSpec(on_finding=action),
    )


def test_build_evaluator_prompt_without_llm_raises():
    spec = PromptEvaluatorSpec(
        kind="prompt",
        system="find PHI between {{nonce}}",
        user="<content-{{nonce}}>{{content}}</content-{{nonce}}>",
        categories=(PhiCategory.NAME,),
    )
    with pytest.raises(ValueError):
        build_evaluator(spec, llm=None, deid=None)


def test_build_evaluator_deid_without_provider_raises():
    with pytest.raises(
        ValueError, match="deid evaluator requires a de-identification model"
    ):
        build_evaluator(DeidEvaluatorSpec(kind="deid"), llm=None, deid=None)


def test_build_evaluator_unknown_kind_raises():
    spec = types.SimpleNamespace(kind="nope")
    with pytest.raises(ValueError, match="unknown evaluator kind"):
        build_evaluator(spec, llm=None, deid=None)


def test_build_evaluator_unknown_builtin_name_raises():
    spec = BuiltinEvaluatorSpec.model_construct(kind="builtin", name="nope")
    with pytest.raises(ValueError, match="unknown builtin evaluator"):
        build_evaluator(spec, llm=None, deid=None)


def test_cascade_kind_string():
    spec = CascadeEvaluatorSpec(
        kind="cascade",
        steps=(
            PatternEvaluatorSpec(kind="pattern", identifier_types=PHI_IDENTIFIER_TYPES),
            DeidEvaluatorSpec(kind="deid"),
        ),
    )
    evaluator = build_evaluator(spec, llm=None, deid=SubstringDeid())
    assert evaluator.kind == "cascade(pattern,deid)"


def test_input_redact_mutates_in_place():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("input",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    nested = {"mrn": 123456, "notes": "SSN 900-12-3456"}
    arguments = {"patient": nested, "tags": ["a.b@example.com"]}

    entry = asyncio.run(runtime.run_input(arguments, _ctx()))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert entry.findings == 3
    # The very same dict objects were mutated in place.
    assert arguments["patient"] is nested
    assert arguments["patient"]["mrn"] == "[REDACTED:MRN]"
    assert arguments["patient"]["notes"] == "SSN [REDACTED:SSN]"
    assert arguments["tags"][0] == "[REDACTED:EMAIL]"


def test_output_redact_returns_new_result_and_leaves_original():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(
        content=[TextContent(type="text", text="SSN 900-12-3456")],
        structured_content={"mrn": 123456},
        is_error=False,
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result is not original
    assert new_result.content[0].text == "SSN [REDACTED:SSN]"
    assert new_result.structured_content["mrn"] == "[REDACTED:MRN]"
    # The original result is untouched.
    assert original.content[0].text == "SSN 900-12-3456"
    assert original.structured_content["mrn"] == 123456


def test_block_verdict_leaves_content():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("input",), action="block"),
        vault=InMemoryMarkerVault(),
    )
    arguments = {"ssn": "900-12-3456"}

    entry = asyncio.run(runtime.run_input(arguments, _ctx()))

    assert entry.verdict is GuardrailVerdict.BLOCKED
    assert arguments["ssn"] == "900-12-3456"


def test_annotate_verdict_leaves_content():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="annotate"),
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(
        content=[TextContent(type="text", text="SSN 900-12-3456")], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.ANNOTATED
    assert new_result.content[0].text == "SSN 900-12-3456"


def test_uninspectable_output_block_is_error():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(
        content=[ImageContent(type="image", data="aGk=", mime_type="image/png")],
        is_error=False,
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0
    assert new_result is original


def test_evaluator_exception_is_error_and_log_is_sanitised(caplog):
    definition = _pattern_definition(stages=("input",), action="redact")
    runtime = GuardrailRuntime(definition, vault=InMemoryMarkerVault())
    runtime._evaluator = RaisingEvaluator(RuntimeError("SENTINEL-PHI"))

    with caplog.at_level(logging.WARNING):
        entry = asyncio.run(runtime.run_input({"note": "secret-content"}, _ctx()))

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0
    assert "SENTINEL-PHI" not in caplog.text
    assert "secret-content" not in caplog.text
    assert definition.id in caplog.text
    assert "RuntimeError" in caplog.text


def test_evaluator_timeout_is_error(caplog):
    definition = _pattern_definition(stages=("input",), action="redact", timeout_ms=100)
    runtime = GuardrailRuntime(definition, vault=InMemoryMarkerVault())
    runtime._evaluator = SlowEvaluator(1.0)

    with caplog.at_level(logging.WARNING):
        entry = asyncio.run(runtime.run_input({"note": "anything"}, _ctx()))

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0


def _pseudonymizing_runtime():
    vault = InMemoryMarkerVault()
    llm = FakeLLM([("Testpatient Alpha", "NAME")])
    return GuardrailRuntime(phi_deidentification(), llm=llm, vault=vault)


def test_pseudonymize_round_trip():
    runtime = _pseudonymizing_runtime()
    patient_context = _patient_context()

    original = CallToolResult(
        content=[TextContent(type="text", text="Testpatient Alpha seen")],
        is_error=False,
    )
    new_result, out_entry = asyncio.run(
        runtime.run_output(original, _ctx(patient_context, stage="output"))
    )
    assert out_entry.verdict is GuardrailVerdict.PSEUDONYMIZED
    assert new_result.content[0].text == "[NAME_1] seen"

    arguments = {"note": "update [NAME_1]"}
    in_entry = asyncio.run(
        runtime.run_input(arguments, _ctx(patient_context, stage="input"))
    )
    assert in_entry.verdict is GuardrailVerdict.REIDENTIFIED
    assert arguments["note"] == "update Testpatient Alpha"


def test_unknown_marker_is_blocked_and_arg_unchanged():
    runtime = _pseudonymizing_runtime()
    patient_context = _patient_context()
    arguments = {"note": "update [NAME_9]"}

    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(patient_context, stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.BLOCKED
    assert arguments["note"] == "update [NAME_9]"


def test_marker_with_spaces_resolves():
    runtime = _pseudonymizing_runtime()
    patient_context = _patient_context()
    original = CallToolResult(
        content=[TextContent(type="text", text="Testpatient Alpha")], is_error=False
    )
    asyncio.run(runtime.run_output(original, _ctx(patient_context, stage="output")))

    arguments = {"note": "x [ NAME_1 ]"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(patient_context, stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.REIDENTIFIED
    assert arguments["note"] == "x Testpatient Alpha"


def test_marker_from_another_context_cannot_resolve():
    runtime = _pseudonymizing_runtime()
    original = CallToolResult(
        content=[TextContent(type="text", text="Testpatient Alpha")], is_error=False
    )
    asyncio.run(
        runtime.run_output(original, _ctx(_patient_context("ctx-1"), stage="output"))
    )

    arguments = {"note": "update [NAME_1]"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(_patient_context("ctx-2"), stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.BLOCKED
    assert arguments["note"] == "update [NAME_1]"


def test_reidentify_without_patient_context_is_error():
    # No markers: without the explicit `patient_context is None` guard this path
    # would return PASS, so removing the guard turns this test red.
    runtime = GuardrailRuntime(
        phi_deidentification(), llm=FakeLLM([]), vault=SucceedingVault()
    )

    entry = asyncio.run(
        runtime.run_input({"note": "plain text"}, _ctx(None, stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0


def test_reidentify_with_marker_and_no_patient_context_is_error():
    # The stub vault would resolve the marker, so only the None guard can fail.
    runtime = GuardrailRuntime(
        phi_deidentification(), llm=FakeLLM([]), vault=SucceedingVault()
    )

    entry = asyncio.run(
        runtime.run_input({"note": "update [NAME_1]"}, _ctx(None, stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0


def test_reidentify_timeout_is_error_and_args_unchanged(caplog):
    definition = GuardrailDefinition(
        id="com.example.test/slow-reidentify",
        version="1.0.0",
        title="slow re-identify guardrail",
        stages=("output", "input"),
        evaluator=PatternEvaluatorSpec(
            kind="pattern", identifier_types=PHI_IDENTIFIER_TYPES
        ),
        action=ActionSpec(on_finding="pseudonymize"),
        timeout_ms=100,
    )
    runtime = GuardrailRuntime(definition, vault=SleepingVault())
    arguments = {"note": "update [NAME_1]"}

    with caplog.at_level(logging.WARNING):
        entry = asyncio.run(
            runtime.run_input(arguments, _ctx(_patient_context(), stage="input"))
        )

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0
    assert arguments["note"] == "update [NAME_1]"


def test_pseudonymize_output_without_patient_context_is_error():
    runtime = GuardrailRuntime(
        phi_deidentification(), llm=FakeLLM([]), vault=SucceedingVault()
    )
    original = CallToolResult(
        content=[TextContent(type="text", text="Testpatient Alpha")], is_error=False
    )

    new_result, entry = asyncio.run(
        runtime.run_output(original, _ctx(None, stage="output"))
    )

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0
    assert new_result is original


# --- untested branches -----------------------------------------------------


def test_output_redact_embedded_text_resource():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(
        content=[
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="test://resource",
                    mime_type="text/plain",
                    text="SSN 900-12-3456",
                ),
            )
        ],
        is_error=False,
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result is not original
    assert new_result.content[0].resource.text == "SSN [REDACTED:SSN]"
    assert original.content[0].resource.text == "SSN 900-12-3456"


def test_reidentify_unit_without_marker_passes():
    runtime = _pseudonymizing_runtime()

    entry = asyncio.run(
        runtime.run_input(
            {"note": "plain text"}, _ctx(_patient_context(), stage="input")
        )
    )

    assert entry.verdict is GuardrailVerdict.PASS
    assert entry.findings == 0


def test_reidentify_only_the_marker_unit_changes():
    runtime = _pseudonymizing_runtime()
    patient_context = _patient_context()
    # Seed [NAME_1] in this context.
    original = CallToolResult(
        content=[TextContent(type="text", text="Testpatient Alpha")], is_error=False
    )
    asyncio.run(runtime.run_output(original, _ctx(patient_context, stage="output")))

    arguments = {"a": "plain text", "b": "update [NAME_1]", "c": "more plain"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(patient_context, stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.REIDENTIFIED
    assert entry.findings == 1
    assert arguments["a"] == "plain text"
    assert arguments["b"] == "update Testpatient Alpha"
    assert arguments["c"] == "more plain"


def test_input_bools_are_left_untouched():
    # Dict keys are inspected now, so a pattern matching letters would also
    # redact the keys (and collide); use a digits pattern so only the value is.
    runtime = GuardrailRuntime(
        _digits_definition(stages=("input",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    arguments = {"flag": True, "other": False, "name": "123456"}

    entry = asyncio.run(runtime.run_input(arguments, _ctx()))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert arguments["flag"] is True
    assert arguments["other"] is False
    assert arguments["name"] == "[REDACTED:OTHER_ID]"


# --- root-level scalar structured_content (BLOCKER) ------------------------


def test_output_redact_bare_string_structured_content():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(
        content=[], structured_content="900-12-3456", is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result.structured_content == "[REDACTED:SSN]"
    assert not set("900-12-3456") & set(new_result.structured_content)
    assert original.structured_content == "900-12-3456"


def test_output_redact_bare_int_structured_content():
    runtime = GuardrailRuntime(
        _digits_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(content=[], structured_content=123456, is_error=False)

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result.structured_content == "[REDACTED:OTHER_ID]"
    assert "123456" not in new_result.structured_content
    assert original.structured_content == 123456


def test_output_pseudonymize_bare_string_structured_content():
    runtime = _pseudonymizing_runtime()
    original = CallToolResult(
        content=[], structured_content="Testpatient Alpha", is_error=False
    )

    new_result, entry = asyncio.run(
        runtime.run_output(original, _ctx(_patient_context(), stage="output"))
    )

    assert entry.verdict is GuardrailVerdict.PSEUDONYMIZED
    assert new_result.structured_content == "[NAME_1]"
    assert "Testpatient Alpha" not in new_result.structured_content


def test_output_pseudonymize_bare_int_structured_content():
    vault = InMemoryMarkerVault()
    llm = FakeLLM([("123456", "NAME")])
    runtime = GuardrailRuntime(phi_deidentification(), llm=llm, vault=vault)
    original = CallToolResult(content=[], structured_content=123456, is_error=False)

    new_result, entry = asyncio.run(
        runtime.run_output(original, _ctx(_patient_context(), stage="output"))
    )

    assert entry.verdict is GuardrailVerdict.PSEUDONYMIZED
    assert new_result.structured_content == "[NAME_1]"
    assert "123456" not in new_result.structured_content


def test_unappliable_input_scalar_replacement_is_error(caplog):
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("input",), action="redact"),
        vault=InMemoryMarkerVault(),
    )

    with caplog.at_level(logging.WARNING):
        entry = asyncio.run(runtime.run_input("900-12-3456", _ctx()))

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0


# --- cascade union uses the unit text (MAJOR) ------------------------------


def test_cascade_union_uses_unit_text():
    text = "01234567890123456789"  # 20 characters
    step_a = FixedEvaluator(
        [Finding(category="NAME", start=0, end=5, source="prompt", value=text[0:5])]
    )
    step_b = FixedEvaluator(
        [Finding(category="DATE", start=3, end=20, source="pattern", value=text[3:20])]
    )
    cascade = CascadeEvaluator([step_a, step_b])

    merged = asyncio.run(cascade.find([TextUnit(text)], _ctx()))[0]

    assert len(merged) == 1
    finding = merged[0]
    assert (finding.start, finding.end) == (0, 20)
    # The union value comes from the unit text, not a partial constituent.
    assert finding.value == text
    redacted = redact(text, merged, "[REDACTED:{category}]")
    assert not any(character.isdigit() for character in redacted)


# --- JSON text blocks (architect's live-demo PHI leak) ---------------------


def test_output_redact_json_text_reproduction():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    source = '{"name": "Testpatient Alpha", "ssn": "900-12-3456"}'
    original = CallToolResult(
        content=[TextContent(type="text", text=source)],
        structured_content=None,
        is_error=False,
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result is not original
    payload = json.loads(new_result.content[0].text)
    assert payload == {"name": "[REDACTED:NAME]", "ssn": "[REDACTED:SSN]"}
    assert original.content[0].text == source


def test_output_redact_json_array_text():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    source = '[{"ssn": "900-12-3456"}, {"ssn": "900-12-9999"}]'
    original = CallToolResult(
        content=[TextContent(type="text", text=source)], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert json.loads(new_result.content[0].text) == [
        {"ssn": "[REDACTED:SSN]"},
        {"ssn": "[REDACTED:SSN]"},
    ]


def test_output_redact_json_nested_dict_text():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    source = '{"patient": {"name": "Testpatient Alpha", "ssn": "900-12-3456"}}'
    original = CallToolResult(
        content=[TextContent(type="text", text=source)], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert json.loads(new_result.content[0].text) == {
        "patient": {"name": "[REDACTED:NAME]", "ssn": "[REDACTED:SSN]"}
    }


def test_output_redact_non_json_braces_stays_plain_text():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    source = "note { ssn 900-12-3456 }"
    original = CallToolResult(
        content=[TextContent(type="text", text=source)], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result.content[0].text == "note { ssn [REDACTED:SSN] }"
    assert original.content[0].text == source


def test_pseudonymize_round_trip_through_json_text():
    vault = InMemoryMarkerVault()
    llm = FakeLLM([("Testpatient Alpha", "NAME")])
    runtime = GuardrailRuntime(phi_deidentification(), llm=llm, vault=vault)
    patient_context = _patient_context()
    source = '{"name": "Testpatient Alpha", "ssn": "900-12-3456"}'
    original = CallToolResult(
        content=[TextContent(type="text", text=source)], is_error=False
    )

    new_result, out_entry = asyncio.run(
        runtime.run_output(original, _ctx(patient_context, stage="output"))
    )
    assert out_entry.verdict is GuardrailVerdict.PSEUDONYMIZED
    masked = json.loads(new_result.content[0].text)
    assert masked == {"name": "[NAME_1]", "ssn": "[SSN_1]"}

    arguments = {"payload": new_result.content[0].text}
    in_entry = asyncio.run(
        runtime.run_input(arguments, _ctx(patient_context, stage="input"))
    )
    assert in_entry.verdict is GuardrailVerdict.REIDENTIFIED
    assert json.loads(arguments["payload"]) == {
        "name": "Testpatient Alpha",
        "ssn": "900-12-3456",
    }


# --- bare markers (models often drop the brackets) -------------------------


def _seeded_runtime():
    """A phi_deidentification runtime with [NAME_1] already in its vault."""
    runtime = _pseudonymizing_runtime()
    original = CallToolResult(
        content=[TextContent(type="text", text="Testpatient Alpha")], is_error=False
    )
    asyncio.run(runtime.run_output(original, _ctx(_patient_context(), stage="output")))
    return runtime


def test_reidentify_bare_marker_restores_value():
    runtime = _seeded_runtime()

    arguments = {"note": "note NAME_1"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(_patient_context(), stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.REIDENTIFIED
    assert entry.findings == 1
    assert arguments["note"] == "note Testpatient Alpha"


def test_reidentify_unknown_bare_token_passes_through():
    runtime = _seeded_runtime()

    arguments = {"note": "code ICD_10 line LINE_1"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(_patient_context(), stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.PASS
    assert entry.findings == 0
    assert arguments["note"] == "code ICD_10 line LINE_1"


def test_reidentify_unknown_bracketed_marker_blocks_and_leaves_args():
    runtime = _seeded_runtime()

    arguments = {"note": "[NAME_9] and NAME_1"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(_patient_context(), stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.BLOCKED
    assert arguments["note"] == "[NAME_9] and NAME_1"


def test_reidentify_bare_token_needs_a_word_boundary():
    runtime = _seeded_runtime()

    arguments = {"note": "XNAME_1 NAME_12"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(_patient_context(), stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.PASS
    assert entry.findings == 0
    assert arguments["note"] == "XNAME_1 NAME_12"


def test_reidentify_bare_token_needs_a_letter_boundary():
    runtime = _seeded_runtime()

    arguments = {"note": "fooNAME_1 NAME_1x bar"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(_patient_context(), stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.PASS
    assert entry.findings == 0
    assert arguments["note"] == "fooNAME_1 NAME_1x bar"


def test_reidentify_bracketed_and_bare_marker_together():
    runtime = _seeded_runtime()

    arguments = {"note": "update [NAME_1] and NAME_1"}
    entry = asyncio.run(
        runtime.run_input(arguments, _ctx(_patient_context(), stage="input"))
    )

    assert entry.verdict is GuardrailVerdict.REIDENTIFIED
    assert entry.findings == 2
    assert arguments["note"] == "update Testpatient Alpha and Testpatient Alpha"


# --- dict keys are inspected too (D-K1) ------------------------------------


def _redacting_runtime():
    return GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )


def test_output_redact_json_text_key_is_inspected():
    runtime = _redacting_runtime()
    source = '{"900-12-3456": "on file"}'
    original = CallToolResult(
        content=[TextContent(type="text", text=source)], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert json.loads(new_result.content[0].text) == {"[REDACTED:SSN]": "on file"}
    assert "900-12-3456" not in new_result.content[0].text
    assert original.content[0].text == source


def test_output_redact_structured_content_key_is_inspected():
    runtime = _redacting_runtime()
    original = CallToolResult(
        content=[], structured_content={"900-12-3456": "on file"}, is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result.structured_content == {"[REDACTED:SSN]": "on file"}
    assert original.structured_content == {"900-12-3456": "on file"}


def test_output_redact_nested_json_key_is_inspected():
    runtime = _redacting_runtime()
    source = '{"outer": {"900-12-3456": "x"}}'
    original = CallToolResult(
        content=[TextContent(type="text", text=source)], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert json.loads(new_result.content[0].text) == {"outer": {"[REDACTED:SSN]": "x"}}


def test_output_redact_preserves_key_order_after_rename():
    runtime = _redacting_runtime()
    original = CallToolResult(
        content=[],
        structured_content={"a": "x", "900-12-3456": "y", "b": "z"},
        is_error=False,
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert list(new_result.structured_content) == ["a", "[REDACTED:SSN]", "b"]


def test_output_redact_key_collision_is_error():
    runtime = _redacting_runtime()
    original = CallToolResult(
        content=[],
        structured_content={"900-12-3456": 1, "900-12-3457": 2},
        is_error=False,
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.ERROR
    assert new_result is original


def test_output_redact_key_and_value_both_phi_post_order():
    runtime = _redacting_runtime()
    original = CallToolResult(
        content=[],
        structured_content={"900-12-3456": "900-12-3456"},
        is_error=False,
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result.structured_content == {"[REDACTED:SSN]": "[REDACTED:SSN]"}
    assert "900-12-3456" not in new_result.structured_content


def test_output_redact_json_key_does_not_use_json_keys_rules():
    runtime = _redacting_runtime()
    source = '{"name": {"first": "x"}}'
    original = CallToolResult(
        content=[TextContent(type="text", text=source)], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.PASS
    assert json.loads(new_result.content[0].text) == {"name": {"first": "x"}}


def test_pseudonymize_json_key_round_trip():
    runtime = _pseudonymizing_runtime()
    patient_context = _patient_context()
    original = CallToolResult(
        content=[], structured_content={"Testpatient Alpha": "x"}, is_error=False
    )

    new_result, out_entry = asyncio.run(
        runtime.run_output(original, _ctx(patient_context, stage="output"))
    )
    assert out_entry.verdict is GuardrailVerdict.PSEUDONYMIZED
    assert new_result.structured_content == {"[NAME_1]": "x"}

    for arguments in ({"[NAME_1]": "y"}, {"NAME_1": "y"}):
        in_entry = asyncio.run(
            runtime.run_input(arguments, _ctx(patient_context, stage="input"))
        )
        assert in_entry.verdict is GuardrailVerdict.REIDENTIFIED
        assert arguments == {"Testpatient Alpha": "y"}


# --- non-str dict keys (D-K2) ----------------------------------------------


def test_output_redact_int_key_is_inspected():
    runtime = GuardrailRuntime(_digits_definition(), vault=InMemoryMarkerVault())
    original = CallToolResult(
        content=[], structured_content={900123456: "v"}, is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result.structured_content == {"[REDACTED:OTHER_ID]": "v"}
    assert original.structured_content == {900123456: "v"}


def test_output_int_key_does_not_break_json_keys_lookup():
    runtime = GuardrailRuntime(
        _pattern_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(
        content=[], structured_content={1: "Testpatient Alpha"}, is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.PASS
    assert new_result.structured_content == {1: "Testpatient Alpha"}


def test_output_non_str_keys_are_inspected_in_wire_form():
    # pydantic serialises non-str keys as strings on the wire ("true", "2.5"),
    # so the guardrail normalises to that form and inspects the key text rather
    # than ignoring it.
    runtime = GuardrailRuntime(
        _words_definition(stages=("output",), action="redact"),
        vault=InMemoryMarkerVault(),
    )
    content = {True: "1", 2.5: "2"}
    original = CallToolResult(content=[], structured_content=content, is_error=False)

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.REDACTED
    assert new_result.structured_content == {"[REDACTED:OTHER_ID]": "1", "2.5": "2"}


def test_pseudonymize_output_timeout_is_error(caplog):
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
    runtime = GuardrailRuntime(definition, vault=SleepingMarkerVault())
    original = CallToolResult(
        content=[TextContent(type="text", text="123456")], is_error=False
    )

    with caplog.at_level(logging.WARNING):
        new_result, entry = asyncio.run(
            runtime.run_output(original, _ctx(_patient_context(), stage="output"))
        )

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0
    assert new_result is original


# --- classifier evaluator --------------------------------------------------


class RecordingClassifier:
    """A fake classifier provider that records the requests it is given."""

    def __init__(self, *, flagged=False, raises=False):
        self._flagged = flagged
        self._raises = raises
        self.requests = []

    async def classify(self, request):
        self.requests.append(request)
        if self._raises:
            raise RuntimeError("classifier down")
        return {"flagged": self._flagged}


def _classifier_definition(*, stages=("input",), action="block"):
    return GuardrailDefinition(
        id="com.example.test/classifier-check",
        version="1.0.0",
        title="test classifier guardrail",
        stages=stages,
        evaluator=ClassifierEvaluatorSpec(kind="classifier", classifier="toxicity"),
        action=ActionSpec(on_finding=action),
    )


def test_build_evaluator_classifier_without_provider_raises():
    spec = ClassifierEvaluatorSpec(kind="classifier", classifier="toxicity")
    with pytest.raises(
        ValueError, match="classifier evaluator requires a classifier provider"
    ):
        build_evaluator(spec, llm=None, deid=None)


def test_build_evaluator_classifier_with_provider_returns_evaluator():
    spec = ClassifierEvaluatorSpec(kind="classifier", classifier="toxicity")
    evaluator = build_evaluator(
        spec, llm=None, deid=None, classifier=RecordingClassifier()
    )
    assert isinstance(evaluator, ClassifierEvaluator)
    assert evaluator.kind == "classifier"


def test_classifier_input_flagged_blocks():
    provider = RecordingClassifier(flagged=True)
    runtime = GuardrailRuntime(
        _classifier_definition(stages=("input",), action="block"),
        classifier=provider,
        vault=InMemoryMarkerVault(),
    )

    entry = asyncio.run(runtime.run_input({"note": "hello"}, _ctx()))

    assert entry.verdict is GuardrailVerdict.BLOCKED
    assert entry.findings == 1
    assert entry.evaluator == "classifier"
    assert provider.requests
    assert all(request.stage == "input" for request in provider.requests)


def test_classifier_input_annotate_leaves_arguments_unchanged():
    provider = RecordingClassifier(flagged=True)
    runtime = GuardrailRuntime(
        _classifier_definition(stages=("input",), action="annotate"),
        classifier=provider,
        vault=InMemoryMarkerVault(),
    )
    arguments = {"note": "hello", "nested": {"mrn": 123456}}
    before = copy.deepcopy(arguments)

    entry = asyncio.run(runtime.run_input(arguments, _ctx()))

    assert entry.verdict is GuardrailVerdict.ANNOTATED
    assert arguments == before


def test_classifier_not_flagged_passes():
    provider = RecordingClassifier(flagged=False)
    runtime = GuardrailRuntime(
        _classifier_definition(stages=("input",), action="block"),
        classifier=provider,
        vault=InMemoryMarkerVault(),
    )

    entry = asyncio.run(runtime.run_input({"note": "hello"}, _ctx()))

    assert entry.verdict is GuardrailVerdict.PASS
    assert entry.findings == 0


def test_classifier_provider_error_is_error():
    provider = RecordingClassifier(raises=True)
    runtime = GuardrailRuntime(
        _classifier_definition(stages=("input",), action="block"),
        classifier=provider,
        vault=InMemoryMarkerVault(),
    )

    entry = asyncio.run(runtime.run_input({"note": "hello"}, _ctx()))

    assert entry.verdict is GuardrailVerdict.ERROR
    assert entry.findings == 0


def test_classifier_output_flagged_blocks():
    provider = RecordingClassifier(flagged=True)
    runtime = GuardrailRuntime(
        _classifier_definition(stages=("output",), action="block"),
        classifier=provider,
        vault=InMemoryMarkerVault(),
    )
    original = CallToolResult(
        content=[TextContent(type="text", text="flagged text")], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, _ctx(stage="output")))

    assert entry.verdict is GuardrailVerdict.BLOCKED
    assert provider.requests
    assert all(request.stage == "output" for request in provider.requests)
