"""Tests for the deid evaluator over the abstract de-identification model.

No network: every model is a fake returning a de-identification service's
``/process/text`` response shape (plain dicts).
"""

import asyncio
import traceback

import pytest
from mcp.types import CallToolResult, TextContent

from hmcp.shared.guardrails.definition import (
    ActionSpec,
    CascadeEvaluatorSpec,
    DeidEvaluatorSpec,
    GuardrailDefinition,
    PatternEvaluatorSpec,
    PhiCategory,
)
from hmcp.shared.guardrails.engine import GuardrailRuntime, build_evaluator
from hmcp.shared.guardrails.evaluators.base import (
    EvaluationContext,
    EvaluationError,
    TextUnit,
)
from hmcp.shared.guardrails.evaluators.deid import (
    DeidEntity,
    DeidEntityLocation,
    DeidEvaluator,
    DeidRequest,
    label_to_category,
)
from hmcp.shared.guardrails.findings import locate_all
from hmcp.shared.guardrails.reference import PHI_IDENTIFIER_TYPES
from hmcp.shared.guardrails.vault import InMemoryMarkerVault
from hmcp.shared.types import GuardrailVerdict

_SYNTH = "Zqxpatient Beta"


class FakeModel:
    def __init__(self, result):
        self._result = result
        self.requests: list[DeidRequest] = []

    async def process_text(self, request):
        self.requests.append(request)
        if callable(self._result):
            return self._result(request)
        return self._result


def make_ctx() -> EvaluationContext:
    return EvaluationContext(
        patient_context=None, stage="output", tool_name="demo_tool"
    )


def _entity_response(text, pairs):
    """Build a service-shaped result reporting every (value, label) occurrence."""
    entities = []
    for value, label in pairs:
        for start, end in locate_all(text, value):
            entities.append(
                {
                    "text": value,
                    "best_label": label,
                    "location": {"stt_idx": start, "end_idx": end},
                }
            )
    return {"entities": entities}


# --- request shape ---------------------------------------------------------


def test_request_shape_is_exact():
    model = FakeModel(lambda request: [{"entities": []} for _ in request.text])
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    asyncio.run(evaluator.find([TextUnit("Testpatient Alpha")], make_ctx()))

    assert model.requests[-1].model_dump(mode="json") == {
        "text": ["Testpatient Alpha"],
        "entity_detection": {"return_entity": True},
    }


def test_request_shape_with_categories_forwards_no_selector():
    model = FakeModel(lambda request: [{"entities": []} for _ in request.text])
    spec = DeidEvaluatorSpec(kind="deid", categories=(PhiCategory.NAME,))
    evaluator = DeidEvaluator(spec, model)

    asyncio.run(evaluator.find([TextUnit("Testpatient Alpha")], make_ctx()))

    dumped = model.requests[-1].model_dump(mode="json")
    assert dumped == {
        "text": ["Testpatient Alpha"],
        "entity_detection": {"return_entity": True},
    }
    assert "entity_types" not in repr(dumped)


# --- spans and label mapping -----------------------------------------------


def test_reported_span_and_label_mapping():
    text = "Testpatient Alpha born 900-12-3456 in Testville id HEALTH123 code ZZZ"
    pairs = [
        ("Testpatient Alpha", "NAME_GIVEN"),
        ("900-12-3456", "DOB"),
        ("Testville", "LOCATION_ADDRESS_STREET"),
        ("HEALTH123", "HEALTHCARE_NUMBER"),
        ("ZZZ", "ZZZ_NEW"),
    ]
    model = FakeModel(
        lambda request: [_entity_response(t, pairs) for t in request.text]
    )
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    findings = asyncio.run(evaluator.find([TextUnit(text)], make_ctx()))[0]

    by_value: dict[str, set[str]] = {}
    for finding in findings:
        by_value.setdefault(text[finding.start : finding.end], set()).add(
            finding.category
        )
    assert by_value["Testpatient Alpha"] == {"NAME"}
    assert by_value["900-12-3456"] == {"DATE"}
    assert by_value["Testville"] == {"LOCATION"}
    assert by_value["HEALTH123"] == {"HEALTH_PLAN_ID"}
    assert by_value["ZZZ"] == {"OTHER_ID"}


def test_trims_edge_punctuation_and_whitespace():
    text = "lives in Testville MA 00000. age  39. gap .. end"
    entities = [
        {
            "text": "Testville MA 00000.",
            "best_label": "LOCATION_ADDRESS_STREET",
            "location": {"stt_idx": 9, "end_idx": 28},
        },
        {
            "text": " 39.",
            "best_label": "DATE",
            "location": {"stt_idx": 33, "end_idx": 37},
        },
        {
            "text": " .. ",
            "best_label": "OTHER",
            "location": {"stt_idx": 41, "end_idx": 45},
        },
    ]
    model = FakeModel([{"entities": entities}])
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    findings = asyncio.run(evaluator.find([TextUnit(text)], make_ctx()))[0]

    assert {text[f.start : f.end] for f in findings} == {
        "Testville MA 00000",
        "39",
    }


def test_second_occurrence_is_also_a_finding():
    text = "Testpatient Alpha met Testpatient Alpha"
    # Report ONLY the first occurrence: the second (22, 39) can only be found by
    # the evaluator's own locate_all re-occurrence search.
    model = FakeModel(
        [
            {
                "entities": [
                    {
                        "text": "Testpatient Alpha",
                        "best_label": "NAME_GIVEN",
                        "location": {"stt_idx": 0, "end_idx": 17},
                    }
                ]
            }
        ]
    )
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    findings = asyncio.run(evaluator.find([TextUnit(text)], make_ctx()))[0]

    spans = {(f.start, f.end) for f in findings}
    assert (0, 17) in spans
    assert (22, 39) in spans


def test_full_service_shape_is_accepted():
    text = "Patient Testpatient Alpha was seen"
    raw = [
        {
            "entities": [
                {
                    "text": "Testpatient",
                    "best_label": "NAME_GIVEN",
                    "labels": {"NAME_GIVEN": 0.99},
                    "processed_text": None,
                    "location": {
                        "stt_idx": 8,
                        "end_idx": 19,
                        "stt_idx_processed": None,
                        "end_idx_processed": None,
                    },
                }
            ],
            "entities_present": True,
            "characters_processed": 40,
            "languages_detected": {"en": 1.0},
            "processed_text": "Patient [NAME_GIVEN_1] ...",
        }
    ]
    model = FakeModel(raw)
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    findings = asyncio.run(evaluator.find([TextUnit(text)], make_ctx()))[0]

    assert findings
    assert all(f.category == "NAME" for f in findings)
    assert all(text[f.start : f.end] == "Testpatient" for f in findings)


# --- fail closed -----------------------------------------------------------


def _ok_entity(label="NAME_GIVEN"):
    return {
        "text": _SYNTH,
        "best_label": label,
        "location": {"stt_idx": 0, "end_idx": len(_SYNTH)},
    }


def _response_entity_texts(response):
    """Every entity "text" string present anywhere in a raw response value."""
    texts: list[str] = []
    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and isinstance(item.get("entities"), list):
                for entity in item["entities"]:
                    if isinstance(entity, dict) and isinstance(entity.get("text"), str):
                        texts.append(entity["text"])
    return texts


@pytest.mark.parametrize(
    "response",
    [
        [],  # outer length wrong
        {"entities": []},  # response is a dict
        "nope",  # response is a str
        [{}],  # item without "entities"
        [{"entities": "x"}],  # entities not a list
        [{"entities": [{"text": _SYNTH, "best_label": "NAME_GIVEN"}]}],  # no location
        [
            {
                "entities": [
                    _ok_entity()
                    | {"location": {"stt_idx": "0", "end_idx": len(_SYNTH)}}
                ]
            }
        ],
        [
            {
                "entities": [
                    _ok_entity()
                    | {"location": {"stt_idx": 0.0, "end_idx": len(_SYNTH)}}
                ]
            }
        ],
        [
            {
                "entities": [
                    _ok_entity()
                    | {"location": {"stt_idx": False, "end_idx": len(_SYNTH)}}
                ]
            }
        ],
        [
            {
                "entities": [
                    _ok_entity()
                    | {"location": {"stt_idx": 0, "end_idx": float(len(_SYNTH))}}
                ]
            }
        ],
        [{"entities": [_ok_entity() | {"location": {"stt_idx": 0, "end_idx": 999}}]}],
        [{"entities": [_ok_entity() | {"location": {"stt_idx": 5, "end_idx": 5}}]}],
        [{"entities": [_ok_entity() | {"text": "Zqxpatient Beth"}]}],
        [{"entities": [_ok_entity("name")]}],
        [{"entities": [_ok_entity("NAME-X")]}],
        [{"entities": [_ok_entity("")]}],
        [{"entities": [_ok_entity("1ABC")]}],
    ],
)
def test_fails_closed_without_leaking(response):
    model = FakeModel(response)
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    with pytest.raises(EvaluationError) as excinfo:
        asyncio.run(evaluator.find([TextUnit(_SYNTH)], make_ctx()))

    message = str(excinfo.value)
    # Neither the unit text nor any entity "text" echoed by the response may leak.
    for value in (*_response_entity_texts(response), _SYNTH):
        assert value not in message


@pytest.mark.parametrize(
    "location, entity_text",
    [
        ({"stt_idx": 0, "end_idx": 999}, _SYNTH),  # span out of bounds
        ({"stt_idx": 0, "end_idx": len(_SYNTH)}, "Zqxpatient Beth"),  # mismatch
    ],
)
def test_span_errors_never_echo_label_or_text(location, entity_text):
    label = "TESTPATIENT_ALPHA"  # a model may upper-case a name into a label
    model = FakeModel(
        [
            {
                "entities": [
                    {"text": entity_text, "best_label": label, "location": location}
                ]
            }
        ]
    )
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    with pytest.raises(EvaluationError) as excinfo:
        asyncio.run(evaluator.find([TextUnit(_SYNTH)], make_ctx()))

    message = str(excinfo.value)
    assert label not in message
    assert entity_text not in message
    assert _SYNTH not in message


def test_pydantic_failure_is_suppressed():
    model = FakeModel([{"entities": "Zqxpatient Beta"}])
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    with pytest.raises(EvaluationError) as excinfo:
        asyncio.run(evaluator.find([TextUnit("anything")], make_ctx()))

    exc = excinfo.value
    assert str(exc) == "invalid deid response"
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert exc.__suppress_context__ is True


class RaisingModel:
    """A de-identification model whose provider raises with PHI in the message."""

    def __init__(self, message):
        self._message = message

    async def process_text(self, request):
        raise RuntimeError(self._message)


def test_model_exception_is_fixed_message_with_no_cause():
    model = RaisingModel("deid down with ssn 123-45-6789")
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)

    with pytest.raises(EvaluationError) as excinfo:
        asyncio.run(evaluator.find([TextUnit("anything")], make_ctx()))

    exc = excinfo.value
    assert str(exc) == "deid model failed"
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert exc.__suppress_context__ is True
    assert "123-45-6789" not in "".join(traceback.format_exception(exc))


# --- filtering, batching, repr ---------------------------------------------


def test_categories_filter_is_applied_locally():
    text = "Testpatient Alpha 555-0100"
    pairs = [("Testpatient Alpha", "NAME_GIVEN"), ("555-0100", "PHONE_NUMBER")]
    model = FakeModel(
        lambda request: [_entity_response(t, pairs) for t in request.text]
    )
    spec = DeidEvaluatorSpec(kind="deid", categories=(PhiCategory.NAME,))
    evaluator = DeidEvaluator(spec, model)

    findings = asyncio.run(evaluator.find([TextUnit(text)], make_ctx()))[0]

    assert {f.category for f in findings} == {"NAME"}


def test_batches_of_at_most_50_align_to_units():
    calls: list[int] = []

    def respond(request):
        calls.append(len(request.text))
        results = []
        for text in request.text:
            start = text.find("Testpatient Alpha")
            entities = []
            if start >= 0:
                entities.append(
                    {
                        "text": "Testpatient Alpha",
                        "best_label": "NAME_GIVEN",
                        "location": {
                            "stt_idx": start,
                            "end_idx": start + len("Testpatient Alpha"),
                        },
                    }
                )
            results.append({"entities": entities})
        return results

    model = FakeModel(respond)
    evaluator = DeidEvaluator(DeidEvaluatorSpec(kind="deid"), model)
    units = [
        TextUnit(f"unit {i} Testpatient Alpha" if i % 2 == 0 else f"unit {i}")
        for i in range(120)
    ]

    results = asyncio.run(evaluator.find(units, make_ctx()))

    assert calls == [50, 50, 20]
    assert len(results) == 120
    for index, findings in enumerate(results):
        if index % 2 == 0:
            assert findings
            assert all(
                units[index].text[f.start : f.end] == "Testpatient Alpha"
                for f in findings
            )
        else:
            assert findings == []


def test_entity_repr_hides_text():
    entity = DeidEntity(
        text="Testpatient Alpha",
        best_label="NAME_GIVEN",
        location=DeidEntityLocation(stt_idx=0, end_idx=17),
    )
    assert "Testpatient Alpha" not in repr(entity)


def test_request_repr_hides_text():
    request = DeidRequest(text=["Zqxpatient Beta"])
    assert "Zqxpatient Beta" not in repr(request)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("NAME_GIVEN", PhiCategory.NAME),
        ("NAME_FAMILY", PhiCategory.NAME),
        ("DATE", PhiCategory.DATE),
        ("DATE_INTERVAL", PhiCategory.DATE),
        ("DOB", PhiCategory.DATE),
        ("AGE", PhiCategory.DATE),
        ("PHONE_NUMBER", PhiCategory.PHONE),
        ("FAX_NUMBER", PhiCategory.FAX),
        ("EMAIL_ADDRESS", PhiCategory.EMAIL),
        ("SSN", PhiCategory.SSN),
        ("MEDICAL_RECORD_NUMBER", PhiCategory.MRN),
        ("HEALTHCARE_NUMBER", PhiCategory.HEALTH_PLAN_ID),
        ("BANK_ACCOUNT", PhiCategory.ACCOUNT_NUMBER),
        ("CREDIT_CARD", PhiCategory.ACCOUNT_NUMBER),
        ("ROUTING_NUMBER", PhiCategory.ACCOUNT_NUMBER),
        ("ACCOUNT_NUMBER", PhiCategory.ACCOUNT_NUMBER),
        ("DRIVER_LICENSE", PhiCategory.LICENSE_NUMBER),
        ("PASSPORT_NUMBER", PhiCategory.LICENSE_NUMBER),
        ("VEHICLE_ID", PhiCategory.VEHICLE_ID),
        ("DEVICE_ID", PhiCategory.DEVICE_ID),
        ("URL", PhiCategory.URL),
        ("IP_ADDRESS", PhiCategory.IP_ADDRESS),
        ("LOCATION_CITY", PhiCategory.LOCATION),
        ("LOCATION_ADDRESS_STREET", PhiCategory.LOCATION),
        ("TIME", PhiCategory.OTHER_ID),
        ("ZZZ_NEW", PhiCategory.OTHER_ID),
    ],
)
def test_label_to_category_table(label, expected):
    assert label_to_category(label) is expected


# --- end to end ------------------------------------------------------------


def test_runtime_redacts_through_cascade():
    text = "Testpatient Alpha SSN 900-12-3456"
    pairs = [("Testpatient Alpha", "NAME_GIVEN"), ("900-12-3456", "SSN")]
    model = FakeModel(
        lambda request: [_entity_response(t, pairs) for t in request.text]
    )
    definition = GuardrailDefinition(
        id="com.example.test/cascade-deid-redact",
        version="1.0.0",
        title="test cascade deid redact",
        stages=("output",),
        evaluator=CascadeEvaluatorSpec(
            kind="cascade",
            steps=(
                PatternEvaluatorSpec(
                    kind="pattern", identifier_types=PHI_IDENTIFIER_TYPES
                ),
                DeidEvaluatorSpec(kind="deid"),
            ),
        ),
        action=ActionSpec(on_finding="redact"),
    )
    runtime = GuardrailRuntime(definition, deid=model, vault=InMemoryMarkerVault())
    original = CallToolResult(
        content=[TextContent(type="text", text=text)], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, make_ctx()))

    assert entry.verdict is GuardrailVerdict.REDACTED
    output = new_result.content[0].text
    assert "Testpatient Alpha" not in output
    assert "900-12-3456" not in output


def test_overlapping_occurrences_are_redacted():
    def respond(request):
        results = []
        for text in request.text:
            entities = []
            if text == "AAA":
                entities.append(
                    {
                        "text": "AA",
                        "best_label": "NAME_GIVEN",
                        "location": {"stt_idx": 0, "end_idx": 2},
                    }
                )
            elif text == "banana":
                entities.append(
                    {
                        "text": "ana",
                        "best_label": "NAME_GIVEN",
                        "location": {"stt_idx": 1, "end_idx": 4},
                    }
                )
            results.append({"entities": entities})
        return results

    model = FakeModel(respond)
    definition = GuardrailDefinition(
        id="com.example.test/deid-redact",
        version="1.0.0",
        title="test deid redact",
        stages=("output",),
        evaluator=DeidEvaluatorSpec(kind="deid"),
        action=ActionSpec(on_finding="redact"),
    )
    runtime = GuardrailRuntime(definition, deid=model, vault=InMemoryMarkerVault())
    original = CallToolResult(
        content=[], structured_content=["AAA", "banana"], is_error=False
    )

    new_result, entry = asyncio.run(runtime.run_output(original, make_ctx()))

    assert entry.verdict is GuardrailVerdict.REDACTED
    # The reported span alone would leave an "A" / "a" behind.
    assert new_result.structured_content == ["[REDACTED:NAME]", "b[REDACTED:NAME]"]


def test_build_evaluator_deid_without_model_raises():
    with pytest.raises(
        ValueError, match="deid evaluator requires a de-identification model"
    ):
        build_evaluator(DeidEvaluatorSpec(kind="deid"), llm=None, deid=None)
