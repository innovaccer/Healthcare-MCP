"""Tests for the classifier evaluator. No network: fake classifier providers."""

import asyncio
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from hmcp.shared.guardrails.definition import (
    ClassifierEvaluatorSpec,
    GuardrailDefinition,
)
from hmcp.shared.guardrails.evaluators.base import (
    EvaluationContext,
    EvaluationError,
    TextUnit,
)
from hmcp.shared.guardrails.evaluators.classifier import (
    ClassificationRequest,
    ClassificationResult,
    ClassifierEvaluator,
)


def _ctx(stage="input"):
    return EvaluationContext(patient_context=None, stage=stage, tool_name="demo_tool")


def _spec(**overrides):
    values = {"kind": "classifier", "classifier": "prompt-injection"}
    values.update(overrides)
    return ClassifierEvaluatorSpec(**values)


class FakeClassifier:
    """Records requests; returns or raises per call (index-based callable or fixed)."""

    def __init__(self, replies):
        self._replies = replies
        self.requests: list[ClassificationRequest] = []

    async def classify(self, request):
        index = len(self.requests)
        self.requests.append(request)
        if callable(self._replies):
            return self._replies(index)
        return self._replies


class RaisingClassifier:
    async def classify(self, request):
        raise RuntimeError("SECRET-CONTENT-123")


# --- spec -----------------------------------------------------------------


def test_spec_round_trips_and_default():
    spec = ClassifierEvaluatorSpec.model_validate(
        {
            "kind": "classifier",
            "classifier": "prompt-injection",
            "maxCharsPerCall": 4000,
        }
    )
    assert spec.classifier == "prompt-injection"
    assert spec.max_chars_per_call == 4000

    dumped = spec.model_dump(by_alias=True)
    assert dumped["maxCharsPerCall"] == 4000
    assert ClassifierEvaluatorSpec.model_validate(dumped) == spec

    default = _spec(classifier="toxicity")
    assert default.max_chars_per_call == 12000


@pytest.mark.parametrize("name", ["", "Prompt", "-x", "a b", "a" * 129])
def test_invalid_classifier_names(name):
    with pytest.raises(ValidationError):
        _spec(classifier=name)


@pytest.mark.parametrize("name", ["prompt-injection", "toxicity.v2", "a_b"])
def test_valid_classifier_names(name):
    assert _spec(classifier=name).classifier == name


@pytest.mark.parametrize("value", [499, 200001])
def test_max_chars_per_call_bounds(value):
    with pytest.raises(ValidationError):
        _spec(max_chars_per_call=value)


def test_spec_extra_field_rejected():
    with pytest.raises(ValidationError):
        ClassifierEvaluatorSpec.model_validate(
            {"kind": "classifier", "classifier": "toxicity", "unexpected": 1}
        )


# --- definition cross-validation ------------------------------------------


def _classifier_definition(**overrides):
    base = {
        "id": "com.example/injection-guard",
        "version": "1.0.0",
        "title": "Injection guard",
        "stages": ["input"],
        "evaluator": {"kind": "classifier", "classifier": "prompt-injection"},
        "action": {"onFinding": "block"},
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("action", ["redact", "pseudonymize"])
def test_classifier_requires_block_or_annotate(action):
    with pytest.raises(ValidationError) as excinfo:
        GuardrailDefinition.model_validate(
            _classifier_definition(action={"onFinding": action})
        )
    assert (
        "classifier evaluators require action.onFinding 'block' or 'annotate'"
        in str(excinfo.value)
    )


@pytest.mark.parametrize("action", ["block", "annotate"])
def test_classifier_accepts_block_and_annotate(action):
    definition = GuardrailDefinition.model_validate(
        _classifier_definition(action={"onFinding": action})
    )
    assert definition.action.on_finding == action


def test_cascade_rejects_classifier_step():
    with pytest.raises(ValidationError):
        GuardrailDefinition.model_validate(
            _classifier_definition(
                evaluator={
                    "kind": "cascade",
                    "steps": [
                        {"kind": "classifier", "classifier": "prompt-injection"},
                        {"kind": "deid", "categories": ["NAME"]},
                    ],
                }
            )
        )


# --- find() semantics ------------------------------------------------------


async def test_not_flagged_no_findings_and_request_fields():
    provider = FakeClassifier(ClassificationResult(flagged=False))
    evaluator = ClassifierEvaluator(_spec(), provider)
    units = [TextUnit("hello"), TextUnit("world")]

    for stage in ("input", "output"):
        provider.requests.clear()
        results = await evaluator.find(units, _ctx(stage=stage))

        assert results == [[], []]
        assert len(provider.requests) == 1
        request = provider.requests[0]
        assert request.classifier == "prompt-injection"
        assert request.stage == stage
        assert request.text == "\n---\n".join(["hello", "world"])


async def test_flagged_single_chunk_finding_on_first_non_empty_unit():
    provider = FakeClassifier(ClassificationResult(flagged=True))
    evaluator = ClassifierEvaluator(_spec(classifier="toxicity"), provider)
    units = [TextUnit(""), TextUnit("flagged text")]

    results = await evaluator.find(units, _ctx())

    assert results[0] == []
    assert len(results[1]) == 1
    finding = results[1][0]
    assert finding.category == "CLASSIFIER"
    assert finding.source == "classifier"
    assert (finding.start, finding.end) == (0, len("flagged text"))
    assert finding.value == ""


async def test_multi_chunk_flagged_second_chunk():
    provider = FakeClassifier(lambda index: ClassificationResult(flagged=index == 1))
    evaluator = ClassifierEvaluator(_spec(max_chars_per_call=500), provider)
    units = [TextUnit("A" * 300), TextUnit("B" * 300), TextUnit("C" * 300)]

    results = await evaluator.find(units, _ctx())

    assert len(provider.requests) == 3
    assert results[0] == []
    assert len(results[1]) == 1
    assert results[1][0].end == 300
    assert results[2] == []
    assert sum(len(result) for result in results) == 1


async def test_blank_text_not_sent():
    provider = FakeClassifier(ClassificationResult(flagged=True))
    evaluator = ClassifierEvaluator(_spec(), provider)

    assert await evaluator.find([TextUnit("")], _ctx()) == [[]]
    assert await evaluator.find([TextUnit("   ")], _ctx()) == [[]]
    assert provider.requests == []


async def test_whitespace_only_multi_unit_chunk_not_sent():
    provider = FakeClassifier(ClassificationResult(flagged=True))
    evaluator = ClassifierEvaluator(_spec(), provider)

    assert await evaluator.find([TextUnit(""), TextUnit("")], _ctx()) == [[], []]
    assert await evaluator.find([TextUnit("  "), TextUnit("  ")], _ctx()) == [[], []]
    assert await evaluator.find([TextUnit(""), TextUnit(" \n ")], _ctx()) == [[], []]
    assert provider.requests == []


async def test_mixed_chunk_with_leading_whitespace_is_sent():
    provider = FakeClassifier(ClassificationResult(flagged=True))
    evaluator = ClassifierEvaluator(_spec(), provider)

    results = await evaluator.find([TextUnit("  "), TextUnit("real text")], _ctx())

    assert len(provider.requests) == 1
    assert results[0] == []
    assert len(results[1]) == 1


async def test_finding_on_first_unit_with_non_whitespace_text():
    provider = FakeClassifier(ClassificationResult(flagged=True))
    evaluator = ClassifierEvaluator(_spec(), provider)

    results = await evaluator.find(
        [TextUnit("  "), TextUnit("x"), TextUnit("y")], _ctx()
    )

    assert results[0] == []
    assert len(results[1]) == 1
    assert (results[1][0].start, results[1][0].end) == (0, 1)
    assert results[2] == []


async def test_no_units_returns_empty_and_no_call():
    provider = FakeClassifier(ClassificationResult(flagged=True))
    evaluator = ClassifierEvaluator(_spec(), provider)

    assert await evaluator.find([], _ctx()) == []
    assert provider.requests == []


async def test_provider_failure_is_fixed_message_unchained():
    evaluator = ClassifierEvaluator(_spec(), RaisingClassifier())

    with pytest.raises(EvaluationError) as excinfo:
        await evaluator.find([TextUnit("some text")], _ctx())

    exc = excinfo.value
    assert str(exc) == "classifier call failed"
    assert exc.__context__ is None
    assert exc.__cause__ is None
    assert "SECRET-CONTENT-123" not in repr(exc)
    assert "SECRET-CONTENT-123" not in str(exc)


@pytest.mark.parametrize(
    "reply", [{"flagged": "true"}, {}, {"flagged": 1}, None, 1, "yes"]
)
async def test_invalid_results_raise(reply):
    evaluator = ClassifierEvaluator(_spec(), FakeClassifier(reply))

    with pytest.raises(EvaluationError) as excinfo:
        await evaluator.find([TextUnit("text")], _ctx())

    exc = excinfo.value
    assert str(exc) == "invalid classifier result"
    assert exc.__context__ is None
    assert exc.__cause__ is None


async def test_extra_fields_ignored():
    evaluator = ClassifierEvaluator(
        _spec(), FakeClassifier({"flagged": True, "extra": 1})
    )

    results = await evaluator.find([TextUnit("text")], _ctx())

    assert len(results[0]) == 1


class BadMapping(Mapping):
    """A Mapping whose iteration raises, to prove find() fails closed."""

    def __getitem__(self, key):
        raise RuntimeError("SECRET-MAP-1")

    def __iter__(self):
        raise RuntimeError("SECRET-MAP-1")

    def __len__(self):
        return 0


async def test_mapping_that_raises_does_not_escape():
    evaluator = ClassifierEvaluator(_spec(), FakeClassifier(BadMapping()))

    with pytest.raises(EvaluationError) as excinfo:
        await evaluator.find([TextUnit("text")], _ctx())

    exc = excinfo.value
    assert str(exc) == "invalid classifier result"
    assert exc.__context__ is None
    assert exc.__cause__ is None
    assert "SECRET-MAP-1" not in repr(exc)
    assert "SECRET-MAP-1" not in str(exc)


async def test_one_failing_chunk_among_several_fails_whole_find():
    def reply(index):
        if index == 1:
            raise RuntimeError("SECRET-CONTENT-123")
        return ClassificationResult(flagged=True)

    provider = FakeClassifier(reply)
    evaluator = ClassifierEvaluator(_spec(max_chars_per_call=500), provider)
    units = [TextUnit("A" * 300), TextUnit("B" * 300), TextUnit("C" * 300)]

    with pytest.raises(EvaluationError) as excinfo:
        await evaluator.find(units, _ctx())

    assert str(excinfo.value) == "classifier call failed"
    assert excinfo.value.__context__ is None
    assert excinfo.value.__cause__ is None


class ConcurrencyClassifier:
    """Records the maximum number of classify calls in flight at once."""

    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    async def classify(self, request):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls += 1
        try:
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        return ClassificationResult(flagged=False)


async def test_concurrency_cap():
    provider = ConcurrencyClassifier()
    evaluator = ClassifierEvaluator(_spec(max_chars_per_call=500), provider)
    units = [TextUnit(f"{index}" + "x" * 300) for index in range(10)]

    await evaluator.find(units, _ctx())

    assert provider.calls == 10
    assert 2 <= provider.max_in_flight <= 4


def test_request_repr_hides_text():
    request = ClassificationRequest(
        classifier="toxicity", stage="input", text="SECRET-CONTENT-123"
    )
    assert "SECRET-CONTENT-123" not in repr(request)
