"""Tests for the prompt evaluator. No network: fake LLM providers."""

import traceback

import pytest

from hmcp.shared.guardrails.definition import PhiCategory, PromptEvaluatorSpec
from hmcp.shared.guardrails.evaluators.base import (
    EvaluationContext,
    EvaluationError,
    TextUnit,
)
from hmcp.shared.guardrails.evaluators.prompt import PromptEvaluator


class FakeLLM:
    def __init__(self, reply):
        self._reply = reply
        self.calls: list[tuple[str, str, int]] = []

    async def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        self.calls.append((system, user, max_tokens))
        if callable(self._reply):
            return self._reply(len(self.calls) - 1)
        return self._reply


class RaisingLLM:
    async def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        raise RuntimeError("provider exploded with ssn 123-45-6789")


def make_spec(**overrides) -> PromptEvaluatorSpec:
    values = {
        "kind": "prompt",
        "system": "system {{nonce}}",
        "user": "cats={{categories}} nonce={{nonce}} body={{content}}",
        "categories": (PhiCategory.NAME, PhiCategory.PHONE),
    }
    values.update(overrides)
    return PromptEvaluatorSpec(**values)


def make_ctx() -> EvaluationContext:
    return EvaluationContext(patient_context=None, stage="input", tool_name="demo_tool")


def test_nonce_in_both_and_differs_between_calls():
    spec = make_spec(
        system="{{nonce}}", user="{{nonce}}|{{content}}", max_chars_per_call=500
    )
    llm = FakeLLM('{"findings": []}')
    evaluator = PromptEvaluator(spec, llm)

    units = [TextUnit("A" * 300), TextUnit("B" * 300)]
    # joined = 300 + len("\n---\n") + 300 = 605 chars > 500 -> two calls
    import asyncio

    asyncio.run(evaluator.find(units, make_ctx()))

    assert len(llm.calls) == 2
    nonces = []
    for system, user, _ in llm.calls:
        nonce = system
        nonces.append(nonce)
        assert nonce in user
        assert len(nonce) == 16
        int(nonce, 16)  # hex
    assert nonces[0] != nonces[1]


def test_categories_rendered():
    spec = make_spec()
    llm = FakeLLM('{"findings": []}')
    evaluator = PromptEvaluator(spec, llm)

    import asyncio

    asyncio.run(evaluator.find([TextUnit("body")], make_ctx()))

    _, user, _ = llm.calls[0]
    assert "NAME, PHONE" in user


def test_content_literal_nonce_not_expanded():
    spec = make_spec(system="{{nonce}}", user="{{nonce}}|{{content}}")
    llm = FakeLLM('{"findings": []}')
    evaluator = PromptEvaluator(spec, llm)

    import asyncio

    asyncio.run(evaluator.find([TextUnit("hello {{nonce}} world")], make_ctx()))

    nonce, user, _ = llm.calls[0]
    # The literal placeholder from content survives untouched.
    assert "{{nonce}}" in user
    # Only the template's own placeholder was expanded.
    assert user.count(nonce) == 1


def test_fenced_json_reply_parses():
    spec = make_spec(categories=(PhiCategory.NAME,))
    reply = '```json\n{"findings": [{"text": "Testpatient Alpha", "category": "NAME"}]}\n```'
    evaluator = PromptEvaluator(spec, FakeLLM(reply))

    text = "Testpatient Alpha arrived"
    import asyncio

    results = asyncio.run(evaluator.find([TextUnit(text)], make_ctx()))

    assert len(results[0]) == 1
    finding = results[0][0]
    assert finding.category == "NAME"
    assert finding.source == "prompt"
    assert (finding.start, finding.end) == (0, len("Testpatient Alpha"))
    assert text[finding.start : finding.end] == "Testpatient Alpha"


def test_findings_located_in_right_unit_with_right_spans():
    spec = make_spec(categories=(PhiCategory.NAME, PhiCategory.PHONE))
    reply = (
        '{"findings": ['
        '{"text": "Testpatient Alpha", "category": "NAME"},'
        '{"text": "555-0123", "category": "PHONE"}]}'
    )
    evaluator = PromptEvaluator(spec, FakeLLM(reply))

    unit_a = TextUnit("x 555-0123")
    unit_b = TextUnit("Testpatient Alpha")
    import asyncio

    results = asyncio.run(evaluator.find([unit_a, unit_b], make_ctx()))

    assert len(results) == 2
    assert [f.category for f in results[0]] == ["PHONE"]
    assert (results[0][0].start, results[0][0].end) == (2, 2 + len("555-0123"))
    assert [f.category for f in results[1]] == ["NAME"]
    assert (results[1][0].start, results[1][0].end) == (0, len("Testpatient Alpha"))


def test_category_outside_spec_is_ignored():
    spec = make_spec(categories=(PhiCategory.NAME,))
    reply = '{"findings": [{"text": "555-0123", "category": "PHONE"}]}'
    evaluator = PromptEvaluator(spec, FakeLLM(reply))

    import asyncio

    results = asyncio.run(evaluator.find([TextUnit("call 555-0123")], make_ctx()))
    assert results == [[]]


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        None,  # non-str reply
        b'{"findings": []}',  # bytes reply
        '{"other": []}',
        '{"findings": [{"category": "NAME"}]}',  # missing text
        '{"findings": [{"text": "", "category": "NAME"}]}',  # empty text
        '{"findings": [{"text": "x", "category": 5}]}',  # non-str category
        '{"findings": ["nope"]}',  # item not a dict
    ],
)
def test_invalid_replies_raise(reply):
    spec = make_spec()
    evaluator = PromptEvaluator(spec, FakeLLM(reply))

    import asyncio

    with pytest.raises(EvaluationError):
        asyncio.run(evaluator.find([TextUnit("some text")], make_ctx()))


def test_llm_failure_is_fixed_message_with_no_cause():
    spec = make_spec()
    evaluator = PromptEvaluator(spec, RaisingLLM())

    import asyncio

    with pytest.raises(EvaluationError) as excinfo:
        asyncio.run(evaluator.find([TextUnit("some text")], make_ctx()))

    exc = excinfo.value
    assert str(exc) == "LLM call failed"
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert exc.__suppress_context__ is True
    assert "123-45-6789" not in "".join(traceback.format_exception(exc))


def test_invalid_reply_with_phi_has_no_chained_context():
    spec = make_spec()
    # Invalid JSON (trailing comma) that carries the synthetic SSN, so the
    # decoder error must not be chained onto the EvaluationError.
    reply = '{"findings": [{"text": "ssn 123-45-6789", "category": "NAME"},]}'
    evaluator = PromptEvaluator(spec, FakeLLM(reply))

    import asyncio

    with pytest.raises(EvaluationError) as excinfo:
        asyncio.run(evaluator.find([TextUnit("some text")], make_ctx()))

    exc = excinfo.value
    assert str(exc) == "invalid LLM verdict"
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert exc.__suppress_context__ is True
    assert "123-45-6789" not in "".join(traceback.format_exception(exc))


def test_chunking_splits_and_keeps_oversized_unit_whole():
    import asyncio

    # Three units that exceed max_chars together -> more than one call.
    spec = make_spec(max_chars_per_call=500)
    llm = FakeLLM('{"findings": []}')
    evaluator = PromptEvaluator(spec, llm)
    units = [TextUnit("A" * 200), TextUnit("B" * 200), TextUnit("C" * 200)]
    asyncio.run(evaluator.find(units, make_ctx()))
    assert len(llm.calls) >= 2

    # A single oversized unit is sent whole, in one call.
    spec = make_spec(max_chars_per_call=500)
    llm = FakeLLM('{"findings": []}')
    evaluator = PromptEvaluator(spec, llm)
    oversized = "X" * 1000
    asyncio.run(evaluator.find([TextUnit(oversized)], make_ctx()))
    assert len(llm.calls) == 1
    assert oversized in llm.calls[0][1]
