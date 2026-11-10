"""Text-classifier evaluator for HMCP 0.1.0 guardrails.

A ``classifier`` guardrail calls a deployer-supplied classifier provider, for
example an SLM service that flags prompt injection or toxicity for a whole text.
HMCP ships no classifier: the enforcing party supplies one. Content is sent in
chunks (packed so each call stays under ``maxCharsPerCall``), each chunk is one
provider call, and a flagged chunk becomes one finding on its first non-empty
unit. Calls run concurrently under a per-evaluation cap (a provider that needs a
global limit applies its own). Everything fails closed and never logs or raises
content, results or a provider's exception message.
"""

import asyncio
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from hmcp.shared.types import GuardrailStage

from ..definition import ClassifierEvaluatorSpec
from ..findings import Finding
from .base import EvaluationContext, EvaluationError, TextUnit
from .prompt import _CHUNK_SEPARATOR, _chunk

# At most this many provider calls are in flight per evaluation (one stage of
# one tool call); a provider that needs a global limit applies its own.
_MAX_CONCURRENT_CALLS = 4

# Per-chunk outcome markers. A sentinel keeps a failure distinguishable from a
# valid result without carrying any content.
_CALL_FAILED = object()
_INVALID = object()


class ClassificationRequest(BaseModel):
    """One chunk of text to classify with a named classifier at a stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    classifier: str
    stage: GuardrailStage
    text: str = Field(repr=False)


class ClassificationResult(BaseModel):
    """A classifier's verdict for one chunk."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    # REQUIRED: a missing value is an error, never "not flagged".
    flagged: StrictBool


class ClassifierProvider(Protocol):
    """The classifier a deployer supplies for a classifier-kind guardrail.

    HMCP builds the request itself and calls ``classify`` once per chunk. Return
    ``flagged=True`` when the text should be acted on. Raise for an unknown
    classifier. Never log or raise the text or a response body.
    """

    async def classify(
        self, request: ClassificationRequest
    ) -> ClassificationResult | Mapping[str, Any]:
        """Classify request.text with the classifier named request.classifier at request.stage.

        Return flagged=True when the text should be acted on. Raise for an unknown
        classifier. Never log or raise the text or a response body.
        """
        ...


class ClassifierEvaluator:
    """Flag whole chunks of inspected text with a deployer-supplied classifier."""

    kind = "classifier"

    def __init__(
        self, spec: ClassifierEvaluatorSpec, provider: ClassifierProvider
    ) -> None:
        self._spec = spec
        self._provider = provider

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        results: list[list[Finding]] = [[] for _ in units]
        if not units:
            return results

        # (offset into units, chunk, joined text) for chunks worth sending.
        work: list[tuple[int, list[TextUnit], str]] = []
        offset = 0
        for chunk in _chunk(units, self._spec.max_chars_per_call):
            text = _CHUNK_SEPARATOR.join(unit.text for unit in chunk)
            if any(unit.text.strip() for unit in chunk):
                work.append((offset, chunk, text))
            offset += len(chunk)

        if not work:
            return results

        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CALLS)

        async def classify_chunk(text: str) -> object:
            async with semaphore:
                request = ClassificationRequest(
                    classifier=self._spec.classifier, stage=ctx.stage, text=text
                )
                call_failed = False
                reply: object = None
                try:
                    reply = await self._provider.classify(request)
                except (
                    Exception
                ):  # noqa: BLE001 - fail closed, but keep the cause clean
                    call_failed = True
                if call_failed:
                    return _CALL_FAILED
                if isinstance(reply, ClassificationResult):
                    return reply
                if isinstance(reply, Mapping):
                    result: ClassificationResult | None = None
                    invalid = False
                    try:
                        result = ClassificationResult.model_validate(dict(reply))
                    except Exception:  # noqa: BLE001 - a bad Mapping must not escape
                        invalid = True
                    if not invalid:
                        return result
                return _INVALID

        outcomes = await asyncio.gather(*(classify_chunk(text) for _, _, text in work))

        if any(outcome is _CALL_FAILED for outcome in outcomes):
            # Raised OUTSIDE any except block so __context__ stays None: a
            # provider message could otherwise carry content into the exception.
            raise EvaluationError("classifier call failed") from None
        if any(outcome is _INVALID for outcome in outcomes):
            # Raised OUTSIDE any except block so __context__ stays None (a
            # chained pydantic error would carry the response body).
            raise EvaluationError("invalid classifier result") from None

        for (chunk_offset, chunk, _), outcome in zip(work, outcomes):
            if not outcome.flagged:
                continue
            for unit_index, unit in enumerate(chunk):
                if unit.text.strip():
                    results[chunk_offset + unit_index].append(
                        Finding(
                            category="CLASSIFIER",
                            start=0,
                            end=len(unit.text),
                            source="classifier",
                            value="",
                        )
                    )
                    break

        return results
