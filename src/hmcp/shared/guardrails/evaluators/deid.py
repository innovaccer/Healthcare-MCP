"""De-identification evaluator over an abstract de-identification model.

A ``deid`` guardrail calls a de-identification model that the enforcer controls;
HMCP ships no model. The request/response types mirror a de-identification
service's ``/process/text`` API (text in; labelled entities with character spans
out): a batch of texts goes out as
``{"text": [...], "entity_detection": {"return_entity": true}}`` and a JSON list
comes back with one object per input text, in order. Everything fails closed and
never logs or raises content (notably never a response body).
"""

import re
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
)

from ..definition import DeidEvaluatorSpec, PhiCategory
from ..findings import Finding
from .base import EvaluationContext, EvaluationError, TextUnit

_BATCH_SIZE = 50
_LABEL = re.compile(r"^[A-Z][A-Z0-9_]*$")
_EDGE_PUNCTUATION = ".,;:!?"


class DeidEntityDetection(BaseModel):
    """The de-identification request's entity-detection options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    return_entity: Literal[True] = True


class DeidRequest(BaseModel):
    """A batch of texts to de-identify."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: list[str] = Field(repr=False)
    entity_detection: DeidEntityDetection = DeidEntityDetection()


class DeidEntityLocation(BaseModel):
    """Character span of an entity within its input text."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    stt_idx: StrictInt
    end_idx: StrictInt


class DeidEntity(BaseModel):
    """One labelled entity the de-identification service reported."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    text: StrictStr = Field(repr=False)
    best_label: StrictStr
    labels: dict[str, float] = Field(default_factory=dict)
    location: DeidEntityLocation


class DeidTextResult(BaseModel):
    """The entities reported for one input text."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    # REQUIRED: a missing entities list is an error, never "no PHI".
    entities: list[DeidEntity]


class DeidentificationModel(Protocol):
    """A de-identification service called with a batch of texts."""

    async def process_text(
        self, request: DeidRequest
    ) -> Sequence[DeidTextResult | Mapping[str, Any]]:
        """Return one result per ``request.text`` item, in order.

        For a compatible HTTP service the body to send is
        ``request.model_dump(mode="json")`` and its JSON list response may be
        returned as is. Implementations must not add an entity selection (HMCP
        filters locally with ``DeidEvaluatorSpec.categories``). Never log or
        raise the response body.
        """
        ...


def label_to_category(label: str) -> PhiCategory:
    """Map a de-identification model label (a service's label vocabulary, e.g.
    NAME_GIVEN, DOB, LOCATION_ADDRESS_STREET) to a PhiCategory. The first
    matching rule wins; an unknown label maps to OTHER_ID, so it is still acted
    on.
    """
    label = label.upper()
    if label.startswith("NAME"):
        return PhiCategory.NAME
    if label.startswith("DATE") or label in ("DOB", "AGE"):
        return PhiCategory.DATE
    if label == "PHONE_NUMBER":
        return PhiCategory.PHONE
    if label == "FAX_NUMBER":
        return PhiCategory.FAX
    if label == "EMAIL_ADDRESS":
        return PhiCategory.EMAIL
    if label == "SSN":
        return PhiCategory.SSN
    if label == "MEDICAL_RECORD_NUMBER":
        return PhiCategory.MRN
    if label == "HEALTHCARE_NUMBER":
        return PhiCategory.HEALTH_PLAN_ID
    if label in ("BANK_ACCOUNT", "CREDIT_CARD", "ROUTING_NUMBER", "ACCOUNT_NUMBER"):
        return PhiCategory.ACCOUNT_NUMBER
    if label in ("DRIVER_LICENSE", "PASSPORT_NUMBER"):
        return PhiCategory.LICENSE_NUMBER
    if label == "VEHICLE_ID":
        return PhiCategory.VEHICLE_ID
    if label == "DEVICE_ID":
        return PhiCategory.DEVICE_ID
    if label == "URL":
        return PhiCategory.URL
    if label == "IP_ADDRESS":
        return PhiCategory.IP_ADDRESS
    if label.startswith("LOCATION"):
        return PhiCategory.LOCATION
    return PhiCategory.OTHER_ID


def _occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Every OVERLAPPING occurrence of ``value`` in ``text``; empty value -> []."""
    if not value:
        return []
    spans: list[tuple[int, int]] = []
    index = text.find(value)
    while index >= 0:
        spans.append((index, index + len(value)))
        index = text.find(value, index + 1)
    return spans


class DeidEvaluator:
    """Turn a de-identification model's entities into PHI findings."""

    kind = "deid"

    def __init__(self, spec: DeidEvaluatorSpec, model: DeidentificationModel) -> None:
        self._spec = spec
        self._model = model

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        results: list[list[Finding]] = [[] for _ in units]
        for batch_start in range(0, len(units), _BATCH_SIZE):
            batch = units[batch_start : batch_start + _BATCH_SIZE]
            request = DeidRequest(text=[unit.text for unit in batch])
            model_failed = False
            try:
                raw = await self._model.process_text(request)
            except Exception:  # noqa: BLE001 - fail closed, but keep the cause clean
                model_failed = True
            if model_failed:
                # Raised OUTSIDE the except block so __context__ stays None: a
                # provider message could otherwise carry PHI into the exception.
                raise EvaluationError("deid model failed") from None
            if not isinstance(raw, (list, tuple)) or len(raw) != len(batch):
                raise EvaluationError("invalid deid response")
            for offset, item in enumerate(raw):
                text_index = batch_start + offset
                text = batch[offset].text
                result: DeidTextResult | None
                if isinstance(item, DeidTextResult):
                    result = item
                else:
                    result = None
                    try:
                        result = DeidTextResult.model_validate(item)
                    except (ValidationError, TypeError, ValueError):
                        # A pydantic message embeds the input, which is PHI.
                        result = None
                if result is None:
                    # Raised OUTSIDE the except block so __context__ stays None
                    # (a chained pydantic error would carry PHI).
                    raise EvaluationError("invalid deid response") from None
                findings: list[Finding] = []
                for entity_index, entity in enumerate(result.entities):
                    label = entity.best_label
                    if not _LABEL.fullmatch(label):
                        raise EvaluationError(
                            f"deid entity {entity_index} of text {text_index}: invalid label"
                        )
                    start = entity.location.stt_idx
                    end = entity.location.end_idx
                    if not (0 <= start < end <= len(text)):
                        raise EvaluationError(
                            f"deid entity {entity_index} of text {text_index}: "
                            "span out of bounds"
                        )
                    if text[start:end] != entity.text:
                        raise EvaluationError(
                            f"deid entity {entity_index} of text {text_index}: "
                            "span does not match text"
                        )
                    while end > start and (
                        text[end - 1].isspace() or text[end - 1] in _EDGE_PUNCTUATION
                    ):
                        end -= 1
                    while start < end and text[start].isspace():
                        start += 1
                    if start >= end:
                        continue
                    category = label_to_category(label)
                    if self._spec.categories and category not in self._spec.categories:
                        continue
                    value = text[start:end]
                    # Err toward redaction: the reported span plus every
                    # overlapping occurrence in the text. Duplicate spans are
                    # fine; the engine merges overlaps.
                    spans = [(start, end), *_occurrences(text, value)]
                    for span_start, span_end in spans:
                        findings.append(
                            Finding(
                                category=category.value,
                                start=span_start,
                                end=span_end,
                                source="deid",
                                value=value,
                            )
                        )
                results[text_index] = findings
        return results
