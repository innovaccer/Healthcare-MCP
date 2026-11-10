"""Evaluator contracts shared by every guardrail evaluator."""

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from hmcp.shared.types import GuardrailStage, PatientContext

from ..findings import Finding


class EvaluationError(Exception):
    """An evaluator could not produce a verdict. Its message is a FIXED string — never content."""


@dataclass(frozen=True)
class TextUnit:
    """A string to inspect, with the dict key it sits under (None if top level)."""

    text: str = field(repr=False)
    # dict key the string sits under (None for top-level / list items without a parent key)
    key: str | None = None


@dataclass(frozen=True)
class EvaluationContext:
    """The patient context, stage and tool name an evaluation runs under."""

    patient_context: PatientContext | None
    stage: GuardrailStage
    tool_name: str


class Evaluator(Protocol):
    """An evaluator turns text units into findings for each unit."""

    kind: str  # e.g. "pattern", "prompt", "deid", "classifier", "builtin:redaction_placeholder"

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        """Return exactly one list of findings per unit, in the same order."""
        ...
