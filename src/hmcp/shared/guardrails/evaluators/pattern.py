"""Deterministic regex / JSON-key evaluator."""

import re
from typing import Sequence

from ..definition import PatternEvaluatorSpec
from ..findings import Finding, merge_overlaps
from .base import EvaluationContext, TextUnit


class PatternEvaluator:
    """Find PHI by regex text patterns and JSON keys."""

    kind = "pattern"

    def __init__(self, spec: PatternEvaluatorSpec) -> None:
        self._spec = spec
        # Compile once; match case-insensitively.
        self._compiled = [
            (
                identifier_type,
                [re.compile(p, re.IGNORECASE) for p in identifier_type.text_patterns],
            )
            for identifier_type in spec.identifier_types
        ]
        self._json_keys = [
            (identifier_type, {key.casefold() for key in identifier_type.json_keys})
            for identifier_type in spec.identifier_types
        ]

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        results: list[list[Finding]] = []
        for unit in units:
            findings: list[Finding] = []
            for identifier_type, patterns in self._compiled:
                for pattern in patterns:
                    for match in pattern.finditer(unit.text):
                        if match.start(1) < 0:
                            continue  # optional capture group did not participate
                        findings.append(
                            Finding(
                                category=identifier_type.category.value,
                                start=match.start(1),
                                end=match.end(1),
                                source="pattern",
                                identifier_type=identifier_type.name,
                                value=match.group(1),
                            )
                        )
            if unit.key is not None and unit.text.strip():
                key = unit.key.casefold()
                for identifier_type, keys in self._json_keys:
                    if key in keys:
                        findings.append(
                            Finding(
                                category=identifier_type.category.value,
                                start=0,
                                end=len(unit.text),
                                source="pattern",
                                identifier_type=identifier_type.name,
                                value=unit.text,
                            )
                        )
            results.append(merge_overlaps(findings, text=unit.text))
        return results
