"""HMCP-defined builtin evaluators."""

import re
from typing import Sequence

from ..definition import IdentifierType, PhiCategory
from ..findings import Finding
from .base import EvaluationContext, EvaluationError, TextUnit

_REDACTION_PLACEHOLDER = re.compile(r"\[REDACTED:[A-Z_]+\]")
_PATIENT_REFERENCE = re.compile(r"\bPatient/([A-Za-z0-9\-.]{1,64})")


class RedactionPlaceholderEvaluator:
    """Refuse write-back of already-redacted text into a system of record."""

    kind = "builtin:redaction_placeholder"

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        results: list[list[Finding]] = []
        for unit in units:
            results.append(
                [
                    Finding(
                        category="REDACTION_PLACEHOLDER",
                        start=match.start(),
                        end=match.end(),
                        source="builtin",
                        value=match.group(0),
                    )
                    for match in _REDACTION_PLACEHOLDER.finditer(unit.text)
                ]
            )
        return results


class PatientContextConsistencyEvaluator:
    """Flag patient identifiers in the payload that do not match the active context."""

    kind = "builtin:patient_context_consistency"

    def __init__(self, identifier_types: Sequence[IdentifierType] = ()) -> None:
        self._identifier_types = tuple(identifier_types)
        self._access_key_patterns = [
            (
                identifier_type,
                [re.compile(p, re.IGNORECASE) for p in identifier_type.text_patterns],
            )
            for identifier_type in self._identifier_types
            if identifier_type.is_access_key
        ]
        self._access_key_json_keys = [
            (identifier_type, {key.casefold() for key in identifier_type.json_keys})
            for identifier_type in self._identifier_types
            if identifier_type.is_access_key
        ]

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        patient_context = ctx.patient_context
        if patient_context is None:
            raise EvaluationError("no patient context")

        allowed = {patient_context.patient.reference.split("/", 1)[1].upper()}
        allowed.update(
            identifier.value.upper()
            for identifier in patient_context.patient.identifiers
        )

        results: list[list[Finding]] = []
        for unit in units:
            findings: list[Finding] = []
            for match in _PATIENT_REFERENCE.finditer(unit.text):
                value = match.group(1)
                if value.upper() not in allowed:
                    findings.append(
                        Finding(
                            category=PhiCategory.OTHER_ID.value,
                            start=match.start(1),
                            end=match.end(1),
                            source="builtin",
                            value=value,
                        )
                    )
            for identifier_type, patterns in self._access_key_patterns:
                for pattern in patterns:
                    for match in pattern.finditer(unit.text):
                        if match.start(1) < 0:
                            continue  # optional capture group did not participate
                        value = match.group(1)
                        if value.upper() not in allowed:
                            findings.append(
                                Finding(
                                    category=identifier_type.category.value,
                                    start=match.start(1),
                                    end=match.end(1),
                                    source="builtin",
                                    identifier_type=identifier_type.name,
                                    value=value,
                                )
                            )
            if unit.key is not None and unit.text.strip():
                key = unit.key.casefold()
                for identifier_type, keys in self._access_key_json_keys:
                    if key in keys and unit.text.upper() not in allowed:
                        findings.append(
                            Finding(
                                category=identifier_type.category.value,
                                start=0,
                                end=len(unit.text),
                                source="builtin",
                                identifier_type=identifier_type.name,
                                value=unit.text,
                            )
                        )
            results.append(findings)
        return results
