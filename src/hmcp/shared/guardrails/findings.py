"""Findings and text-mutation helpers for HMCP 0.1.0 guardrails."""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Finding:
    """One PHI span in a text unit: its category, half-open span and source."""

    category: str  # a PhiCategory value (or "REDACTION_PLACEHOLDER" for that builtin)
    start: int
    end: int  # exclusive
    source: str  # "pattern" | "prompt" | "deid" | "classifier" | "builtin"
    identifier_type: str | None = None
    # matched text; repr=False so a logged Finding never shows PHI
    value: str = field(default="", repr=False, compare=False)


def merge_overlaps(
    findings: Sequence[Finding], *, text: str | None = None
) -> list[Finding]:
    """Merge overlapping spans into their union; adjacent spans stay separate.

    A union keeps the category/source/identifier_type of its longest constituent
    (ties: earliest). Its ``value`` is ``text[start:end]`` when ``text`` is given,
    else the longest constituent's value.
    """
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    merged: list[Finding] = []
    union_start = 0
    union_end = 0
    representative: Finding | None = None

    def flush() -> None:
        nonlocal union_start, union_end, representative
        if representative is None:
            return
        value = (
            text[union_start:union_end] if text is not None else representative.value
        )
        merged.append(
            Finding(
                category=representative.category,
                start=union_start,
                end=union_end,
                source=representative.source,
                identifier_type=representative.identifier_type,
                value=value,
            )
        )
        representative = None

    for candidate in ordered:
        if representative is not None and candidate.start < union_end:
            union_end = max(union_end, candidate.end)
            if (candidate.end - candidate.start) > (
                representative.end - representative.start
            ):
                representative = candidate
        else:
            flush()
            union_start = candidate.start
            union_end = candidate.end
            representative = candidate
    flush()
    return merged


def locate_all(text: str, needle: str) -> list[tuple[int, int]]:
    """Every non-overlapping exact occurrence of ``needle``; empty needle -> []."""
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        spans.append((index, index + len(needle)))
        start = index + len(needle)
    return spans


def redact(text: str, findings: Sequence[Finding], placeholder: str) -> str:
    """Merge overlaps, then replace right-to-left so earlier spans stay valid."""
    for finding in sorted(
        merge_overlaps(findings, text=text), key=lambda f: f.start, reverse=True
    ):
        text = (
            text[: finding.start]
            + placeholder.format(category=finding.category)
            + text[finding.end :]
        )
    return text
