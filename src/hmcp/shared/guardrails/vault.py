"""Context-scoped marker vault for HMCP 0.1.0 guardrails.

A pseudonymizing guardrail replaces PHI with a marker such as ``[NAME_1]`` and
later restores it, but only inside the patient context that produced it. The
vault maps ``(context_id, category, value)`` to a stable marker and back. PHI
must not outlive its context: entries expire at the context's ``expires_at``
and every call first drops expired contexts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


class MarkerVault(Protocol):
    """Store and resolve context-scoped PHI markers."""

    async def marker_for(
        self, context_id: str, expires_at: datetime, category: str, value: str
    ) -> str: ...

    async def resolve(self, context_id: str, marker: str) -> str | None: ...


class VaultFull(Exception):
    """A context reached its marker limit; the caller must fail closed."""


@dataclass
class _ContextEntries:
    expires_at: datetime
    # (category, value) -> marker, so the same value always maps to one marker
    by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    # marker -> value, for resolution within this context only
    by_marker: dict[str, str] = field(default_factory=dict)
    # category -> last issued counter (n in "[CATEGORY_n]")
    counters: dict[str, int] = field(default_factory=dict)


class InMemoryMarkerVault:
    """A process-local marker vault. No persistence, no cross-context leakage."""

    def __init__(self, *, max_entries_per_context: int = 10_000) -> None:
        self._max_entries_per_context = max_entries_per_context
        self._contexts: dict[str, _ContextEntries] = {}

    def _drop_expired(self) -> None:
        now = datetime.now(timezone.utc)
        for context_id in [
            cid for cid, entries in self._contexts.items() if entries.expires_at <= now
        ]:
            del self._contexts[context_id]

    async def marker_for(
        self, context_id: str, expires_at: datetime, category: str, value: str
    ) -> str:
        self._drop_expired()

        entries = self._contexts.get(context_id)
        if entries is None:
            entries = _ContextEntries(expires_at=expires_at)
            self._contexts[context_id] = entries
        else:
            entries.expires_at = expires_at

        key = (category, value)
        existing = entries.by_value.get(key)
        if existing is not None:
            return existing

        if len(entries.by_value) >= self._max_entries_per_context:
            raise VaultFull(
                f"marker vault full for context ({self._max_entries_per_context} entries)"
            )

        n = entries.counters.get(category, 0) + 1
        entries.counters[category] = n
        marker = f"[{category}_{n}]"
        entries.by_value[key] = marker
        entries.by_marker[marker] = value
        return marker

    async def resolve(self, context_id: str, marker: str) -> str | None:
        self._drop_expired()
        entries = self._contexts.get(context_id)
        if entries is None:
            return None
        return entries.by_marker.get(marker)
