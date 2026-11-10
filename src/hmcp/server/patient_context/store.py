"""Patient-context storage for HMCP 0.1.0.

A context store holds server-minted patient contexts. The default is an
in-memory implementation; deployments may plug in a shared/persistent store by
implementing the :class:`ContextStore` protocol.

The store holds PHI-bearing records, so it never logs or puts a record's
contents into an exception message.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Protocol

from hmcp.shared.types import ContextStatus, PatientContext


class ContextStoreFull(Exception):
    """``put`` could not make room even after purging expired contexts."""


class ContextStore(Protocol):
    """Async storage for :class:`PatientContext` records."""

    async def put(self, ctx: PatientContext) -> None: ...

    async def get(self, context_id: str) -> PatientContext | None: ...

    async def set_status(
        self, context_id: str, status: ContextStatus
    ) -> PatientContext | None:
        """Return the updated context, or ``None`` when the id is unknown."""
        ...

    async def find(
        self,
        *,
        patient_reference: str | None = None,
        subject: str | None = None,
        client_id: str | None = None,
    ) -> list[PatientContext]:
        """Return contexts matching the AND of the given filters.

        At least one filter is required; calling with none raises ``ValueError``
        so a lifecycle sweep cannot accidentally select every context.
        """
        ...

    async def purge_expired(self, now: datetime) -> int:
        """Delete contexts whose ``expires_at <= now`` and return the count."""
        ...


class InMemoryContextStore:
    """Single-process, asyncio-safe implementation of :class:`ContextStore`."""

    def __init__(self, *, max_contexts: int = 10_000) -> None:
        self._max_contexts = max_contexts
        self._contexts: dict[str, PatientContext] = {}
        self._lock = asyncio.Lock()

    async def put(self, ctx: PatientContext) -> None:
        async with self._lock:
            # Re-putting an existing id is exempt from the "full" check: an
            # overwrite replaces a record and does not grow the store.
            if (
                ctx.context_id not in self._contexts
                and len(self._contexts) >= self._max_contexts
            ):
                self._purge_expired(datetime.now(timezone.utc))
                if len(self._contexts) >= self._max_contexts:
                    raise ContextStoreFull(
                        f"context store is full ({self._max_contexts} contexts)"
                    )
            self._contexts[ctx.context_id] = ctx

    async def get(self, context_id: str) -> PatientContext | None:
        async with self._lock:
            return self._contexts.get(context_id)

    async def set_status(
        self, context_id: str, status: ContextStatus
    ) -> PatientContext | None:
        async with self._lock:
            ctx = self._contexts.get(context_id)
            if ctx is None:
                return None
            updated = ctx.model_copy(update={"status": status})
            self._contexts[context_id] = updated
            return updated

    async def find(
        self,
        *,
        patient_reference: str | None = None,
        subject: str | None = None,
        client_id: str | None = None,
    ) -> list[PatientContext]:
        if patient_reference is None and subject is None and client_id is None:
            raise ValueError("find requires at least one filter")
        async with self._lock:
            return [
                ctx
                for ctx in self._contexts.values()
                if (
                    patient_reference is None
                    or ctx.patient.reference == patient_reference
                )
                and (subject is None or ctx.principal.subject == subject)
                and (client_id is None or ctx.principal.client_id == client_id)
            ]

    async def purge_expired(self, now: datetime) -> int:
        async with self._lock:
            return self._purge_expired(now)

    def _purge_expired(self, now: datetime) -> int:
        expired = [
            context_id
            for context_id, ctx in self._contexts.items()
            if ctx.expires_at <= now
        ]
        for context_id in expired:
            del self._contexts[context_id]
        return len(expired)
