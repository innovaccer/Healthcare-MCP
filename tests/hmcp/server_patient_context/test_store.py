"""Tests for the in-memory patient-context store."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from hmcp.server.patient_context.store import ContextStoreFull, InMemoryContextStore
from hmcp.shared.types import (
    ContextStatus,
    Identifier,
    PatientContext,
    PatientRef,
    Principal,
)

pytestmark = pytest.mark.asyncio

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_ctx(
    context_id: str = "pc_1",
    *,
    reference: str = "Patient/test-0001",
    subject: str = "user-a",
    client_id: str = "client-a",
    status: ContextStatus = ContextStatus.ACTIVE,
    expires_at: datetime | None = None,
    identifiers: tuple[Identifier, ...] = (),
) -> PatientContext:
    return PatientContext(
        context_id=context_id,
        patient=PatientRef(reference=reference, identifiers=identifiers),
        purpose_of_use="TREAT",
        principal=Principal(client_id=client_id, subject=subject),
        status=status,
        created_at=NOW,
        expires_at=expires_at or (NOW + timedelta(hours=1)),
    )


async def test_put_get_round_trip() -> None:
    store = InMemoryContextStore()
    ctx = make_ctx()
    await store.put(ctx)
    assert await store.get("pc_1") == ctx


async def test_get_unknown_returns_none() -> None:
    store = InMemoryContextStore()
    assert await store.get("pc_missing") is None


async def test_set_status_returns_updated_copy_and_get_reflects() -> None:
    store = InMemoryContextStore()
    original = make_ctx()
    await store.put(original)

    updated = await store.set_status("pc_1", ContextStatus.SUSPENDED)

    assert updated is not None
    assert updated.status is ContextStatus.SUSPENDED
    assert original.status is ContextStatus.ACTIVE
    assert updated is not original
    stored = await store.get("pc_1")
    assert stored is not original
    assert stored.status is ContextStatus.SUSPENDED


async def test_set_status_unknown_returns_none() -> None:
    store = InMemoryContextStore()
    assert await store.set_status("pc_missing", ContextStatus.CLOSED) is None


async def test_find_by_each_filter() -> None:
    store = InMemoryContextStore()
    await store.put(
        make_ctx(
            "pc_a",
            reference="Patient/test-0001",
            subject="user-a",
            client_id="client-a",
        )
    )
    await store.put(
        make_ctx(
            "pc_b",
            reference="Patient/test-0002",
            subject="user-b",
            client_id="client-b",
        )
    )

    by_patient = await store.find(patient_reference="Patient/test-0001")
    by_subject = await store.find(subject="user-b")
    by_client = await store.find(client_id="client-a")

    assert [c.context_id for c in by_patient] == ["pc_a"]
    assert [c.context_id for c in by_subject] == ["pc_b"]
    assert [c.context_id for c in by_client] == ["pc_a"]


async def test_find_combined_filters_are_anded() -> None:
    store = InMemoryContextStore()
    await store.put(
        make_ctx(
            "pc_a",
            reference="Patient/test-0001",
            subject="user-a",
            client_id="client-a",
        )
    )
    await store.put(
        make_ctx(
            "pc_b",
            reference="Patient/test-0001",
            subject="user-b",
            client_id="client-a",
        )
    )

    found = await store.find(
        patient_reference="Patient/test-0001", subject="user-a", client_id="client-a"
    )

    assert [c.context_id for c in found] == ["pc_a"]


async def test_find_with_no_filter_raises() -> None:
    store = InMemoryContextStore()
    with pytest.raises(ValueError):
        await store.find()


async def test_purge_expired_removes_only_expired_and_returns_count() -> None:
    store = InMemoryContextStore()
    now = datetime.now(timezone.utc)
    await store.put(make_ctx("pc_expired", expires_at=now - timedelta(hours=1)))
    await store.put(make_ctx("pc_live", expires_at=now + timedelta(hours=1)))

    removed = await store.purge_expired(now)

    assert removed == 1
    assert await store.get("pc_expired") is None
    assert await store.get("pc_live") is not None


async def test_put_when_full_purges_expired_first() -> None:
    store = InMemoryContextStore(max_contexts=2)
    now = datetime.now(timezone.utc)
    await store.put(make_ctx("pc_expired", expires_at=now - timedelta(hours=1)))
    await store.put(make_ctx("pc_live", expires_at=now + timedelta(hours=1)))

    await store.put(make_ctx("pc_new", expires_at=now + timedelta(hours=2)))

    assert await store.get("pc_new") is not None
    assert await store.get("pc_expired") is None
    assert await store.get("pc_live") is not None


async def test_put_when_full_and_still_full_raises() -> None:
    store = InMemoryContextStore(max_contexts=2)
    now = datetime.now(timezone.utc)
    await store.put(make_ctx("pc_1", expires_at=now + timedelta(hours=1)))
    await store.put(make_ctx("pc_2", expires_at=now + timedelta(hours=1)))

    with pytest.raises(ContextStoreFull):
        await store.put(make_ctx("pc_3", expires_at=now + timedelta(hours=1)))


async def test_concurrent_puts_all_land() -> None:
    store = InMemoryContextStore()

    await asyncio.gather(*(store.put(make_ctx(f"pc_{i}")) for i in range(50)))

    for i in range(50):
        assert await store.get(f"pc_{i}") is not None
