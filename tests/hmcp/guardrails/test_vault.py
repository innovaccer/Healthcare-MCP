"""Tests for the context-scoped marker vault. No network."""

import asyncio
from datetime import datetime, timezone

import pytest

from hmcp.shared.guardrails.vault import InMemoryMarkerVault, VaultFull

FUTURE = datetime(2030, 1, 1, tzinfo=timezone.utc)
PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)


def test_same_value_returns_same_marker():
    vault = InMemoryMarkerVault()

    async def run():
        first = await vault.marker_for("c1", FUTURE, "NAME", "Testpatient Alpha")
        again = await vault.marker_for("c1", FUTURE, "NAME", "Testpatient Alpha")
        other = await vault.marker_for("c1", FUTURE, "NAME", "Testpatient Beta")
        return first, again, other

    first, again, other = asyncio.run(run())
    assert first == again == "[NAME_1]"
    assert other == "[NAME_2]"


def test_counters_are_per_category():
    vault = InMemoryMarkerVault()

    async def run():
        return (
            await vault.marker_for("c1", FUTURE, "NAME", "A"),
            await vault.marker_for("c1", FUTURE, "DATE", "B"),
            await vault.marker_for("c1", FUTURE, "NAME", "C"),
        )

    assert asyncio.run(run()) == ("[NAME_1]", "[DATE_1]", "[NAME_2]")


def test_resolve_roundtrip():
    vault = InMemoryMarkerVault()

    async def run():
        marker = await vault.marker_for("c1", FUTURE, "NAME", "Testpatient Alpha")
        return await vault.resolve("c1", marker)

    assert asyncio.run(run()) == "Testpatient Alpha"


def test_other_context_never_resolves():
    vault = InMemoryMarkerVault()

    async def run():
        marker = await vault.marker_for("c1", FUTURE, "NAME", "A")
        return await vault.resolve("c2", marker)

    assert asyncio.run(run()) is None


def test_expired_context_is_dropped_on_the_next_call():
    vault = InMemoryMarkerVault()

    async def run():
        marker = await vault.marker_for("c1", PAST, "NAME", "A")
        return await vault.resolve("c1", marker)

    assert asyncio.run(run()) is None


def test_full_context_raises_vault_full():
    vault = InMemoryMarkerVault(max_entries_per_context=1)

    async def run():
        await vault.marker_for("c1", FUTURE, "NAME", "A")
        await vault.marker_for("c1", FUTURE, "NAME", "B")

    with pytest.raises(VaultFull):
        asyncio.run(run())


def test_full_context_still_returns_known_marker():
    vault = InMemoryMarkerVault(max_entries_per_context=1)

    async def run():
        first = await vault.marker_for("c1", FUTURE, "NAME", "A")
        again = await vault.marker_for("c1", FUTURE, "NAME", "A")
        return first, again

    assert asyncio.run(run()) == ("[NAME_1]", "[NAME_1]")
