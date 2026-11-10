"""Tests for SmartLaunchPolicy, the default access policy."""

from __future__ import annotations

import pytest
from mcp.server.auth.provider import AccessToken

from hmcp.server.patient_context.policy import SmartLaunchPolicy
from hmcp.shared.types import Identifier, PatientRef, Principal

pytestmark = pytest.mark.asyncio

PRINCIPAL = Principal(client_id="client-a", subject="user-a")


def make_token(
    *,
    claims: dict | None = None,
    scopes: list[str] | None = None,
) -> AccessToken:
    return AccessToken(
        token="test-token",
        client_id="client-a",
        scopes=scopes if scopes is not None else ["user/Patient.read"],
        expires_at=None,
        resource=None,
        subject="user-a",
        claims=claims,
    )


def patient(reference: str = "Patient/test-0001") -> PatientRef:
    return PatientRef(reference=reference)


# --- authorize_open ---------------------------------------------------------


async def test_matching_patient_claim_allowed() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_open(
        principal=PRINCIPAL,
        token=make_token(claims={"patient": "test-0001"}),
        patient=patient(),
        purpose_of_use="TREAT",
    )
    assert decision.allowed is True
    assert decision.verified_identifiers == ()


async def test_mismatching_claim_denied() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_open(
        principal=PRINCIPAL,
        token=make_token(claims={"patient": "test-9999"}),
        patient=patient(),
        purpose_of_use="TREAT",
    )
    assert decision.allowed is False
    assert decision.reason == "token_patient_mismatch"


async def test_no_claim_denied() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_open(
        principal=PRINCIPAL,
        token=make_token(claims={}),
        patient=patient(),
        purpose_of_use="TREAT",
    )
    assert decision.allowed is False
    assert decision.reason == "no_patient_launch_context"


async def test_none_token_on_open_denied() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_open(
        principal=PRINCIPAL,
        token=None,
        patient=patient(),
        purpose_of_use="TREAT",
    )
    assert decision.allowed is False
    assert decision.reason == "no_patient_launch_context"


async def test_purpose_not_allowed_denied() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_open(
        principal=PRINCIPAL,
        token=make_token(claims={"patient": "test-0001"}),
        patient=patient(),
        purpose_of_use="XYZ",
    )
    assert decision.allowed is False
    assert decision.reason == "purpose_not_allowed"


async def test_purpose_checked_before_claim() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_open(
        principal=PRINCIPAL,
        token=make_token(claims={"patient": "test-9999"}),
        patient=patient(),
        purpose_of_use="XYZ",
    )
    assert decision.reason == "purpose_not_allowed"


async def test_verified_identifiers_always_empty() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_open(
        principal=PRINCIPAL,
        token=make_token(claims={"patient": "test-0001"}),
        patient=PatientRef(
            reference="Patient/test-0001",
            identifiers=(Identifier(system="http://mrn", value="test-value"),),
        ),
        purpose_of_use="TREAT",
    )
    assert decision.allowed is True
    assert decision.verified_identifiers == ()


# --- authorize_population ---------------------------------------------------


async def test_population_none_token_denied() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_population(
        principal=PRINCIPAL, token=None, tool_name="search"
    )
    assert decision.allowed is False
    assert decision.reason == "unauthenticated"


async def test_population_single_patient_token_denied() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_population(
        principal=PRINCIPAL,
        token=make_token(claims={"patient": "test-0001"}),
        tool_name="search",
    )
    assert decision.allowed is False
    assert decision.reason == "single_patient_token"


async def test_population_user_scope_allowed() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_population(
        principal=PRINCIPAL,
        token=make_token(claims={}, scopes=["user/Patient.read"]),
        tool_name="search",
    )
    assert decision.allowed is True
    assert decision.reason == "population_scope"


async def test_population_system_scope_allowed() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_population(
        principal=PRINCIPAL,
        token=make_token(claims={}, scopes=["system/Patient.read"]),
        tool_name="search",
    )
    assert decision.allowed is True
    assert decision.reason == "population_scope"


async def test_population_patient_scope_only_denied() -> None:
    policy = SmartLaunchPolicy()
    decision = await policy.authorize_population(
        principal=PRINCIPAL,
        token=make_token(claims={}, scopes=["patient/Patient.read"]),
        tool_name="search",
    )
    assert decision.allowed is False
    assert decision.reason == "no_population_scope"
