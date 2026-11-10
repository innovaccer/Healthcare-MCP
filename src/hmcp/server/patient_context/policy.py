"""Access policy for HMCP patient contexts.

The policy answers two questions before any patient-scoped work happens:

* ``authorize_open`` — may this principal open a context for this patient?
* ``authorize_population`` — may this principal call a population-scoped tool?

Every decision carries a fixed machine ``reason`` code. Reasons are never PHI:
they name a rule, not a patient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mcp.server.auth.provider import AccessToken

from hmcp.shared.types import Identifier, PatientRef, Principal

# Allow reason for authorize_open. The protocol fixes the wire error codes, not
# these reason strings; the allow reason is free, so we use a short, stable one.
_OPEN_ALLOWED = "ok"


@dataclass(frozen=True)
class AccessDecision:
    """The answer to one access-policy question.

    ``allowed`` is the decision and ``reason`` is a short machine code naming
    the rule (never PHI) that is recorded in the audit trail.
    ``verified_identifiers`` carries identifiers the policy independently
    confirmed for the patient (for example from an EMPI), stored on the opened
    context; it is empty when the policy verifies nothing.
    """

    allowed: bool
    reason: str
    verified_identifiers: tuple[Identifier, ...] = ()


class AccessPolicy(Protocol):
    """Decides whether a principal may open a context or call a population tool."""

    async def authorize_open(
        self,
        *,
        principal: Principal,
        token: AccessToken | None,
        patient: PatientRef,
        purpose_of_use: str,
    ) -> AccessDecision:
        """Decide whether ``principal`` may open a context for ``patient``.

        Called by ``hmcp/patientContext/open`` before any context is created.
        ``token`` is the caller's access token, or ``None`` when the caller is
        the anonymous principal admitted by ``allow_unauthenticated``. Return an
        :class:`AccessDecision`; ``allowed=False`` denies the request with
        ``PATIENT_NOT_AUTHORIZED`` (-31005). Raising makes the request fail
        closed with ``PATIENT_CONTEXT_UNAVAILABLE`` (-31006).
        """
        ...

    async def authorize_population(
        self,
        *,
        principal: Principal,
        token: AccessToken | None,
        tool_name: str,
    ) -> AccessDecision:
        """Decide whether ``principal`` may call the population tool ``tool_name``.

        Called before the tool runs. ``token`` is the caller's access token, or
        ``None`` for the anonymous principal. Return an :class:`AccessDecision`;
        ``allowed=False`` denies the call with ``PATIENT_NOT_AUTHORIZED``
        (-31005). Raising makes the call fail closed with
        ``PATIENT_CONTEXT_UNAVAILABLE`` (-31006).
        """
        ...


class SmartLaunchPolicy:
    """Default policy for SMART-on-FHIR-style launches.

    Opening a patient context requires a token whose ``patient`` claim matches
    the requested patient. Population tools require a user- or system-scoped
    token. It never verifies identifiers, so ``verified_identifiers`` is always
    empty; a deployment that can confirm identifiers should supply its own
    policy (for example one that calls an EMPI access API).
    """

    def __init__(
        self,
        *,
        allowed_purposes: frozenset[str] = frozenset({"TREAT", "HPAYMT", "HOPERAT"}),
    ) -> None:
        self._allowed_purposes = allowed_purposes

    async def authorize_open(
        self,
        *,
        principal: Principal,
        token: AccessToken | None,
        patient: PatientRef,
        purpose_of_use: str,
    ) -> AccessDecision:
        if purpose_of_use not in self._allowed_purposes:
            return AccessDecision(allowed=False, reason="purpose_not_allowed")
        claim = (token.claims or {}).get("patient") if token else None
        if isinstance(claim, str) and claim:
            if patient.reference == f"Patient/{claim}":
                return AccessDecision(allowed=True, reason=_OPEN_ALLOWED)
            return AccessDecision(allowed=False, reason="token_patient_mismatch")
        return AccessDecision(allowed=False, reason="no_patient_launch_context")

    async def authorize_population(
        self,
        *,
        principal: Principal,
        token: AccessToken | None,
        tool_name: str,
    ) -> AccessDecision:
        if token is None:
            return AccessDecision(allowed=False, reason="unauthenticated")
        claim = (token.claims or {}).get("patient")
        if isinstance(claim, str):
            return AccessDecision(allowed=False, reason="single_patient_token")
        scopes = token.scopes or ()
        if any(scope.startswith(("user/", "system/")) for scope in scopes):
            return AccessDecision(allowed=True, reason="population_scope")
        return AccessDecision(allowed=False, reason="no_population_scope")
