"""Shared wire models and patient-context accessors for HMCP 0.1.0.

All wire models use camelCase aliases, are frozen and forbid extra fields.
Serialize with ``model_dump(by_alias=True, mode="json")``.

Type aliases
------------
``ToolPatientScope`` is the patient scope a tool may declare: ``patient``,
``population`` or ``none``.

``GuardrailStage`` is the point in a call at which a guardrail runs: ``input``
or ``output``.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Iterator, Literal

from mcp.server.auth.provider import AccessToken, principal_components
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from hmcp.shared.errors import HMCPErrorCode, hmcp_error


class _WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        frozen=True,
        extra="forbid",
    )


class Identifier(_WireModel):
    """A patient identifier: ``system`` (its assigning authority) and ``value``."""

    system: str
    value: str


class PatientRef(_WireModel):
    """A reference to a patient: a FHIR relative reference plus optional identifiers."""

    # FHIR relative reference
    reference: str = Field(pattern=r"^Patient/[A-Za-z0-9\-\.]{1,64}$")
    identifiers: tuple[Identifier, ...] = ()


class ContextStatus(str, Enum):
    """Lifecycle state of a patient context.

    ``ACTIVE`` and ``SUSPENDED`` are the non-terminal states; ``REVOKED``,
    ``CLOSED`` and ``EXPIRED`` are terminal.
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    CLOSED = "closed"
    EXPIRED = "expired"


ToolPatientScope = Literal["patient", "population", "none"]


class Principal(_WireModel):
    """The caller a patient context is bound to.

    Derived from the access token (``client_id``, ``issuer``, ``subject`` and
    ``resource``), not from the raw token, so a token refreshed mid-context
    keeps the same principal. ``subject`` and ``issuer`` are ``None`` when the
    token carries neither, which weakens binding to the client alone.
    """

    client_id: str
    issuer: str | None = None
    subject: str | None = None
    resource: str | None = None

    @classmethod
    def from_access_token(cls, token: AccessToken) -> "Principal":
        client_id, issuer, subject = principal_components(token)
        return cls(
            client_id=client_id,
            issuer=issuer,
            subject=subject,
            resource=token.resource,
        )

    @classmethod
    def anonymous(cls) -> "Principal":
        # Only for allow_unauthenticated dev mode.
        return cls(client_id="anonymous")


class PatientContext(_WireModel):
    """A server-minted patient context bound to one principal.

    ``context_id`` is the opaque handle the host passes back in request
    ``_meta``; ``created_at`` and ``expires_at`` are timezone-aware.
    """

    context_id: str
    patient: PatientRef
    encounter: str | None = Field(
        default=None, pattern=r"^Encounter/[A-Za-z0-9\-\.]{1,64}$"
    )
    # HL7 v3 PurposeOfUse code, e.g. TREAT
    purpose_of_use: str = Field(pattern=r"^[A-Z]{3,12}$")
    principal: Principal
    status: ContextStatus
    created_at: datetime  # must be timezone-aware (validator rejects naive)
    expires_at: datetime  # must be timezone-aware
    correlation_id: str | None = None

    @field_validator("created_at", "expires_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware")
        return value

    def is_usable(self, now: datetime | None = None) -> bool:
        if self.status is not ContextStatus.ACTIVE:
            return False
        return (now or datetime.now(timezone.utc)) < self.expires_at


class GuardrailVerdict(str, Enum):
    """Outcome a guardrail reports for the text it inspected."""

    PASS = "pass"
    REDACTED = "redacted"
    PSEUDONYMIZED = "pseudonymized"
    REIDENTIFIED = "reidentified"
    ANNOTATED = "annotated"
    BLOCKED = "blocked"
    ERROR = "error"


GuardrailStage = Literal["input", "output"]


class GuardrailReportEntry(_WireModel):
    """One guardrail's result for one stage of one call.

    ``findings`` is a count only and never carries matched text.
    """

    id: str
    version: str
    digest: str
    stage: GuardrailStage
    verdict: GuardrailVerdict  # REQUIRED - no default
    evaluator: str  # e.g. "cascade(pattern,prompt)"
    findings: int = Field(ge=0)  # a COUNT only - never matched text
    duration_ms: int = Field(ge=0)  # wire: durationMs


# current patient context, set only by the patient-context extension
_current_patient_context: ContextVar[PatientContext | None] = ContextVar(
    "hmcp_current_patient_context", default=None
)


def current_patient_context() -> PatientContext | None:
    """Return the context bound to the current task, or ``None``."""
    return _current_patient_context.get()


def require_patient_context() -> PatientContext:
    """Return the bound context or raise ``PATIENT_CONTEXT_REQUIRED`` (-31001)."""
    ctx = _current_patient_context.get()
    if ctx is None:
        raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)
    return ctx


@contextmanager
def bind_patient_context(ctx: PatientContext) -> Iterator[None]:
    """Bind ``ctx`` for the duration of the ``with`` block in the current task."""
    token = _current_patient_context.set(ctx)
    try:
        yield
    finally:
        _current_patient_context.reset(token)
