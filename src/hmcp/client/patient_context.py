"""Client-side patient-context wire models and helpers.

This module holds the public :class:`PatientContextHandle` value returned by
:meth:`hmcp.client.client_connector.HMCPClientConnector.open_patient_context`,
the :class:`GuardrailReportMissing` error raised when a required guardrail did
not report, and the request/result models for the patient-context extension
methods. The method wire shapes are defined in
``docs/specification/patient-context.md`` and ``schema/hmcp-0.1.schema.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import mcp.types as types
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _WireModel(BaseModel):
    """Base for wire models: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="allow",
    )


class Identifier(_WireModel):
    """A patient identifier: a system/value pair."""

    system: str
    value: str


class PatientRef(_WireModel):
    """A FHIR-relative patient reference plus verified identifiers."""

    reference: str
    identifiers: list[Identifier] = []


class PatientContextOpenParams(types.RequestParams):
    """Params for ``hmcp/patientContext/open``."""

    patient: PatientRef
    encounter: str | None = None
    purpose_of_use: str = "TREAT"
    correlation_id: str | None = None


class PatientContextGetParams(types.RequestParams):
    """Params for ``hmcp/patientContext/get``."""

    context_id: str


class PatientContextCloseParams(types.RequestParams):
    """Params for ``hmcp/patientContext/close``."""

    context_id: str


class PatientContextOpenResult(_WireModel):
    """Result of ``hmcp/patientContext/open``."""

    context_id: str
    status: str
    expires_at: datetime


class PatientContextGetResult(_WireModel):
    """Result of ``hmcp/patientContext/get``.

    Attributes:
        context_id: The opaque context id the server minted when the context
            was opened.
        status: The context status, e.g. ``"active"`` or ``"closed"``.
        expires_at: When the context expires. ``get_patient_context`` returns
            the server's value unchanged and does not require a UTC offset, so
            this may be naive if the server sends one.
        patient: The bound patient reference and its verified identifiers,
            when the server reports them.
        encounter: The FHIR-relative encounter reference, when set.
        purpose_of_use: The HL7 v3 PurposeOfUse code the context was opened
            with, when reported.
    """

    context_id: str
    status: str
    expires_at: datetime
    patient: PatientRef | None = None
    encounter: str | None = None
    purpose_of_use: str | None = None


class PatientContextCloseResult(_WireModel):
    """Result of ``hmcp/patientContext/close``."""

    context_id: str
    status: str


@dataclass(frozen=True)
class PatientContextHandle:
    """A live patient context returned by ``open_patient_context``."""

    context_id: str
    status: str
    expires_at: datetime


class GuardrailReportMissing(Exception):
    """A required guardrail produced no report entry for the tool call.

    The message names the guardrail id only; it never carries tool arguments,
    result content, or any other identifier.
    """

    def __init__(self, guardrail_id: str) -> None:
        super().__init__(f"No guardrail report for {guardrail_id}")
        self.guardrail_id = guardrail_id


def context_id_of(handle: "PatientContextHandle | str") -> str:
    """Return the context id from a handle or a bare id string."""
    if isinstance(handle, PatientContextHandle):
        return handle.context_id
    return handle


__all__ = [
    "GuardrailReportMissing",
    "Identifier",
    "PatientContextCloseParams",
    "PatientContextCloseResult",
    "PatientContextGetParams",
    "PatientContextGetResult",
    "PatientContextHandle",
    "PatientContextOpenParams",
    "PatientContextOpenResult",
    "PatientRef",
]
