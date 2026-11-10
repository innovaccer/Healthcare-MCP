"""Audit events and sinks for HMCP patient contexts.

An :class:`AuditEvent` deliberately has no field for tool arguments, result
text, identifier values, names or tokens. The model forbids extra fields, so a
new field cannot be smuggled in without a deliberate change here.

Type aliases
------------
``AuditAction`` names the audited operation: ``patient_context.open``,
``patient_context.get``, ``patient_context.close``, ``patient_context.revoke``,
``patient_context.suspend``, ``patient_context.resume`` or ``tool.call``.

``AuditOutcome`` is the result recorded for an action: ``attempt`` (the
fail-closed point was reached), ``success``, ``denied`` (a policy or validity
decision refused the request) or ``error`` (an internal failure).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, Protocol

from pydantic import field_validator

from hmcp.shared.types import Principal, _WireModel

AuditAction = Literal[
    "patient_context.open",
    "patient_context.get",
    "patient_context.close",
    "patient_context.revoke",
    "patient_context.suspend",
    "patient_context.resume",
    "tool.call",
]

AuditOutcome = Literal["attempt", "success", "denied", "error"]


class AuditEvent(_WireModel):
    """One audited patient-context decision or tool call.

    Records what was attempted and its outcome, never the request's payload:
    there is no field for tool arguments, result text, identifier values, names
    or tokens. ``patient_reference`` and ``context_id`` are present so the
    access trail can be reconstructed.
    """

    timestamp: datetime  # tz-aware UTC
    action: AuditAction
    outcome: AuditOutcome
    principal: Principal | None = None
    context_id: str | None = None
    patient_reference: str | None = None  # "Patient/<id>" — the access trail needs it
    purpose_of_use: str | None = None
    tool: str | None = None
    error_code: int | None = None
    reason: str | None = None  # AccessDecision.reason or a fixed code
    guardrails: tuple[tuple[str, str], ...] = ()  # (guardrail id, verdict)
    correlation_id: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware")
        return value


class AuditSink(Protocol):
    """Destination for :class:`AuditEvent` records."""

    async def emit(self, event: AuditEvent) -> None:
        """Record ``event``.

        Called after every patient-context decision and tool call, including the
        ``attempt`` event emitted before a context is stored or a tool runs.
        Raising is fail-closed at the required pre-action emits (the ones that
        must not be skipped), where the extension maps it to
        ``PATIENT_CONTEXT_UNAVAILABLE`` (-31006); at the remaining best-effort
        emits a failure is logged and ignored.
        """
        ...


class LoggingAuditSink:
    """Default sink: one INFO line of JSON per event."""

    def __init__(self, logger_name: str = "hmcp.audit") -> None:
        self._logger = logging.getLogger(logger_name)

    async def emit(self, event: AuditEvent) -> None:
        self._logger.info(event.model_dump_json(by_alias=True))
