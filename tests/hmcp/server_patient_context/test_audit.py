"""Tests for audit events and the logging sink."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hmcp.server.patient_context.audit import AuditEvent, LoggingAuditSink


def make_event() -> AuditEvent:
    return AuditEvent(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        action="patient_context.open",
        outcome="success",
        patient_reference="Patient/test-0001",
        context_id="pc_1",
    )


@pytest.mark.asyncio
async def test_logging_sink_emits_one_json_info_record(caplog) -> None:
    sink = LoggingAuditSink()

    with caplog.at_level(logging.INFO, logger="hmcp.audit"):
        await sink.emit(make_event())

    records = [
        r
        for r in caplog.records
        if r.name == "hmcp.audit" and r.levelno == logging.INFO
    ]
    assert len(records) == 1

    payload = json.loads(records[0].getMessage())
    assert payload["patientReference"] == "Patient/test-0001"
    assert payload["contextId"] == "pc_1"


def test_audit_event_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            action="tool.call",
            outcome="attempt",
            arguments={"patient": "secret"},
        )


def test_audit_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(
            timestamp=datetime(2026, 1, 1, 12, 0),
            action="tool.call",
            outcome="attempt",
        )
