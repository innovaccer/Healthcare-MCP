"""D4b: JSON Schema drift test and live validation of HMCP wire shapes.

Two checks:

* **Drift** -- re-running ``scripts/generate_schema.py`` in memory produces the
  committed ``schema/hmcp-0.1.schema.json`` byte-for-byte. If a model changes
  without regenerating, this fails.
* **Live** -- an in-memory ``HMCPServer`` (patient context + a reference
  guardrail) is driven with the SDK client, and the *actual* open/get/close
  results, a tool's advertised ``_meta``, a guardrail report from a real call,
  the request ``_meta`` objects and both extensions' ``server/discover``
  settings are validated with ``jsonschema``. This is what makes the
  hand-written schema fragments trustworthy: a wrong fragment fails here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import mcp.types as types
from jsonschema import Draft202012Validator
from mcp.client import Client
from pydantic import TypeAdapter

from hmcp.server.guardrails import GuardrailsConfig
from hmcp.server.hmcp_server import HMCPServer
from hmcp.server.patient_context import PatientContextConfig
from hmcp.server.patient_context.extension import PatientContextOpenParams
from hmcp.server.patient_context.policy import AccessDecision
from hmcp.shared.constants import (
    GUARDRAILS_EXTENSION,
    META_CONTEXT_ID,
    META_REPORT,
    META_REQUIRE,
    METHOD_PATIENT_CONTEXT_CLOSE,
    METHOD_PATIENT_CONTEXT_GET,
    METHOD_PATIENT_CONTEXT_OPEN,
    PATIENT_CONTEXT_EXTENSION,
)
from hmcp.shared.guardrails import (
    REDACTION_PLACEHOLDER_WRITEBACK_ID,
    redaction_placeholder_writeback,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = _REPO_ROOT / "schema" / "hmcp-0.1.schema.json"
_GENERATOR_PATH = _REPO_ROOT / "scripts" / "generate_schema.py"

_DICT = TypeAdapter(dict)


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hmcp_generate_schema", _GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Schema:
    """Validates instances against ``#/$defs/<name>`` of the committed schema."""

    def __init__(self) -> None:
        self._document = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._defs = self._document["$defs"]

    def validate(self, name: str, instance: Any) -> None:
        validator = Draft202012Validator(
            {"$ref": f"#/$defs/{name}", "$defs": self._defs}
        )
        validator.validate(instance)


class _AllowAllPolicy:
    async def authorize_open(self, *, principal, token, patient, purpose_of_use):
        return AccessDecision(allowed=True, reason="ok")

    async def authorize_population(self, *, principal, token, tool_name):
        return AccessDecision(allowed=True, reason="ok")


def test_committed_schema_matches_generator() -> None:
    module = _load_generator()
    generated = module.render(module.build_schema())
    committed = SCHEMA_PATH.read_text(encoding="utf-8")
    assert generated == committed, (
        "schema/hmcp-0.1.schema.json is stale; run "
        "`python scripts/generate_schema.py`"
    )


async def test_live_wire_shapes_validate_against_schema() -> None:
    server = HMCPServer(
        "schema-test",
        version="0.1.0",
        patient_context=PatientContextConfig(
            policy=_AllowAllPolicy(), allow_unauthenticated=True
        ),
        guardrails=GuardrailsConfig(
            definitions=[redaction_placeholder_writeback()],
            mandatory=[REDACTION_PLACEHOLDER_WRITEBACK_ID],
        ),
    )

    @server.tool(patient_scope="patient")
    async def read() -> str:
        return "hello"

    schema = _Schema()

    async with Client(server) as client:
        # server/discover settings, as the client saw them.
        extensions = client.server_capabilities.extensions or {}
        schema.validate("PatientContextSettings", extensions[PATIENT_CONTEXT_EXTENSION])
        schema.validate("GuardrailsSettings", extensions[GUARDRAILS_EXTENSION])

        # A tool's advertised _meta carries the patient scope.
        tools = (await client.list_tools()).tools
        tool = next(t for t in tools if t.name == "read")
        schema.validate("PatientContextToolMeta", tool.meta)

        # hmcp/patientContext/open
        open_request = types.Request[PatientContextOpenParams, str](
            method=METHOD_PATIENT_CONTEXT_OPEN,
            params=PatientContextOpenParams(
                patient={"reference": "Patient/test-0001"}, purpose_of_use="TREAT"
            ),
        )
        opened = await client.session.send_request(open_request, _DICT)
        schema.validate("PatientContextOpenResult", opened)
        context_id = opened["contextId"]

        # request _meta fragments for both extensions.
        schema.validate("PatientContextRequestMeta", {META_CONTEXT_ID: context_id})
        schema.validate(
            "GuardrailsRequestMeta",
            {META_REQUIRE: [REDACTION_PLACEHOLDER_WRITEBACK_ID]},
        )

        # hmcp/patientContext/get
        get_request = types.Request[Any, str](
            method=METHOD_PATIENT_CONTEXT_GET, params={"contextId": context_id}
        )
        got = await client.session.send_request(get_request, _DICT)
        schema.validate("PatientContextGetResult", got)

        # A real call: the mandatory guardrail runs and reports.
        result = await client.call_tool(
            "read", {}, meta={PATIENT_CONTEXT_EXTENSION: {META_CONTEXT_ID: context_id}}
        )
        result_meta = result.meta or {}
        report = result_meta[GUARDRAILS_EXTENSION]
        schema.validate("GuardrailsResultMeta", report)
        assert report[META_REPORT], "expected a report entry from the live call"
        schema.validate("GuardrailReportEntry", report[META_REPORT][0])

        # hmcp/patientContext/close
        close_request = types.Request[Any, str](
            method=METHOD_PATIENT_CONTEXT_CLOSE, params={"contextId": context_id}
        )
        closed = await client.session.send_request(close_request, _DICT)
        schema.validate("PatientContextCloseResult", closed)
