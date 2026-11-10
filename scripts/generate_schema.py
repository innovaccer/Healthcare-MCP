#!/usr/bin/env python3
"""Generate ``schema/hmcp-0.1.schema.json`` from the HMCP 0.1.0 wire models.

Run from the repository root::

    python scripts/generate_schema.py

The document is a JSON Schema (draft 2020-12) whose ``$defs`` describe every
HMCP-specific wire shape: the guardrail definition and report entry (generated
from the Pydantic models with ``model_json_schema(by_alias=True)``), the patient
reference and identifier, the patient-context extension method params (generated
from the server ``RequestParams`` models) and results, the request ``_meta``
objects of both extensions, the result ``_meta`` report object, the tool ``_meta``
scope object and both extensions' ``server/discover`` settings objects.

The Pydantic-derived shapes are authoritative: they come from the models in
``src/``. The dict-shaped fragments (results, ``_meta`` objects, settings) are
hand-written here because they have no Pydantic model; ``tests/hmcp/test_schema.py``
validates real server output against them, so a fragment that drifts from the
wire shape fails the test.

Output is deterministic: keys are sorted and the file ends with one newline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The package is not necessarily installed; make ``src`` importable.
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hmcp.server.patient_context.extension import (  # noqa: E402
    PatientContextIdParams,
    PatientContextOpenParams,
)
from hmcp.shared.constants import (  # noqa: E402
    GUARDRAILS_EXTENSION,
    META_CONTEXT_ID,
    META_REPORT,
    META_REQUIRE,
    META_SCOPE,
    PATIENT_CONTEXT_EXTENSION,
)
from hmcp.shared.guardrails.definition import GuardrailDefinition  # noqa: E402
from hmcp.shared.types import (  # noqa: E402
    GuardrailReportEntry,
    Identifier,
    PatientRef,
)

SCHEMA_PATH = _REPO_ROOT / "schema" / "hmcp-0.1.schema.json"
SCHEMA_ID = "https://innovaccer.com/hmcp/schemas/hmcp-0.1.schema.json"

_CONTEXT_STATUS = ["active", "suspended", "revoked", "closed", "expired"]
_TOOL_SCOPES = ["patient", "population", "none"]
_STAGES = ["input", "output"]

# The MCP result envelope keys every method result carries on the wire; allowed
# (not required) so a bare domain result also validates.
_ENVELOPE_PROPERTIES: dict[str, Any] = {
    "resultType": {"type": "string"},
    "_meta": {"type": "object"},
}


def _result_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """A method-result object: domain fields plus the MCP result envelope."""
    return {
        "type": "object",
        "properties": {**properties, **_ENVELOPE_PROPERTIES},
        "required": required,
        "additionalProperties": False,
    }


def _add_model(defs: dict[str, Any], name: str, model: type) -> None:
    """Merge ``model``'s JSON Schema into ``defs`` under ``name``.

    Pydantic emits nested models under ``$defs`` and references them as
    ``#/$defs/<Name>``, so the nested definitions are hoisted to the top level,
    where those references resolve. A name collision with different content is
    an error (the two models genuinely disagree and cannot share one definition).
    """
    schema = model.model_json_schema(by_alias=True)
    nested = schema.pop("$defs", {})
    for nested_name, nested_schema in nested.items():
        existing = defs.get(nested_name)
        if existing is not None and existing != nested_schema:
            raise ValueError(f"schema $defs collision for {nested_name!r}")
        defs[nested_name] = nested_schema
    schema["title"] = name
    existing = defs.get(name)
    if existing is not None and existing != schema:
        raise ValueError(f"schema $defs collision for {name!r}")
    defs[name] = schema


def _handwritten_defs() -> dict[str, Any]:
    """The dict-shaped wire shapes that have no Pydantic model."""
    return {
        # -- patient-context method results --------------------------------
        "PatientContextOpenResult": _result_schema(
            {
                "contextId": {"type": "string", "minLength": 1},
                "status": {"enum": _CONTEXT_STATUS},
                "expiresAt": {"type": "string", "format": "date-time"},
            },
            ["contextId", "status", "expiresAt"],
        ),
        "PatientContextGetResult": _result_schema(
            {
                "contextId": {"type": "string", "minLength": 1},
                "status": {"enum": _CONTEXT_STATUS},
                "expiresAt": {"type": "string", "format": "date-time"},
                "patient": {"$ref": "#/$defs/PatientRef"},
                "encounter": {"type": ["string", "null"]},
                "purposeOfUse": {"type": "string"},
            },
            [
                "contextId",
                "status",
                "expiresAt",
                "patient",
                "encounter",
                "purposeOfUse",
            ],
        ),
        "PatientContextCloseResult": _result_schema(
            {
                "contextId": {"type": "string", "minLength": 1},
                "status": {"enum": _CONTEXT_STATUS},
            },
            ["contextId", "status"],
        ),
        # -- request _meta objects -----------------------------------------
        "PatientContextRequestMeta": {
            "type": "object",
            "properties": {"contextId": {"type": "string", "minLength": 1}},
            "required": [META_CONTEXT_ID],
            "additionalProperties": False,
        },
        "GuardrailsRequestMeta": {
            "type": "object",
            "properties": {
                META_REQUIRE: {"type": "array", "items": {"type": "string"}}
            },
            "required": [META_REQUIRE],
            "additionalProperties": False,
        },
        # -- result _meta report object ------------------------------------
        "GuardrailsResultMeta": {
            "type": "object",
            "properties": {
                META_REPORT: {
                    "type": "array",
                    "items": {"$ref": "#/$defs/GuardrailReportEntry"},
                }
            },
            "required": [META_REPORT],
            "additionalProperties": False,
        },
        # -- tool _meta scope object ---------------------------------------
        "PatientContextToolMeta": {
            "type": "object",
            "properties": {
                PATIENT_CONTEXT_EXTENSION: {
                    "type": "object",
                    "properties": {META_SCOPE: {"enum": _TOOL_SCOPES}},
                    "required": [META_SCOPE],
                    "additionalProperties": False,
                }
            },
            "required": [PATIENT_CONTEXT_EXTENSION],
            "additionalProperties": True,
        },
        # -- server/discover settings --------------------------------------
        "PatientContextSettings": {
            "type": "object",
            "properties": {
                "revision": {"type": "string"},
                "contextTtlSeconds": {
                    "type": "integer",
                    "minimum": 60,
                    "maximum": 86400,
                },
                "defaultToolScope": {"enum": _TOOL_SCOPES},
            },
            "required": ["revision", "contextTtlSeconds", "defaultToolScope"],
            "additionalProperties": False,
        },
        "GuardrailsSettings": {
            "type": "object",
            "properties": {
                "revision": {"type": "string"},
                "available": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "version": {"type": "string"},
                            "digest": {"type": "string"},
                            "stages": {
                                "type": "array",
                                "items": {"enum": _STAGES},
                                "minItems": 1,
                            },
                            "definition": {"type": "string"},
                            "enforcedBy": {"const": "server"},
                            "mandatory": {"type": "boolean"},
                        },
                        "required": [
                            "id",
                            "version",
                            "digest",
                            "stages",
                            "definition",
                            "enforcedBy",
                            "mandatory",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["revision", "available"],
            "additionalProperties": False,
        },
    }


def build_schema() -> dict[str, Any]:
    """Build the schema document as a plain (JSON-serialisable) dict."""
    defs: dict[str, Any] = {}
    _add_model(defs, "GuardrailDefinition", GuardrailDefinition)
    _add_model(defs, "GuardrailReportEntry", GuardrailReportEntry)
    _add_model(defs, "PatientRef", PatientRef)
    _add_model(defs, "Identifier", Identifier)
    _add_model(defs, "PatientContextOpenParams", PatientContextOpenParams)
    # get and close share the same RequestParams model on the server.
    _add_model(defs, "PatientContextGetParams", PatientContextIdParams)
    _add_model(defs, "PatientContextCloseParams", PatientContextIdParams)
    for name, fragment in _handwritten_defs().items():
        if name in defs:
            raise ValueError(f"hand-written $defs collision for {name!r}")
        defs[name] = fragment
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "HMCP 0.1.0 wire shapes",
        "description": (
            "JSON Schema (draft 2020-12) for the HMCP 0.1.0 wire shapes. "
            "Generated by scripts/generate_schema.py from the models in src/; "
            "do not edit by hand."
        ),
        "$defs": defs,
    }


def render(schema: dict[str, Any]) -> str:
    """Serialise a schema document deterministically (sorted keys, one newline)."""
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(render(build_schema()), encoding="utf-8")
    print(f"wrote {SCHEMA_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
