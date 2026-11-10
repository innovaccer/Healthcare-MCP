"""Shared protocol constants for HMCP 0.1.0.

Values here are part of the wire protocol documented in docs/specification/
and schema/hmcp-0.1.schema.json. They are imported by the server, client,
patient-context and guardrails extensions.
"""

PATIENT_CONTEXT_EXTENSION = "com.innovaccer/hmcp-patient-context"
GUARDRAILS_EXTENSION = "com.innovaccer/hmcp-guardrails"
PATIENT_CONTEXT_REVISION = "1"
GUARDRAILS_REVISION = "1"

METHOD_PATIENT_CONTEXT_OPEN = "hmcp/patientContext/open"
METHOD_PATIENT_CONTEXT_GET = "hmcp/patientContext/get"
METHOD_PATIENT_CONTEXT_CLOSE = "hmcp/patientContext/close"

# keys inside _meta[PATIENT_CONTEXT_EXTENSION] / _meta[GUARDRAILS_EXTENSION]
META_CONTEXT_ID = "contextId"  # request: {"contextId": "<id>"}
META_SCOPE = "scope"  # tool _meta: {"scope": "patient"|"population"|"none"}
META_REQUIRE = "require"  # request: {"require": ["<id>", "<id>@<major>"]}
META_REPORT = "report"  # result:  {"report": [GuardrailReportEntry, ...]}

GUARDRAIL_RESOURCE_PREFIX = "hmcp://guardrails/"
CONTEXT_ID_PREFIX = "pc_"
