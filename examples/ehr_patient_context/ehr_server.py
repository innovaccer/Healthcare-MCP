"""A tiny EHR server showing HMCP patient context and guardrails.

HMCP 0.1.0 removes the old bidirectional *sampling* feature, so a tool can no
longer ask the host to run an LLM. This example is LLM-free: it shows the two
HMCP extensions that replace the sampling demos.

* **Patient context** — the host (the client application, never the model)
  opens an opaque, server-minted handle for one patient. The handle is bound
  to the caller's token, and every patient-scoped tool call must carry it.
  Patient-scoped tools call :func:`require_patient_context` to read it.
* **Guardrails** — declarative definitions enforced server-side. They run on
  tool arguments (input stage) and tool results (output stage), and each call
  gets a report attached to the result's ``_meta`` so a client can see which
  protections ran.

Run the server with::

    python examples/ehr_patient_context/ehr_server.py

It listens on http://127.0.0.1:8050/mcp. Start ``ehr_client.py`` in a second
terminal to drive it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.auth.settings import AuthSettings

from hmcp.server.guardrails import GuardrailsConfig
from hmcp.server.hmcp_server import HMCPServer
from hmcp.server.patient_context import PatientContextConfig
from hmcp.shared.guardrails import (
    PATIENT_CONTEXT_CONSISTENCY_ID,
    REDACTION_PLACEHOLDER_WRITEBACK_ID,
    PHI_IDENTIFIER_TYPES,
    ActionSpec,
    GuardrailDefinition,
    PatternEvaluatorSpec,
    patient_context_consistency,
    redaction_placeholder_writeback,
)
from hmcp.shared.types import require_patient_context

# demo_auth lives next to this file; running the script puts that directory on
# sys.path, and the smoke test adds it too.
from demo_auth import DEMO_ISSUER, SERVER_URL, DemoTokenVerifier

# The id of the example-local guardrail defined below.
DEMO_PHI_REDACTION_ID = "com.example.ehr-demo/phi-patterns"

# ---------------------------------------------------------------------------
# Synthetic data (never real patients)
# ---------------------------------------------------------------------------

_PATIENTS: dict[str, dict[str, str]] = {
    "test-0001": {
        "name": "Testpatient Alpha",
        "dob": "1980-01-01",
        "ssn": "900-12-3456",
        "mrn": "TEST-0001",
    },
    "test-0002": {
        "name": "Testpatient Beta",
        "dob": "1975-05-05",
        "ssn": "900-12-6543",
        "mrn": "TEST-0002",
    },
}

_MEDICATIONS: dict[str, list[str]] = {
    "test-0001": ["Lisinopril 10 mg", "Atorvastatin 20 mg"],
    "test-0002": ["Metformin 500 mg"],
}


# ---------------------------------------------------------------------------
# Guardrail definitions
# ---------------------------------------------------------------------------


def demo_phi_redaction() -> GuardrailDefinition:
    """An example-local, pattern-only PHI redaction guardrail.

    The reference ``phi_redaction()`` definition also calls an LLM to find
    names and locations. This demo must stay LLM-free, so it uses only the
    deterministic :class:`PatternEvaluatorSpec` over HMCP's built-in
    identifier patterns (:data:`PHI_IDENTIFIER_TYPES`: SSN, dates, MRN, phone,
    email, ...). A finding is *redacted*: the matched text is replaced with the
    placeholder ``[REDACTED:<category>]``.
    """

    return GuardrailDefinition(
        id=DEMO_PHI_REDACTION_ID,
        version="1.0.0",
        title="Demo PHI redaction (pattern only)",
        description="Redacts identifier patterns from tool output; no LLM.",
        stages=("output",),
        evaluator=PatternEvaluatorSpec(
            kind="pattern", identifier_types=PHI_IDENTIFIER_TYPES
        ),
        action=ActionSpec(on_finding="redact"),
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def build_server() -> HMCPServer:
    """Build the demo server with auth, patient context and guardrails."""

    server = HMCPServer(
        "EHR demo",
        token_verifier=DemoTokenVerifier(),
        auth=AuthSettings(
            issuer_url=DEMO_ISSUER,
            resource_server_url=SERVER_URL,
            validate_token_resource=True,
        ),
        patient_context=PatientContextConfig(),
        guardrails=GuardrailsConfig(
            definitions=[
                patient_context_consistency(),
                redaction_placeholder_writeback(),
                demo_phi_redaction(),
            ],
            # Mandatory guardrails run on every call they apply to, whether or
            # not the client asked for them.
            mandatory=[
                PATIENT_CONTEXT_CONSISTENCY_ID,
                REDACTION_PLACEHOLDER_WRITEBACK_ID,
                DEMO_PHI_REDACTION_ID,
            ],
        ),
    )

    # Notes written through add_note, kept in memory for the demo.
    notes: list[str] = []

    # -- patient-scoped tools ------------------------------------------------
    # Each declares patient_scope="patient": the patient-context extension
    # advertises that scope in the tool's _meta and refuses the call unless a
    # valid context handle for this token is attached.

    @server.tool(patient_scope="patient")
    def get_medications() -> str:
        # require_patient_context() returns the context the extension bound for
        # this call; it raises if somehow none is bound.
        context = require_patient_context()
        patient = context.patient.reference.split("/", 1)[1]
        return "Medications for " + patient + ": " + ", ".join(_MEDICATIONS[patient])

    @server.tool(patient_scope="patient")
    def get_demographics() -> dict[str, str]:
        # Returning a dict (not a free-text string) matters: the pattern
        # guardrail redacts both structured identifiers (SSN, date of birth,
        # MRN, by their value shape) and *keyed* fields whose key names are PHI
        # ("name", "dob", ...). In free text it could not find a name at all.
        context = require_patient_context()
        patient = context.patient.reference.split("/", 1)[1]
        return dict(_PATIENTS[patient])

    @server.tool(patient_scope="patient")
    def add_note(text: str) -> str:
        # Writing back text that already contains a [REDACTED:...] placeholder
        # is refused by the write-back guardrail before this body runs.
        require_patient_context()
        notes.append(text)
        return "saved"

    @server.tool(patient_scope="patient")
    def lookup_with_bug() -> str:
        # A deliberate bug: this tool returns the OTHER patient's record. The
        # patient-context-consistency guardrail sees an identifier that does
        # not match the active context and blocks the result.
        require_patient_context()
        other = _PATIENTS["test-0002"]
        return f"MRN {other['mrn']}: {other['name']}"

    # -- population-scoped tool ---------------------------------------------

    @server.tool(patient_scope="population")
    def population_count() -> str:
        # A population tool needs a user- or system-scoped token. The demo
        # token is a single-patient token, so SmartLaunchPolicy denies it.
        return f"{len(_PATIENTS)} patients"

    # -- unscoped tool -------------------------------------------------------

    @server.tool(patient_scope="none")
    def server_time() -> str:
        return datetime.now(timezone.utc).isoformat()

    return server


if __name__ == "__main__":
    build_server().run(transport="streamable-http", host="127.0.0.1", port=8050)
