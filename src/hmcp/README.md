# HMCP Python SDK

**_Python implementation of the Healthcare Model Context Protocol (HMCP)_**

This is the SDK for HMCP **0.1.0**, which targets base MCP spec **2026-07-28**. It builds
on the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and
implements the [HMCP specification](../../docs/specification/index.md).

> HMCP makes no accuracy, completeness, compliance or patient-safety guarantee, and the implementer is responsible for validation, compliance and outcomes. See the [Disclaimer](../../README.md#disclaimer).

## Package layout

`hmcp.__version__` reports the SDK version (`"0.1.0"`).

- **`hmcp.server`:** the server side. `HMCPServer` is re-exported lazily, so
  `from hmcp.server import HMCPServer` works.
  - `hmcp.server.hmcp_server.HMCPServer`: a thin HMCP layer over the MCP SDK's
    `MCPServer` that provides authentication defaults, patient context, guardrails and the
    transport rule.
  - `hmcp.server.patient_context`: the patient-context extension plus its store, access
    policy and audit sink (`PatientContextConfig`, `AccessPolicy`, `SmartLaunchPolicy`,
    `AuditSink`, `LoggingAuditSink`, `InMemoryContextStore`).
  - `hmcp.server.guardrails`: the guardrails extension and its config
    (`GuardrailsConfig`, `GuardrailsExtension`).
- **`hmcp.client`:** the client side.
  - `hmcp.client.client_connector.HMCPClientConnector`: a helper for streamable-http
    connections, optional client-credentials auth, tool calls with a patient context and
    required guardrails, resources and prompts.
  - `hmcp.client.patient_context`: client-side patient-context wire models,
    `PatientContextHandle` and `GuardrailReportMissing`.
- **`hmcp.shared.guardrails`:** declarative guardrails, including definitions, evaluators,
  the engine and the four reference definitions (`phi_redaction`, `phi_deidentification`,
  `patient_context_consistency`, `redaction_placeholder_writeback`).
- **`hmcp.shared.auth`:** the OAuth 2.1 client (OAuth 2.0 + PKCE) (`OAuthClient`),
  `AuthenticationError` and its `InvalidStateError` subclass.
- **`hmcp.shared.types` / `hmcp.shared.constants`:** shared types (`ToolPatientScope`)
  and the extension identifiers and `_meta` keys.

## Implementation status (0.1.0)

The status matches that of the [top-level README](../../README.md):

| Area | Status |
| --- | --- |
| Patient context extension (`com.innovaccer/hmcp-patient-context`) | Implemented |
| Guardrails extension (`com.innovaccer/hmcp-guardrails`) with four reference guardrails | Implemented |
| OAuth 2.1 client (OAuth 2.0 + PKCE) (`OAuthClient`); server-side auth enforcement is opt-in | Implemented |
| RFC 8707 audience-bound auth defaults | Implemented |
| Patient-context audit events via a pluggable `AuditSink` | Implemented |
| Transports: `streamable-http`, `stdio` | Implemented |
| Encryption at rest | Not implemented |
| Rate limiting | Not implemented |
| Signed guardrail reports | Not implemented |
| Cohort contexts | Not implemented |

HMCP 0.1.0 removed bidirectional sampling and the HTTP+SSE transport (both deprecated by
MCP 2026-07-28). See
[Migrating to 0.1](../../docs/specification/migration-0.1.md).

## Installation

Python 3.11 or newer is required.

```bash
pip install innovaccer-hmcp
```

The import name is `hmcp`. (An unrelated PyPI project named `hmcp` is not this SDK.)

To install from a source checkout instead:

```bash
pip install -e ".[dev]"
```

## Usage

Create a server with patient context and guardrails, register patient-scoped tools, and authenticate every request with a `TokenVerifier`:

```python
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from hmcp.server.guardrails import GuardrailsConfig
from hmcp.server import HMCPServer
from hmcp.server.patient_context import PatientContextConfig
from hmcp.shared.guardrails import (
    patient_context_consistency,
    redaction_placeholder_writeback,
)
from hmcp.shared.types import require_patient_context

SERVER_URL = "http://127.0.0.1:8050/mcp"
DEMO_TOKEN = "demo-token-test-0001"


class DemoTokenVerifier(TokenVerifier):
    """DEMO ONLY: accept one static token and reject everything else.

    A real deployment validates a signed token here. The token's ``patient``
    claim is the SMART launch context the default policy checks.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != DEMO_TOKEN:
            return None
        return AccessToken(
            token=token,
            client_id="demo-app",
            scopes=["patient/*.rs"],
            resource=SERVER_URL,
            subject="dr-demo",
            claims={"iss": "https://auth.example.com", "patient": "test-0001"},
        )


server = HMCPServer(
    name="patient-demo",
    version="0.1.0",
    token_verifier=DemoTokenVerifier(),
    auth=AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url=SERVER_URL,
        validate_token_resource=True,
    ),
    patient_context=PatientContextConfig(default_tool_scope="patient"),
    guardrails=GuardrailsConfig(
        definitions=[patient_context_consistency(), redaction_placeholder_writeback()],
    ),
)


@server.tool(patient_scope="patient")
def get_observations() -> dict:
    context = require_patient_context()
    return {"patient": context.patient.reference}


if __name__ == "__main__":
    server.run(transport="streamable-http", host="127.0.0.1", port=8050)
```

Connect with the client connector, sending the bearer token; it advertises both extensions by default:

```python
import asyncio

from hmcp.client.client_connector import HMCPClientConnector

DEMO_TOKEN = "demo-token-test-0001"


async def main() -> None:
    async with HMCPClientConnector(
        url="http://127.0.0.1:8050/mcp",
        headers={"Authorization": f"Bearer {DEMO_TOKEN}"},
    ) as connector:
        handle = await connector.open_patient_context(
            "Patient/test-0001", purpose_of_use="TREAT"
        )
        result = await connector.call_tool(
            "get_observations",
            patient_context=handle,
            require_guardrails=[
                "com.innovaccer.hmcp.guardrails/patient-context-consistency"
            ],
        )
        print(result)

        # get_patient_context returns a PatientContextGetResult whose
        # attributes are snake_case (not a camelCase dict).
        context = await connector.get_patient_context(handle)
        print(context.context_id, context.status, context.patient.reference)

        await connector.close_patient_context(handle)


if __name__ == "__main__":
    asyncio.run(main())
```

> **Important:** `SmartLaunchPolicy` (the default) refuses every `open` when no token carries a SMART `patient` claim, which fails closed. A production deployment supplies tokens from a SMART authorization server (or a custom `AccessPolicy`); the static token above is demo-only.

See the [top-level README](../../README.md#quick-start) for the full walk-through and
[examples/ehr_patient_context](../../examples/ehr_patient_context/README.md) for a runnable
example.

## Errors and authentication

- A request that resolves to no principal is rejected with `PATIENT_NOT_AUTHORIZED`
  (-31005) and the message `Request is not authenticated`. This applies to a server with
  neither `token_verifier` nor `auth_server_provider` configured and
  `PatientContextConfig(allow_unauthenticated=False)` (the default). `HMCPServer` logs a
  warning once at construction in that configuration. With `allow_unauthenticated=True` an
  anonymous principal is used, but the default `SmartLaunchPolicy` still denies (there is
  no token to carry a SMART `patient` claim), so unauthenticated development needs a
  custom `AccessPolicy`.
- `hmcp.shared.auth` exports exactly `OAuthClient`, `AuthenticationError` and
  `InvalidStateError`. The older auth exceptions were removed (`InvalidTokenError`,
  `ClientValidationError`, `ScopeError`, `InvalidCodeError`, `InvalidPKCEError`,
  `InvalidRedirectUriError`, `TokenExpiredError`, `RefreshTokenError`); catch
  `AuthenticationError` / `InvalidStateError` instead.
- `OAuthClient.register_client` defaults `token_endpoint_auth_method` to
  `client_secret_post` (only that or `none` is accepted), and
  `parse_authorization_response(..., expected_state=...)` verifies the CSRF `state`,
  raising `InvalidStateError` on a missing or mismatched value.

## Guardrails behavior

- Result `_meta` is inspected in the same way as structured content: every key except
  the guardrails extension key is walked, and the guardrails key is always replaced
  (non-empty report) or stripped (empty report), so a tool cannot forge it.
- Structured content and `_meta` are normalized with pydantic's JSON serialization before
  inspection, so a non-JSON value cannot hide PHI. Content that cannot be serialized fails
  closed with an `error` verdict.
- `timeout_ms` can only interrupt an evaluator that yields to the event loop (prompt, deid, classifier).
  The synchronous `pattern` and `builtin` evaluators run to completion, so a
  deployer-supplied regex must be linear in the input size.
- A provider or model error surfaces only as a fixed-message evaluation error (`error`
  verdict); the provider's own message is never propagated.
- The `classifier` evaluator kind calls a deployer-supplied `ClassifierProvider`
  (HMCP ships no classifier and no vendor adapter). A `classifier` definition
  supports the `block` and `annotate` actions only and is not a cascade step.

## Running the tests

From the repository root:

```bash
pytest
```

## Specification

[HMCP specification](../../docs/specification/index.md)
