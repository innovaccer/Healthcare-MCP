from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from hmcp.server.guardrails import GuardrailsConfig
from hmcp.server.hmcp_server import HMCPServer
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
