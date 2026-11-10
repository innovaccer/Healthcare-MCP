"""Demo-only bearer-token verifier for the EHR patient-context example.

An HMCP server authenticates every request the same way an MCP server does:
it asks a :class:`~mcp.server.auth.provider.TokenVerifier` to turn the bearer
token into an :class:`~mcp.server.auth.provider.AccessToken`. In production
that verifier would validate a signed JWT or call a token-introspection
endpoint. Here two static tokens stand in for a real identity provider so the
demo runs with no external service.

DEMO ONLY: the tokens below are constants committed to this repository. They
grant access to synthetic patients only. Never ship static tokens like these.
"""

from __future__ import annotations

from mcp.server.auth.provider import AccessToken, TokenVerifier

# The MCP endpoint the server listens on. The server is built with
# validate_token_resource=True, so each token's ``resource`` must equal this
# URL: that is RFC 8707 audience binding, which stops a token minted for one
# server from being replayed against another.
SERVER_URL = "http://127.0.0.1:8050/mcp"

# The issuer of the demo tokens (a real deployment would validate this too).
DEMO_ISSUER = "https://auth.example.com"

# DEMO ONLY: two static bearer tokens, one per synthetic patient. The
# ``patient`` claim is the patient this token is scoped to; SmartLaunchPolicy
# reads it before allowing a patient context to be opened.
DEMO_TOKENS: dict[str, dict[str, str]] = {
    "demo-token-test-0001": {
        "client_id": "ehr-demo-app",
        "subject": "dr-demo",
        "patient": "test-0001",
    },
    "demo-token-test-0002": {
        "client_id": "ehr-demo-app",
        "subject": "dr-demo",
        "patient": "test-0002",
    },
}


class DemoTokenVerifier(TokenVerifier):
    """Recognise the two static demo tokens and reject everything else.

    HMCP (via the MCP SDK) calls :meth:`verify_token` for every HTTP request.
    Returning ``None`` rejects the request; returning an ``AccessToken``
    accepts it and makes that token the request's principal. The principal is
    later compared against the patient context, so one caller cannot use
    another caller's context.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        record = DEMO_TOKENS.get(token)
        if record is None:
            return None
        return AccessToken(
            token=token,
            client_id=record["client_id"],
            # A single read scope for the demo patient's records.
            scopes=["patient/*.rs"],
            resource=SERVER_URL,
            subject=record["subject"],
            claims={"iss": DEMO_ISSUER, "patient": record["patient"]},
        )
