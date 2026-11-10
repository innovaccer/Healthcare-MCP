# Comparison of MCP and HMCP authentication implementations

This document compares the base MCP authorization model (revision `2026-07-28`)
with the healthcare-specific layer that HMCP adds.

## 1. Base standards

- **MCP**: OAuth 2.1, with these `2026-07-28` authorization rules added:
  Protected Resource Metadata (RFC 9728), resource indicators and audience
  validation (RFC 8707), issuer identification (RFC 9207), and OAuth Client ID
  Metadata Documents (CIMD). Dynamic Client Registration (RFC 7591) is
  deprecated but retained for compatibility.
- **HMCP**: SMART on FHIR, which builds upon OAuth 2.0 and OpenID Connect, plus
  the base MCP authorization rules for HTTP transports.

## 2. Healthcare specificity

- **MCP**: general-purpose protocol without healthcare-specific considerations.
- **HMCP**: designed for healthcare data exchange, adding a patient-context
  extension and healthcare-specific scopes.

## 3. Authentication flows

- **MCP**: standard OAuth 2.1 flows with emphasis on the authorization code
  grant, PKCE, and the `resource` parameter on authorization and token requests.
- **HMCP**: defines two flows in addition:
  - OAuth 2.0 Authorization Code Flow (user-mediated)
  - Client Credentials Flow (service-to-service)

  **SDK behavior.** `OAuthClient`
  (`src/hmcp/shared/auth/oauth_client.py`) implements the client-credentials,
  authorization-code (PKCE S256), refresh and revocation flows, discovering
  endpoints from RFC 8414 authorization-server metadata. It does not send the
  RFC 8707 `resource` parameter, does not track token expiry (`OAuthToken`
  carries only the relative `expires_in`) and never refreshes automatically.

## 4. Scopes system

- **MCP**: generic OAuth scopes; servers **SHOULD** advertise required scopes in
  the `WWW-Authenticate` challenge.
- **HMCP**: healthcare-specific scopes. The SMART grammar is
  `[patient|user|system]/<resource>.<permissions>` (e.g. `patient/Observation.rs`);
  HMCP also uses HMCP- and OAuth-specific scope strings such as `hmcp:access`,
  `patient/hmcp:read` and `launch/patient` that do not follow it. The SDK parses
  no scope grammar. See [auth.md](./auth.md#scopes-in-hmcp).

## 5. Patient context

- **MCP**: no patient-context mechanism.
- **HMCP**: carries patient context out of band through the
  `com.innovaccer/hmcp-patient-context` extension (revision `"1"`,
  `src/hmcp/shared/constants.py`) rather than through scopes alone. The SMART
  `patient` token claim is the access ceiling: an opened context must not grant
  access to a patient other than the one named in the token. See
  [patient-context.md](./patient-context.md) for the extension methods, the
  context lifetime and the error codes.

## 6. Token implementation

- **MCP**: bearer tokens per OAuth 2.1; servers **MUST** validate that a token
  was issued for them as the intended audience (RFC 8707).
- **HMCP**: bearer JWTs carrying healthcare-specific claims including `patient`
  (SMART patient claim, the access ceiling), `fhirUser` (FHIR practitioner
  reference) and `tenant` (organization identifier). **SDK behavior.**
  `HMCPServer` (`src/hmcp/server/hmcp_server.py`) enforces RFC 8707 audience
  binding by default by re-validating a supplied `AuthSettings` with
  `validate_token_resource=True` when a `resource_server_url` is set.

## 7. Client registration

- **MCP**: OAuth Client ID Metadata Documents (CIMD) are **preferred**; Dynamic
  Client Registration (RFC 7591) is deprecated.
- **HMCP**: **SDK behavior.** `OAuthClient.register_client`
  (`src/hmcp/shared/auth/oauth_client.py`) implements Dynamic Client
  Registration (RFC 7591) by POSTing to the discovered `registration_endpoint`,
  defaulting `token_endpoint_auth_method` to `client_secret_post` (only that or
  `none` is accepted). That is the deprecated mechanism; prefer CIMD with
  authorization servers that support it.

## 8. OpenID Connect integration

- **MCP**: mentions OpenID Connect discovery as an authorization-server
  discovery mechanism.
- **HMCP**: **Normative / deployment-level.** A conforming SMART deployment may
  use OpenID Connect for user authentication, ID tokens, the UserInfo endpoint,
  and SMART on FHIR claims. **SDK behavior.** HMCP's SDK performs no ID-token
  validation and makes no UserInfo call.

## 9. Security requirements

- **MCP**: OAuth 2.1 security best practices, plus token audience binding and
  validation, RFC 9207 issuer validation, and communication security.
- **HMCP**: the same, with healthcare-specific guidance for protected health
  information, and the patient-context ceiling described above.

HMCP is a specialized adaptation of authentication standards for healthcare use
cases, while the base MCP specification provides a general-purpose authorization
framework. HMCP's additions are the healthcare scopes and the patient-context
extension; all other aspects defer to MCP 2026-07-28.
