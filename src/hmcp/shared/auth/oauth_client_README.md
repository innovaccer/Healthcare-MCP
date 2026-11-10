# HMCP OAuth client

An OAuth 2.1 client (OAuth 2.0 + PKCE) for HMCP that communicates with any OAuth 2.1 authorization server and supports the Client Credentials and Authorization Code flows with PKCE.

`OAuthClient` discovers endpoints from the authorization server's RFC 8414
`/.well-known/oauth-authorization-server` document instead of hard-coding them, uses
`httpx2` (consistent with the MCP SDK), and returns typed `OAuthToken` objects from
`mcp.shared.auth`.

## Features

- **OAuth 2.1 Flows (OAuth 2.0 + PKCE)**
  - Client credentials flow
  - Authorization code flow with PKCE (S256)
  - Refresh token flow
  - Token revocation (RFC 7009)

- **Protocol features**
  - PKCE implementation for Authorization Code flow
  - Token cached in memory on the `OAuthClient` instance (no persistence or encryption)
  - State parameter for CSRF protection
  - Token revocation support

## Usage

### Basic setup

```python
from hmcp.shared.auth.oauth_client import OAuthClient

# Create OAuth client
async with OAuthClient(
    client_id="your-client-id",
    client_secret="your-client-secret",
    server_url="http://localhost:8050"
) as client:
    # Use the client for authentication
    pass
```

The client must be used as an async context manager: entering it opens the underlying
HTTP client, and leaving it closes it.

### HTTP client and the test seam

The client is built on `httpx2`. `OAuthClient(...)` accepts a keyword-only `transport`
argument of type `httpx2.AsyncBaseTransport`. It is a **test seam**: a mock transport can be
passed to exercise the flows without network access. The same `transport` is accepted by
`HMCPClientConnector(...)`, which forwards it to its OAuth client.

```python
import httpx2
from hmcp.shared.auth.oauth_client import OAuthClient

async with OAuthClient(
    client_id="demo-client",
    client_secret="demo-secret",
    server_url="http://localhost:8050",
    transport=httpx2.MockTransport(handler),
) as client:
    ...
```

### Client credentials flow

```python
async with OAuthClient(
    client_id="demo-client",
    client_secret="demo-secret",
    server_url="http://localhost:8050"
) as client:
    # Get access token
    token_response = await client.get_client_credentials_token(scopes=["hmcp:access"])
    
    # The token is cached on the client automatically — no extra step needed.
    
    # Get authorization header for requests
    headers = client.get_auth_header()
    
    # `token_response` is an OAuthToken model with fields:
    # {
    #     "access_token": "...",
    #     "token_type": "Bearer",
    #     "expires_in": 3600,
    #     "scope": "hmcp:access"
    # }
```

### Authorization code flow with PKCE

```python
async with OAuthClient(
    client_id="web-client",
    client_secret="web-secret",
    server_url="http://localhost:8050"
) as client:
    # Start authorization flow
    auth_url, state, code_verifier = await client.start_authorization_code_flow(
        redirect_uri="http://localhost:8050/oauth/callback",
        scope="hmcp:access",
        state="optional_state"  # Optional state parameter for CSRF protection
    )
    
    # Redirect user to auth_url
    # After user authorization, you'll receive the code at your redirect URI
    
    # Parse the redirect URL to get the code and state. Pass the `state`
    # returned above as `expected_state` so a missing or mismatched state is
    # rejected with InvalidStateError (CSRF protection, RFC 6749 §10.12).
    code, returned_state = OAuthClient.parse_authorization_response(
        redirect_url,
        expected_state=state,
    )
    
    # Exchange code for token
    token_response = await client.exchange_code_for_token(
        code=code,
        redirect_uri="http://localhost:8050/oauth/callback",
        code_verifier=code_verifier
    )
    
    # The token is cached on the client automatically.
    
    # `token_response` is an OAuthToken model with these fields:
    # {
    #     "access_token": "...",
    #     "refresh_token": "...",
    #     "token_type": "Bearer",
    #     "expires_in": 3600,
    #     "scope": "hmcp:access"
    # }
```

### Refresh token flow

```python
async with OAuthClient(
    client_id="web-client",
    client_secret="web-secret",
    server_url="http://localhost:8050"
) as client:
    # A refresh token is issued by the Authorization Code flow (a
    # client-credentials grant returns none). `token_response` here is the
    # token returned by the Authorization Code flow shown above; it is cached
    # on the client automatically.
    
    # Refresh the token when needed. Called with no argument, the cached
    # refresh token is used; pass one explicitly if you manage it yourself.
    # If the server does not rotate the refresh token, the one just used is
    # kept on the returned token.
    refresh_response = await client.refresh_access_token()
    
    # `refresh_response` is an OAuthToken model with fields:
    # {
    #     "access_token": "...",
    #     "token_type": "Bearer",
    #     "expires_in": 3600,
    #     "scope": "hmcp:access"
    # }
```

The client does not track token expiry (`OAuthToken` carries only the relative `expires_in`)
and never refreshes automatically; call `refresh_access_token()` when a fresh token is
required.

### Token revocation

```python
async with OAuthClient(
    client_id="web-client",
    client_secret="web-secret",
    server_url="http://localhost:8050"
) as client:
    # `token_response` is the token returned by one of the flows above.
    
    # Revoke access token. Called with no arguments, revoke_token()
    # uses the cached token.
    await client.revoke_token(
        token=token_response.access_token,
        token_type_hint="access_token"  # Optional
    )
    
    # Revoke refresh token
    await client.revoke_token(
        token=token_response.refresh_token,
        token_type_hint="refresh_token"  # Optional
    )
```

### Dynamic client registration (deprecated)

`register_client()` performs RFC 7591 dynamic client registration against the
authorization server's `registration_endpoint`. MCP **2026-07-28 deprecates dynamic client
registration** in favor of **CIMD** (Client ID Metadata Document), in which the client's
metadata is hosted at an HTTPS URL that also serves as its client identifier. CIMD is
preferred; `register_client()` is used only against a server that still advertises a
registration endpoint.
The deprecation is a base-spec change; the method remains in the SDK.

`register_client()` defaults `token_endpoint_auth_method` to `"client_secret_post"` (the
secret travels in the token-request body, which is what the token requests actually send).
It accepts only `"client_secret_post"` or `"none"` (a public client that authenticates with
PKCE only); any other value raises `ValueError` before any request is made. HTTP Basic
client authentication is not implemented.

```python
info = await client.register_client(
    redirect_uris=["http://localhost:8050/oauth/callback"],
    token_endpoint_auth_method="client_secret_post",
)
# `info.client_id` and `info.client_secret` are stored on `client`.
```

## Error handling

The OAuth client raises `AuthenticationError` for OAuth error conditions. For an error
response from the authorization server, the message carries only the HTTP status and, when
the server returns one, the **validated RFC 6749 `error` code**. No other part of the
response body is included, because an authorization server may echo the submitted request
(including the `client_secret`).

A failed token request therefore surfaces as, for example,
`OAuth /token HTTP 401 (invalid_client)`. The `(error_code)` suffix appears only when the
JSON body carries a short lowercase `error` string in the RFC 6749 form; anything else is
dropped.

`hmcp.shared.auth` exports exactly `OAuthClient`, `AuthenticationError`, and
`InvalidStateError` (an `AuthenticationError` subclass, raised on a missing or mismatched
authorization-response `state`). A 2xx response whose body is not valid JSON, or is not a
valid token, metadata or client-registration document, also raises `AuthenticationError`,
with a fixed message that does not include the body. An `AuthenticationError` never chains
the underlying `httpx2` error: the HTTP error is not its `__cause__` or its `__context__`.

```python
from hmcp.shared.auth.oauth_client import AuthenticationError

try:
    async with OAuthClient(...) as client:
        token_response = await client.get_client_credentials_token()
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except RuntimeError as e:
    print(f"Client must be used as async context manager: {e}")
```

Common error scenarios:
- Invalid client credentials
- Invalid authorization code
- Invalid refresh token
- Token revocation failure
- Missing refresh token
- Not authenticated (when getting auth header)

## Best practices

1. **Token Management**
   - Tokens are cached on the client automatically; read them via the `access_token` property
   - Use `get_auth_header()` for authenticated requests
   - Refresh tokens before they expire
   - Revoke tokens when no longer needed

2. **PKCE Usage**
   - Always use PKCE for Authorization Code flow
   - Store code verifier securely
   - Use state parameter for CSRF protection
   - Parse redirect URL using `parse_authorization_response()`

3. **Error Handling**
   - Use async context manager (`async with`)
   - Handle `AuthenticationError` for OAuth errors
   - Handle `RuntimeError` for context manager issues
   - Implement retry logic for network issues

4. **Security**
   - Keep client credentials secret
   - Use HTTPS for all communications
   - Validate all responses
   - Implement proper token storage
   - Use state parameter for CSRF protection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit the changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License; see the LICENSE file for details.
