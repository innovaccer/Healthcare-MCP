# HMCP OAuth Client

A secure OAuth 2.0 client SDK for interacting with the HMCP OAuth server, supporting Client Credentials and Authorization Code flows with PKCE.

## Features

- **OAuth 2.0 Flows**
  - Client Credentials Flow
  - Authorization Code Flow with PKCE
  - Refresh Token Flow
  - Token Revocation

- **Security Features**
  - PKCE implementation for Authorization Code flow
  - Secure token storage
  - State parameter for CSRF protection
  - Token revocation support

## Usage

### Basic Setup

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

### Client Credentials Flow

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

### Authorization Code Flow with PKCE

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
    
    # Parse the redirect URL to get the code and state
    code, returned_state = OAuthClient.parse_authorization_response(redirect_url)
    # Compare returned_state against the `state` returned above for CSRF protection
    
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

### Refresh Token Flow

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
    refresh_response = await client.refresh_access_token()
    
    # `refresh_response` is an OAuthToken model with fields:
    # {
    #     "access_token": "...",
    #     "token_type": "Bearer",
    #     "expires_in": 3600,
    #     "scope": "hmcp:access"
    # }
```

### Token Revocation

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

### Using with HMCP Client

```python
from hmcp.client.hmcp_client import HMCPClient
from hmcp.shared.auth.oauth_client import OAuthClient
from mcp.client.sse import sse_client
from mcp.types import SamplingMessage, TextContent
from mcp import ClientSession

async def connect_to_agent():
    # Initialize OAuth client
    async with OAuthClient(
        client_id="your-client-id",
        client_secret="your-client-secret",
        server_url="http://localhost:8050"
    ) as oauth_client:
        # Get access token (cached on the client; read via get_auth_header() below)
        token_response = await oauth_client.get_client_credentials_token()
        
        # Connect to HMCP server
        async with sse_client(
            "http://localhost:8050/sse",
            headers=oauth_client.get_auth_header()
        ) as (read, write):
            async with ClientSession(read, write) as session:
                client = HMCPClient(session)
                
                # Send a message
                response = await client.create_message(messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text="Your message here"
                        )
                    )
                ])
                
                # Process the response
                print(response.content.text)
```

## Error Handling

The OAuth client raises `AuthenticationError` for various error conditions:

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

## Best Practices

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
   - Keep client credentials secure
   - Use HTTPS for all communications
   - Validate all responses
   - Implement proper token storage
   - Use state parameter for CSRF protection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 