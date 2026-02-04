# HMCP OAuth Client

A secure OAuth 2.0 client SDK for interacting with the HMCP OAuth server, supporting Client Credentials and Authorization Code flows with PKCE.

## Features

- **OAuth 2.0 Flows**
  - Client Credentials Flow
  - Authorization Code Flow with PKCE
  - Refresh Token Flow
  - Token Revocation
  - Token Introspection

- **Security Features**
  - PKCE implementation for Authorization Code flow
  - Secure token storage
  - State parameter for CSRF protection
  - Token revocation support
  - Token introspection capabilities
  - Patient context support

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
    token_response = await client.get_client_credentials_token(scope="hmcp:access")
    
    # Store token response
    client.set_token(token_response)
    
    # Get authorization header for requests
    headers = client.get_auth_header()
    
    # Token response contains:
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
    auth_url, code_verifier = await client.start_authorization_code_flow(
        redirect_uri="http://localhost:8050/oauth/callback",
        scope="hmcp:access",
        state="optional_state"  # Optional state parameter for CSRF protection
    )
    
    # Redirect user to auth_url
    # After user authorization, you'll receive the code at your redirect URI
    
    # Parse the redirect URL to get the code and state
    code, state = OAuthClient.parse_authorization_response(redirect_url)
    
    # Exchange code for token
    token_response = await client.exchange_code_for_token(
        code=code,
        redirect_uri="http://localhost:8050/oauth/callback",
        code_verifier=code_verifier
    )
    
    # Store token response
    client.set_token(token_response)
    
    # Token response contains:
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
    # Set initial token response
    client.set_token(token_response)
    
    # Refresh token when needed
    refresh_response = await client.refresh_access_token(
        refresh_token=token_response["refresh_token"]
    )
    
    # Store new token response
    client.set_token(refresh_response)
    
    # Refresh response contains:
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
    # Revoke access token
    await client.revoke_token(
        token=token_response["access_token"],
        token_type_hint="access_token"  # Optional
    )
    
    # Revoke refresh token
    await client.revoke_token(
        token=token_response["refresh_token"],
        token_type_hint="refresh_token"  # Optional
    )
```

### Token Introspection

```python
async with OAuthClient(
    client_id="web-client",
    client_secret="web-secret",
    server_url="http://localhost:8050"
) as client:
    # Introspect token
    introspection_response = await client.introspect_token(
        token=token_response["access_token"]
    )
    
    # Introspection response contains:
    # {
    #     "active": true,
    #     "scope": "hmcp:access",
    #     "client_id": "web-client",
    #     "token_type": "access_token",
    #     "exp": 1746723900,
    #     "iat": 1746720300,
    #     "iss": "HMCP_Server",
    #     "aud": "https://hmcp-server.example.com"
    # }
```

### Using with HMCP Client

```python
from hmcp.mcpclient.hmcp_client import HMCPClient
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
        # Get access token
        token_response = await oauth_client.get_client_credentials_token()
        oauth_client.set_token(token_response)
        
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
- Token introspection failure
- Missing refresh token
- Not authenticated (when getting auth header)

## Best Practices

1. **Token Management**
   - Always use `set_token()` to store token responses
   - Use `get_auth_header()` for authenticated requests
   - Refresh tokens before they expire
   - Revoke tokens when no longer needed
   - Use token introspection to validate tokens

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