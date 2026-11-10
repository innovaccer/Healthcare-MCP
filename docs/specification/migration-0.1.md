# Migrating from HMCP 0.0.x to 0.1.0

HMCP 0.1.0 moves from `mcp[cli]==1.29.0` (FastMCP, MCP spec **2025-03-26**) to
`mcp>=2.3.0,<2.4` (MCPServer, MCP spec **2026-07-28**). HMCP no longer installs the
`[cli]` extra; if the `mcp` command-line tool is used, install `mcp[cli]` separately. The
2026 revision is a breaking change to the base protocol; HMCP removes two features as part
of the change:

* **Bidirectional sampling.** `@server.sampling()`, `HMCPClient` and
  `create_message` are removed.
* **HTTP+SSE transport.** HMCP 0.1.0 speaks stdio and streamable HTTP only.

This guide lists every removed or renamed public name, shows how agent-to-agent
use cases map onto the 2026 protocol, and gives before/after snippets. Every
statement is labeled **normative** (a rule from the MCP 2026-07-28 spec),
**SDK behavior** (what `src/` does today) or **non-goal** (not provided).

## Rationale

### Sessions and the initialize handshake were removed

The 2026-07-28 revision removed protocol sessions (SEP-2567) and the
`initialize` handshake (SEP-2575). Every request is self-contained: it carries
its protocol version, client identity and client capabilities in `_meta`, and
the server accepts or rejects each request independently. See
[Versioning and Compatibility](https://modelcontextprotocol.io/specification/2026-07-28/basic/lifecycle)
and [Transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports).

Consequences for HMCP:

* **Normative.** There is no `initialize` request. Servers **MUST** implement
  [`server/discover`](https://modelcontextprotocol.io/specification/2026-07-28/server/discover);
  clients **MAY** call it first. Extensions are advertised in
  `server/discover` and per-request `clientCapabilities.extensions`.
* **SDK behavior.** `HMCPServer` and `HMCPClientConnector` build on the mcp
  2.3.0 client/server, so they no longer call `session.initialize()`, and the
  connector no longer constructs a `ClientSession` itself. It drives an
  `mcp.client.Client` over the streamable-HTTP transport (see
  `src/hmcp/client/client_connector.py`); that client's `.session` is still a
  `ClientSession`.

### Sampling is deprecated

**Normative.** Sampling is deprecated as of protocol `2026-07-28` (SEP-2577) and
will remain in the specification for at least twelve months before it becomes
eligible for removal; new implementations **SHOULD NOT** adopt it. See
[Sampling](https://modelcontextprotocol.io/specification/2026-07-28/client/sampling)
and the [deprecated features registry](https://modelcontextprotocol.io/specification/2026-07-28/deprecated).

**SDK behavior.** HMCP 0.1.0 removes its bidirectional sampling feature
entirely rather than carrying a deprecated one. The removed surface is listed
below.

### HTTP+SSE is deprecated

**Normative.** The 2026-07-28 revision defines two standard transports:
[stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
and
[Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http).
The older HTTP+SSE transport is deprecated.

**SDK behavior.** `HMCPServer.run(transport="sse")` raises `ValueError` with a
message naming the revision that deprecated SSE, and `HMCPServer.sse_app()`
raises the same error (`src/hmcp/server/hmcp_server.py`). The client connector
is streamable-HTTP only.

## Removed and renamed API

| HMCP 0.0.x | HMCP 0.1.0 |
| --- | --- |
| `HMCPServer(name, host=..., port=..., debug=..., log_level=..., version=..., instructions=..., auth_server_provider=..., event_store=..., samplingCallback=...)` | `HMCPServer(name, *, version="", instructions=None, token_verifier=None, auth=None, auth_server_provider=None, extensions=None, patient_context=None, guardrails=None, **kwargs)`. Transport options (`host`, `port`, `json_response`, `stateless_http`, …) move to `run()` / `streamable_http_app()`. |
| `server.run(transport="sse")` | `server.run(transport="streamable-http")` (or `"stdio"`). `"sse"` raises `ValueError`; `sse_app()` raises too. |
| `@server.sampling()`; `samplingCallback=`; `SamplingFnT`; `_default_sampling_callback`; `patched_get_capabilities`; `_registerSamplingHandler` | Removed. A server cannot send requests to a client outside the MRTR pattern (see below). |
| `HMCPClient` (`src/hmcp/client/hmcp_client.py`) | Module deleted. |
| `HMCPClientConnector.create_message(...)` | Removed. Expose the other agent's work as tools and call them with `call_tool(...)`. |
| `HMCPClientConnector.connect(transport=..., timeout=..., sse_read_timeout=..., headers=...) -> dict` | `connect()` takes no arguments, returns `None`, and is streamable-HTTP only. Configure `timeout=`, `headers=`, `client_id`/`client_secret`/`scopes` and `auth_server_url=` on the constructor. |
| `HMCPClientConnector(host=..., port=...)` (documented form) | `HMCPClientConnector(url="http://host:port/mcp")`; `url` is the full MCP endpoint. |
| `list_tools()` / `list_resources()` / `list_prompts()` returning `List[Dict[str, Any]]`; `read_resource()` returning a `dict`; `get_prompt()` returning a `str`; `call_tool()` returning a content list | `list_tools() -> list[mcp.types.Tool]`, `list_resources() -> list[mcp.types.Resource]`, `list_prompts() -> list[mcp.types.Prompt]`, `read_resource() -> mcp.types.ReadResourceResult`, `get_prompt() -> mcp.types.GetPromptResult`, `call_tool() -> mcp.types.CallToolResult`. |
| `import httpx` in `OAuthClient` and the connector | `import httpx2` (mcp 2.x replaces httpx with httpx2). |
| `experimental.hmcp` capability advertising sampling | Removed. Extensions are `com.innovaccer/hmcp-patient-context` (revision `"1"`) and `com.innovaccer/hmcp-guardrails` (revision `"1"`), advertised via `server/discover` and per-request `clientCapabilities.extensions` (`src/hmcp/shared/constants.py`). |
| `session.initialize()`, `mcp.client.sse.sse_client` | Not used. The connector uses `mcp.client.Client` over `streamable_http_client`; that client's `.session` is still a `ClientSession`, but the connector does not construct one itself. |

`OAuthClient` is not removed: it keeps its public API
(`get_client_credentials_token`, `set_client_credentials_token`,
`start_authorization_code_flow`, `exchange_code_for_token`, `register_client`,
`refresh_access_token`, `revoke_token`). Only its HTTP library changed, from httpx to httpx2
(`src/hmcp/shared/auth/oauth_client.py`).

## Agent-to-agent use cases, mapped

The 0.0.x design allowed an HMCP server to call back into the client with a
sampling request, so that two agents could exchange messages. The 2026 protocol
does not allow servers to initiate JSON-RPC requests, so each old use case maps
onto a different mechanism:

* **Expose the agent as a tool.** Instead of an orchestrator calling
  `create_message` on a peer agent, run the peer as an MCP server that exposes
  its capabilities as tools, and have the orchestrator call them with
  `tools/call`. In HMCP this is `HMCPClientConnector.call_tool(...)`
  (**SDK behavior**).
* **Long-running work: MCP Tasks.** Operations that take minutes or hours
  (batch jobs, human approvals) are handled by the base-MCP
  [Tasks extension](https://modelcontextprotocol.io/extensions/tasks/overview)
  (`io.modelcontextprotocol/tasks`): the server returns a durable task handle
  and the client polls `tasks/get`.

  > **Important:** This is a base-MCP feature that HMCP does not wrap. Use the
  > extension directly.
* **Clarification: MRTR elicitation.** When a server needs extra input before
  finishing a request, the 2026 revision uses the
  [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
  pattern: the server returns an `InputRequiredResult` carrying
  `elicitation/create` requests, the client gathers answers and retries the
  original request.

  > **Important:** This is a base-MCP feature that HMCP does not wrap.
* **Patient identifiers: patient context, not string round-trips.** The 0.0.x
  sampling examples passed a `patient_id` inside message text. HMCP 0.1.0
  carries patient identity out of band, in the patient-context extension: the
  client opens a context and attaches its `contextId` to each call, and
  patient-scoped tools fail closed without one. See
  [patient-context.md](./patient-context.md) (**SDK behavior**).

## New authentication defaults

**SDK behavior.** `HMCPServer` (`src/hmcp/server/hmcp_server.py`) changes how
an `AuthSettings` object is treated:

* If `auth` is supplied with a `resource_server_url` and no explicit
  `validate_token_resource`, HMCP re-validates the settings with
  `validate_token_resource=True`, so RFC 8707 audience binding is enforced by
  default.
* If `auth` is supplied **without** `resource_server_url`, HMCP raises
  `ValueError`, because it cannot audience-bind tokens. Set
  `validate_token_resource=False` only when the `TokenVerifier` checks the
  token's audience itself.
* If neither `token_verifier` nor `auth_server_provider` is supplied, HMCP logs
  the warning `HMCP server running without authentication` once at
  construction.

**Normative.** MCP servers **MUST** validate that an access token was issued
for them as the intended audience, per
[RFC 8707](https://www.rfc-editor.org/rfc/rfc8707.html) (see
[Authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)).

See [auth.md](./auth.md) and
[hmcp_auth_vs_mcp_auth.md](./hmcp_auth_vs_mcp_auth.md) for the complete
description.

## Minimal server: before and after

### Before (HMCP 0.0.x, not executed)

```python
# HMCP 0.0.x only. This snippet is illustrative and is NOT run by the docs build.
from hmcp.server.hmcp_server import HMCPServer

server = HMCPServer(name="healthcare-agent", host="0.0.0.0", port=8050, version="1.0.0")


@server.sampling()
async def handle_sampling(context, params):
    ...  # build and return a mcp.types.CreateMessageResult


server.run(transport="sse")
```

### After (HMCP 0.1.0)

```python
from hmcp.server.hmcp_server import HMCPServer

server = HMCPServer(name="healthcare-agent", version="1.0.0")


@server.tool()
def latest_blood_pressure(patient_id: str) -> str:
    """Return the latest blood pressure reading for a patient."""
    return "120/80"


if __name__ == "__main__":
    server.run(transport="streamable-http")
```

The transport is chosen at `run()` time, not at construction. Use
`run(transport="stdio")` for a client-launched subprocess. To require
authentication, add `token_verifier=` and `auth=AuthSettings(...)`:

```python
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from hmcp.server.hmcp_server import HMCPServer


class StaticTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        if token != "test-token":
            return None
        return AccessToken(
            token=token,
            client_id="test-client",
            scopes=["hmcp:read"],
            resource="https://mcp.example.com/mcp",
        )


server = HMCPServer(
    name="healthcare-agent",
    token_verifier=StaticTokenVerifier(),
    auth=AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url="https://mcp.example.com/mcp",
    ),
)
```

HMCP sets `validate_token_resource=True` here because it is omitted. Constructing
`AuthSettings` directly with `validate_token_resource` unset emits the SDK's
`MCPDeprecationWarning`; pass `validate_token_resource=True` explicitly to
silence it.

## Minimal client: before and after

### Before (HMCP 0.0.x, not executed)

```python
# HMCP 0.0.x only. This snippet is illustrative and is NOT run by the docs build.
from hmcp.client.client_connector import HMCPClientConnector

helper = HMCPClientConnector(url="http://localhost:8050")
await helper.connect(transport="streamable-http")
response = await helper.create_message(
    "What is the recommended treatment for hypertension?"
)
```

### After (HMCP 0.1.0)

```python
import asyncio

from hmcp.client.client_connector import HMCPClientConnector


async def main() -> None:
    async with HMCPClientConnector(url="http://127.0.0.1:8050/mcp") as connector:
        tools = await connector.list_tools()
        result = await connector.call_tool(
            "latest_blood_pressure", {"patient_id": "test-0001"}
        )
        print([tool.name for tool in tools], result.is_error)


if __name__ == "__main__":
    asyncio.run(main())
```

`url` is the full MCP endpoint; the connector does not append `/mcp`, so
include it (e.g. `http://host:port/mcp`). The connector is an
async context manager; `list_tools()` returns SDK `mcp.types.Tool` objects, and
`call_tool()` returns an `mcp.types.CallToolResult` whose attributes are
snake_case (`is_error`, `structured_content`). Client-credentials
authentication is configured on the constructor:

```python
connector = HMCPClientConnector(
    url="http://127.0.0.1:8050/mcp",
    client_id="test-client",
    client_secret="test-secret",
    scopes=["hmcp:read"],
)
```

## Changes since earlier 0.1.0 pre-release builds

These changes are not part of the 0.0.x → 0.1.0 protocol break, but they affect
deployments that began on an earlier 0.1.0 preview.

### Packaging

* **Install name.** The distribution is `innovaccer-hmcp`; the import name stays
  `hmcp`. Install it with `pip install innovaccer-hmcp`. The old instructions
  (`pip install .`, or the never-published `hmcp` distribution) are removed. From
  a source checkout, use `pip install -e ".[dev]"`.
* **mcp dependency.** HMCP now depends on `mcp>=2.3.0,<2.4` and no longer pulls the
  `mcp[cli]` extra. If the `mcp` command-line tool is used, install `mcp[cli]`
  separately.

### Authentication exceptions

The auth exception classes `InvalidTokenError`, `ClientValidationError`, `ScopeError`,
`InvalidCodeError`, `InvalidPKCEError`, `InvalidRedirectUriError`, `TokenExpiredError` and
`RefreshTokenError` were removed; none was ever raised. `hmcp.shared.auth` exports exactly
`OAuthClient`, `AuthenticationError` and `InvalidStateError` (an `AuthenticationError`
subclass). Catch `AuthenticationError` (or `InvalidStateError` for a bad
authorization-response `state`) instead.

### OAuth client

* `register_client` now defaults `token_endpoint_auth_method` to `client_secret_post` (the
  secret travels in the token-request body) and accepts only that or `none`. HTTP Basic
  client authentication is rejected with `ValueError` before any request is made.
* `parse_authorization_response(redirect_url, *, expected_state=...)` is the recommended
  call: pass the `state` returned by `start_authorization_code_flow`, so a missing or
  mismatched state raises `InvalidStateError` (CSRF protection).

### Client connector: `get_patient_context`

`HMCPClientConnector.get_patient_context` returns a `PatientContextGetResult` with
snake_case attributes, not a camelCase `dict`.

Before (0.1.0 previews, not executed):

```python
state = await connector.get_patient_context(handle)
status = state["status"]
expires_at = state["expiresAt"]
```

After (HMCP 0.1.0):

```python
result = await connector.get_patient_context(handle)
status = result.status
expires_at = result.expires_at
patient_reference = result.patient.reference
```

## Non-goals

**Non-goal.** HMCP 0.1.0 does not wrap the MCP Tasks extension or MRTR
elicitation, does not re-introduce a server-initiated request mechanism, and
does not provide the HTTP+SSE transport. Use the base-MCP features directly for
tasks and elicitation, and stdio or streamable HTTP for transport.
