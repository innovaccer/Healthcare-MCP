# Patient context

> **Note:** Statements are labeled **normative** (a requirement of the HMCP protocol,
> RFC 2119), **SDK behavior** (what the code in `src/` does today), or **non-goal /
> future work**. Where the SDK implements less than the protocol requires, it is
> called out. All examples are synthetic.

MCP 2026-07-28 is a [stateless](https://modelcontextprotocol.io/specification/2026-07-28/basic.md#statelessness)
protocol: there are no protocol sessions and no initialize handshake, and state that
spans requests **MUST** be referenced by an explicit identifier the client sends on each
request. HMCP's session concept is the **patient context**.

Patient context is the `com.innovaccer/hmcp-patient-context` extension, revision `"1"`.
It is a server-minted, opaque handle that the **host** (never the model) opens through
extension methods. The handle is bound to the authenticated principal, has a TTL and a
lifecycle, and is carried by every patient-scoped tool call in request `_meta`. Every
check fails closed.

## Concept

> **Note:** A SMART-on-FHIR `patient` claim indicates the patient a token is intended for,
> but it is not a scoped, revocable, server-verified session. HMCP mints a handle only
> after an `AccessPolicy` has authorized the open, so the server keeps control of the
> patient binding and can revoke or suspend it later. The handle is opaque (`pc_` + 32
> random bytes, URL-safe base64) and is the only value a tool call requires to identify the
> patient.

> **Note:** The handle is protocol metadata, not part of a tool's domain arguments.
> Placing it in `_meta` keeps it out of the model's argument surface (the model cannot
> choose or forge a patient), keeps tool schemas clean, and lets the interceptor enforce
> it uniformly before the tool body runs. It also matches the base-spec
> [`_meta`](https://modelcontextprotocol.io/specification/2026-07-28/basic.md#_meta)
> convention. The extension's `_meta` key is:

```json
"com.innovaccer/hmcp-patient-context": { "contextId": "pc_<token>" }
```

> **Important:** A single request carries exactly one context, so a single request can
> never touch two patients. Switching patient means opening another context; the old one
> keeps its own lifecycle. (**normative**)

> **Important:** Missing, malformed, unknown, expired, closed, revoked, suspended or
> wrong-principal context all reject the call with a specific HMCP error code (below), and
> denials are audited. (**normative**)

## Lifecycle

A context is a frozen `PatientContext` record with a `status` from `ContextStatus`:

| Status | Meaning |
| --- | --- |
| `active` | Usable until `expiresAt`. |
| `suspended` | Temporarily blocked (access under review); can be resumed. |
| `revoked` | Permanently blocked. |
| `closed` | Closed by the host through the `close` method. |
| `expired` | TTL elapsed. |

Transitions are an explicit table. `REVOKED`, `CLOSED` and `EXPIRED` are **terminal and
sticky**: a later operation on a terminal context is skipped (not counted) and never
resurrects it. (**SDK behavior**, `src/hmcp/server/patient_context/extension.py`)

| Operation | Allowed from | To |
| --- | --- | --- |
| `revoke` | `active`, `suspended` | `revoked` |
| `suspend` | `active` | `suspended` |
| `resume` | `suspended` | `active` |
| `close` (method) | `active`, `suspended` | `closed`; from a terminal state: no change, the current status is returned |
| expiry (interceptor) | `active`, `suspended` | `expired` |

Contexts in any other state are skipped by `revoke`/`suspend`/`resume`. The TTL is
`contextTtlSeconds` (default 3600, allowed 60–86400); an elapsed TTL wins over `active` or
`suspended` when a tool is called.

## Extension methods

The extension advertises itself in
[`server/discover`](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
under the identifier `com.innovaccer/hmcp-patient-context` and declares three methods.
Its advertised settings are (**SDK behavior**, `PatientContextExtension.settings`):

```json
{ "revision": "1", "contextTtlSeconds": 3600, "defaultToolScope": "patient" }
```

A client that wants to use the extension declares it per request in
`_meta["io.modelcontextprotocol/clientCapabilities"].extensions` (base-spec
[extension negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning.md#extension-negotiation)).
Declaration is **not** a bypass: the interceptor runs regardless. (**SDK behavior**)

The params of `open` are validated with the same patterns as `PatientContext`
(`purposeOfUse` `^[A-Z]{3,12}$`, `encounter` `^Encounter/[A-Za-z0-9\-\.]{1,64}$`), so a bad
value is rejected before the policy runs. The server-side param models are
`PatientContextOpenParams` and `PatientContextIdParams` (subclasses of
`mcp.types.RequestParams`, which aliases snake_case fields to camelCase on the wire).

### `hmcp/patientContext/open`

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "hmcp/patientContext/open",
  "params": {
    "patient": {
      "reference": "Patient/test-0001",
      "identifiers": [
        { "system": "http://hospital.example/mrn", "value": "TEST-0001" }
      ]
    },
    "encounter": "Encounter/test-01",
    "purposeOfUse": "TREAT",
    "correlationId": "abc"
  }
}
```

`patient.reference` is required; `patient.identifiers`, `encounter` and `correlationId`
are optional; `purposeOfUse` is required. Result:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "contextId": "pc_V1StGXR8_Z5jdHi6B-myT4Z2M2b3JQb0h0dG9rZW5fMzI",
    "status": "active",
    "expiresAt": "2026-11-08T12:34:56.789012+00:00"
  }
}
```

Steps (**SDK behavior**):

1. Resolve the principal. No principal → `PATIENT_NOT_AUTHORIZED`.
2. `policy.authorize_open(...)`. A policy exception → `PATIENT_CONTEXT_UNAVAILABLE`.
   Denied → `PATIENT_NOT_AUTHORIZED`.
3. Build the `PatientContext`. **Only policy-verified identifiers are kept**:
   `patient.identifiers` becomes `decision.verified_identifiers`, so client-supplied
   identifiers that the policy did not verify are dropped.
4. Emit the `attempt` audit event. This emit is fail-closed (see [Audit](#audit-events)),
   so a sink failure here stops the open before the context is stored.
5. `store.put(...)`. Any failure, including `ContextStoreFull`, →
   `PATIENT_CONTEXT_UNAVAILABLE`.
6. Emit `success` and return the result.

The result `expiresAt` is ISO-8601 with a UTC offset; `contextId` starts with `pc_`.

### `hmcp/patientContext/get`

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "hmcp/patientContext/get",
  "params": { "contextId": "pc_V1StGXR8_Z5jdHi6B-myT4Z2M2b3JQb0h0dG9rZW5fMzI" }
}
```

Result:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "contextId": "pc_V1StGXR8_Z5jdHi6B-myT4Z2M2b3JQb0h0dG9rZW5fMzI",
    "status": "active",
    "expiresAt": "2026-11-08T12:34:56.789012+00:00",
    "patient": { "reference": "Patient/test-0001", "identifiers": [] },
    "encounter": "Encounter/test-01",
    "purposeOfUse": "TREAT"
  }
}
```

An unknown id **and** an id belonging to a different principal both return
`PATIENT_CONTEXT_INVALID`, deliberately the same error for both so the response is not an
existence oracle. (**SDK behavior**)

### `hmcp/patientContext/close`

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "hmcp/patientContext/close",
  "params": { "contextId": "pc_V1StGXR8_Z5jdHi6B-myT4Z2M2b3JQb0h0dG9rZW5fMzI" }
}
```

Result:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": { "contextId": "pc_V1StGXR8_Z5jdHi6B-myT4Z2M2b3JQb0h0dG9rZW5fMzI", "status": "closed" }
}
```

Closing a context in a terminal state makes no change and returns its current status.
Unknown/other-principal id → `PATIENT_CONTEXT_INVALID`. (**SDK behavior**)

All three methods emit audit events, and no error `message` or `data` ever contains the
patient reference or any identifier.

A request that resolves to no principal (a server with no authentication configured and
`allow_unauthenticated=False`) is rejected by all three methods, and by every patient- or
population-scoped tool call, with `PATIENT_NOT_AUTHORIZED` (-31005) and the message
`Request is not authenticated`. `HMCPServer` logs a warning once at construction in that
configuration. (**SDK behavior**)

The matching client models live in `src/hmcp/client/patient_context.py`:
`PatientContextOpenParams`, `PatientContextGetParams`, `PatientContextCloseParams`, and the
results `PatientContextOpenResult`, `PatientContextGetResult`,
`PatientContextCloseResult`.

## Tool scopes

Each tool has a **patient scope** that decides how the interceptor treats a call
(**normative**):

| Scope | Meaning |
| --- | --- |
| `patient` | Requires a usable patient context in `_meta`. Binds it for the call. |
| `population` | No patient context; the principal must be authorized for population access. |
| `none` | Untouched by the patient-context interceptor (no principal, no audit). |

`ToolPatientScope` is the literal type `"patient" | "population" | "none"`.

`HMCPServer` makes the **advertised** scope equal the **enforced** scope
(**SDK behavior**, `src/hmcp/server/hmcp_server.py`):

- `add_tool(..., patient_scope=...)` and `@server.tool(patient_scope=...)` record the
  effective scope (`patient_scope or default_tool_scope`) in the tool's `_meta` under
  `com.innovaccer/hmcp-patient-context.scope`.
- `_scope_for(name)` reads that same registered `_meta` at call time; an unknown tool or a
  tool without a recorded scope falls back to `default_tool_scope`.
- Declaring `patient_scope` when no `patient_context` is configured raises `ValueError` at
  registration, and passing tools through the `MCPServer` `tools=` argument is refused when
  `patient_context` is configured. Both cases ensure a declared scope can never go
  unenforced.

## Interceptor

For every tool call the extension runs these checks in order (**SDK behavior**,
`PatientContextExtension.intercept_tool_call`):

| # | Condition | Result |
| --- | --- | --- |
| 1 | scope is `none` | call proceeds; no principal, no audit |
| 2 | no principal (and not `allow_unauthenticated`) | audit `denied`; `PATIENT_NOT_AUTHORIZED` |
| 3 | scope is `population` | `policy.authorize_population`; policy exception → `PATIENT_CONTEXT_UNAVAILABLE`; denied → `PATIENT_NOT_AUTHORIZED`; else audit `attempt`, run the tool, audit `success`/`error` |
| 4 | `_meta[com.innovaccer/hmcp-patient-context]` is not a dict, or `contextId` is missing/not a non-empty string | audit `denied`; `PATIENT_CONTEXT_REQUIRED` |
| 5 | `store.get(contextId)` raises | `PATIENT_CONTEXT_UNAVAILABLE` |
| 6 | context unknown, or `context.principal != principal` | audit `denied`; `PATIENT_CONTEXT_INVALID` |
| 7 | status is `active`/`suspended` and `now >= expires_at` | mark `expired` (best effort); audit `denied`; `PATIENT_CONTEXT_INVALID` |
| 8 | status is `suspended` | audit `denied`; `PATIENT_CONTEXT_SUSPENDED` |
| 9 | status is `revoked` | audit `denied`; `PATIENT_CONTEXT_REVOKED` |
| 10 | status is `closed` or `expired` | audit `denied`; `PATIENT_CONTEXT_INVALID` |
| 11 | token has a `patient` claim and `context.patient.reference != "Patient/<claim>"` | audit `denied`; `PATIENT_NOT_AUTHORIZED` |
| 12 | otherwise | audit `attempt` (fail-closed), bind the context, run the tool, audit `success`/`error` |

Every rejection emits an audit event with the appropriate outcome (`denied`, or `error`
for a store or policy failure) and `error_code` set, before raising. If the tool itself
raises, the interceptor emits `error` (best effort) and re-raises. The `attempt` emit before the tool runs is the only fail-closed one: if it
fails, the tool never runs and `PATIENT_CONTEXT_UNAVAILABLE` is raised.

## Principal binding and the SMART patient claim ceiling

The principal is resolved per request from the OAuth access token:
`get_access_token()` → `Principal.from_access_token(token)` (client id, issuer, subject,
resource). When `allow_unauthenticated=True` (dev only, e.g. stdio) an anonymous principal
is used instead; `PatientContextConfig` logs a warning once when it is constructed with
that flag, not on each request. Otherwise there is no principal. (**SDK behavior**)

A context is bound to the derived principal, not to the raw token, so a token refreshed
mid-context keeps the same handle. Binding is therefore only as strong as the verifier's
identity claims: two tokens of the same client that carry neither a subject nor an `iss`
claim resolve to the same principal, and a handle opened with one can be used with the
other. A `TokenVerifier` must populate `AccessToken.subject` (and the `iss` claim) for
per-user binding. Any call whose principal differs from the context's gets
`PATIENT_CONTEXT_INVALID` (check 6). (**normative**)

Independently, the SMART `patient` claim is a **ceiling**: if the current token carries a
string `patient` claim, then `context.patient.reference` **MUST** equal
`Patient/<claim>`, or the call is denied with `PATIENT_NOT_AUTHORIZED` (check 11). The
same rule is applied at open time by the default policy. This means a token can never be
used to reach a patient other than the one it was issued for, even with a valid handle.
(**normative**)

## Access policy

The extension asks an `AccessPolicy` two questions, and every answer is an
`AccessDecision(allowed, reason, verified_identifiers=())`. `reason` is a fixed machine
code and is never PHI. (**SDK behavior**, `src/hmcp/server/patient_context/policy.py`)

- `authorize_open(principal, token, patient, purpose_of_use)`: decides whether this
  principal may open a context for this patient.
- `authorize_population(principal, token, tool_name)`: decides whether this principal may
  call a population-scoped tool.

`SmartLaunchPolicy` is the default. Its rules, in order:

`authorize_open`:

1. `purpose_of_use` not in the allowed set (`{"TREAT", "HPAYMT", "HOPERAT"}`) → deny
   `purpose_not_allowed`.
2. Token has a non-empty string `patient` claim → allow iff
   `patient.reference == "Patient/<claim>"`, else deny `token_patient_mismatch`.
3. No such claim → deny `no_patient_launch_context`. (Population tokens need a custom
   policy, for example one that calls an EMPI access API.)

It never verifies identifiers, so `verified_identifiers` is always empty. A deployment that
can confirm identifiers should supply its own policy.

`authorize_population`:

1. No token → deny `unauthenticated`.
2. Token has a string `patient` claim → deny `single_patient_token`.
3. Any scope starts with `user/` or `system/` → allow `population_scope`.
4. Otherwise → deny `no_population_scope`.

## Audit events

Every open/get/close, every tool call and every lifecycle change emits an `AuditEvent`
(**SDK behavior**, `src/hmcp/server/patient_context/audit.py`). Its fields are:

| Field | Notes |
| --- | --- |
| `timestamp` | tz-aware UTC |
| `action` | `patient_context.open` / `.get` / `.close` / `.revoke` / `.suspend` / `.resume` / `tool.call` |
| `outcome` | `attempt` / `success` / `denied` / `error` |
| `principal` | resolved `Principal` (or null) |
| `context_id` | the handle |
| `patient_reference` | `Patient/<id>`, required by the access trail |
| `purpose_of_use` | HL7 v3 code |
| `tool` | tool name for `tool.call` |
| `error_code` | numeric HMCP error code on denials/errors |
| `reason` | `AccessDecision.reason` or a fixed code |
| `guardrails` | `(guardrail id, verdict)` pairs parsed from the result report |
| `correlation_id` | caller correlation id |

`AuditEvent` deliberately has **no** field for tool arguments, result text, identifier
values, names or tokens, and it forbids extra fields, so none can be added by accident.
The default `LoggingAuditSink` writes one JSON line per event to the `hmcp.audit` logger.

Emission is mostly best effort: a sink failure on a denial, success, error or lifecycle
event is logged by exception **type name only** and never changes the result or the HMCP
error code being raised. The single exception is the `attempt` emit before a tool call (and
before `open`'s `store.put`), which is fail-closed (see [Interceptor](#interceptor)).

## Revocation and suspension

The host drives lifecycle changes on the installed extension through the
`HMCPServer.patient_contexts` property (**SDK behavior**):

```python
extension = server.patient_contexts
if extension is not None:
    revoked = await extension.revoke(patient_reference="Patient/test-0001")
    suspended = await extension.suspend(subject="user-a")
    resumed = await extension.resume(context_id="pc_V1StGXR8_Z5jdHi6B-myT4Z2M2b3JQb0h0dG9rZW5fMzI")
```

Each returns the number of contexts transitioned. When `context_id` is given it selects
that single context and the other filters are ignored; otherwise `patient_reference` and
`subject` are combined with AND by the store (`subject` matches `principal.subject`). At
least one filter is required: calling with no filter raises `ValueError`. This is the
hook for a UAM/access-revocation event. A revoked or
suspended context is rejected on the next tool call with `PATIENT_CONTEXT_REVOKED` or
`PATIENT_CONTEXT_SUSPENDED`, and terminal states are sticky.

## Propagation to downstream services

**normative, not implemented by the SDK.** When an HMCP server calls another
patient-data service, it **MUST NOT** forward the caller's access token or the raw
`contextId` as proof of patient context. Instead it should obtain a new token for the
downstream audience through token exchange (RFC 8693) and open a fresh downstream patient
context for the same patient. The downstream service then applies its own policy and
audit. The SDK does not implement token exchange or downstream context propagation; a
deployment does this itself.

## SDK usage

Server (**verified by execution**, `src/hmcp/server/hmcp_server.py`):

```python
from hmcp.server.hmcp_server import HMCPServer
from hmcp.server.patient_context import PatientContextConfig
from hmcp.shared.types import require_patient_context

server = HMCPServer(
    name="patient-demo",
    version="0.1.0",
    patient_context=PatientContextConfig(
        context_ttl_seconds=900,
        default_tool_scope="patient",
    ),
)


@server.tool(patient_scope="patient")
def get_observations() -> dict:
    context = require_patient_context()
    return {"patient": context.patient.reference}


@server.tool(patient_scope="population")
def count_patients() -> dict:
    return {"count": 0}


@server.tool(patient_scope="none")
def ping() -> str:
    return "pong"
```

Inside a patient-scoped tool, `require_patient_context()` returns the bound
`PatientContext` (raising `PATIENT_CONTEXT_REQUIRED` if none is bound) and
`current_patient_context()` returns it or `None`. The context is bound for the duration of
the call only.

Client (**verified by execution**; `src/hmcp/client/client_connector.py`):

```python
from hmcp.client.client_connector import HMCPClientConnector

async with HMCPClientConnector(
    url="http://localhost:8050/mcp",
    headers={"Authorization": "Bearer <token>"},
) as connector:
    handle = await connector.open_patient_context(
        "Patient/test-0001",
        purpose_of_use="TREAT",
    )
    result = await connector.call_tool(
        "get_observations",
        patient_context=handle,
    )
    context = await connector.get_patient_context(handle)
    print(context.context_id, context.status, context.patient.reference)
    await connector.close_patient_context(handle)
```

`open_patient_context` returns a `PatientContextHandle`. `get_patient_context` returns a
`PatientContextGetResult` with snake_case attributes (`context_id`, `status`, `expires_at`,
`patient`, `encounter`, `purpose_of_use`), not a camelCase dict; `close_patient_context`
returns `None`. `call_tool(patient_context=...)` accepts either the handle or a bare context
id and writes it into request `_meta` under `com.innovaccer/hmcp-patient-context`; a
caller-supplied `_meta` value under either HMCP extension key that is not a mapping raises
`TypeError` before the request is sent. The connector is an async context manager (it
connects on entry and cleans up on exit) and advertises the patient-context and guardrails
client extensions by default.
