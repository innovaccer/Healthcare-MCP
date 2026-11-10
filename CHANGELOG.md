# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

HMCP 0.1.0 is a breaking protocol change: it moves to `mcp>=2.3.0,<2.4` and MCP spec
**2026-07-28**, which removed protocol sessions and the initialize handshake and
deprecated sampling, and it replaces the removed session concept with two
reverse-DNS extensions. See [Migrating to 0.1](./docs/specification/migration-0.1.md)
for the before-and-after mapping of every renamed or removed name.

### Added
- **The patient-context extension**, `com.innovaccer/hmcp-patient-context`
  (revision `"1"`): a server-minted, opaque handle that the host (never the model)
  opens through the `hmcp/patientContext/open`, `.../get` and `.../close` methods,
  bound to the authenticated principal and carried by every patient-scoped tool
  call in request `_meta`. Every check fails closed and every decision is audited.
  Configured with `PatientContextConfig`, installed as `PatientContextExtension`,
  reachable through `HMCPServer.patient_contexts`; a tool reads the bound context
  with `require_patient_context()`. Authorization goes through an `AccessPolicy`
  (`SmartLaunchPolicy` by default), which also enforces the SMART `patient` claim
  as a ceiling.
- **The guardrails extension**, `com.innovaccer/hmcp-guardrails` (revision `"1"`):
  declarative `GuardrailDefinition`s with `pattern`, `prompt`, `deid`, `classifier`,
  `builtin` and `cascade` evaluators and `block` / `redact` / `pseudonymize` / `annotate`
  actions, enforced on `tools/call` arguments and results. Each call receives a list
  of `GuardrailReportEntry` in the result `_meta`, so a guardrail that did not
  run is visible; a client can require guardrails and the server rejects an
  unapplicable one before the tool runs. A `deid` evaluator calls a
  deployer-supplied `DeidentificationModel` whose request and response follow
  a de-identification service's `/process/text` shape (no vendor adapter is shipped). Four
  reference definitions ship with the SDK (`reference_definitions()`); the reference
  `phi-redaction` and `phi-deidentification` use patterns plus a prompt. Configured with
  `GuardrailsConfig`, installed as `GuardrailsExtension`, reachable through
  `HMCPServer.guardrails`.
- **A `classifier` evaluator kind for text classifiers** (such as small language
  models, SLMs): a definition names a deployer-scoped classifier, the deployer
  supplies a `ClassifierProvider`, and `GuardrailsConfig(classifier=...)` wires it
  in. A `classifier` evaluator supports the `block` and `annotate` actions only and
  is not a cascade step. `examples/gravity_guardrails/` adapts Gravity guardrail
  configs to a `ClassifierProvider`, with an offline stand-in for the Gravity
  service.
- **Client APIs for both extensions** on `HMCPClientConnector`:
  `open_patient_context`, `get_patient_context`, `close_patient_context`, and
  `call_tool(patient_context=..., require_guardrails=...)`; the handle is a
  `PatientContextHandle` and a required guardrail that did not report raises
  `GuardrailReportMissing`.
- **A runnable EHR example** (`examples/ehr_patient_context/`): an LLM-free,
  streamable-HTTP server and client that exercise patient context and guardrails
  end to end on synthetic data.
- **A generated JSON Schema of the wire shapes**
  (`schema/hmcp-0.1.schema.json`), its generator (`scripts/generate_schema.py`)
  and a drift plus live-validation test (`tests/hmcp/test_schema.py`).
- **HMCP error codes outside the JSON-RPC reserved range**
  (`PATIENT_CONTEXT_REQUIRED` through `GUARDRAIL_UNAVAILABLE`, `-31001`–`-31006`
  and `-31011`).
- **The PyPI distribution `innovaccer-hmcp`** (import name `hmcp`; an unrelated
  PyPI project named `hmcp` is not this SDK), with `hmcp.__version__`, a `py.typed`
  marker and `from hmcp.server import HMCPServer` as a lazy re-export.
- **`hmcp.client.PatientContextGetResult`** exported from `hmcp.client`, and
  **`HMCPClientConnector` documented as an async context manager** (it connects on
  entry and cleans up on exit).
- **`OAuthClient.parse_authorization_response(..., expected_state=...)`**, which
  raises `InvalidStateError` when the authorization response's `state` is missing or
  does not match the expected value.
- **Two more runnable examples**: `examples/custom_guardrails/` (defining and
  enforcing a custom guardrail) and `examples/quickstart/` (a minimal end-to-end
  walk-through whose README Quick Start blocks are kept in sync by a drift test).
- **Continuous integration and release automation**: the test matrix covers Python
  3.11–3.14 on Linux plus macOS and Windows, and a packaging job builds the wheel,
  checks its metadata, installs it into a fresh environment and runs the suite
  against the installed wheel. A release workflow publishes to PyPI with Trusted
  Publishing (OIDC, no stored token) when a GitHub release is published, and to
  TestPyPI when a pre-release tag (`v*rc*`, `v*.dev*`) is pushed or the workflow is
  run manually. Dependabot keeps the GitHub Actions and
  pip dependencies current.
- **Project documents and templates**: `SECURITY.md` (Innovaccer's
  responsible-disclosure form), `CONTRIBUTING.md`, and issue and pull-request
  templates.
- A **Disclaimer** section in the README: HMCP is provided as is, makes no
  accuracy, completeness, compliance or patient-safety guarantee, and the
  implementer is responsible for validation, compliance and outcomes. The
  specification, SECURITY.md and the example READMEs link to it.

### Changed
- **Upgraded to `mcp>=2.3.0,<2.4`** and MCP spec **2026-07-28** (from `1.29.0` /
  `2025-03-26`). The `[cli]` extra is no longer installed; `mcp[cli]` must be installed
  separately to use the `mcp` command-line tool. `HMCPServer` and `HMCPClientConnector` build
  on the new `MCPServer` and `Client`; transport options moved from construction to
  `run()`, and `connect()` no longer takes transport arguments.
- **HTTP library moved from `httpx` to `httpx2`** (mcp 2.x replaces httpx) in the
  OAuth client and the connector; the `OAuthClient` public API is unchanged.
- **Authentication now audience-binds by default.** When an `AuthSettings` object
  is supplied, HMCP enforces RFC 8707 `validate_token_resource` unless it is
  explicitly disabled, and raises `ValueError` if `resource_server_url` is missing
  (there is nothing to bind to). A server constructed with no authentication logs
  a warning once.
- **`HMCPClientConnector.get_patient_context` returns a `PatientContextGetResult`**
  with snake_case attributes (`context_id`, `status`, `expires_at`, `patient`,
  `encounter`, `purpose_of_use`) instead of a camelCase `dict`.
- **`OAuthClient.register_client` defaults `token_endpoint_auth_method` to
  `client_secret_post`** and accepts only that or `none`; any other value raises
  `ValueError` before a request is made.
- **`HMCPServer.run()` rejects an unsupported transport.** `transport="sse"` keeps
  the HTTP+SSE deprecation message; any other unsupported value raises `ValueError`
  naming the value and the supported transports.
- **An unauthenticated server with patient context configured** answers
  `PATIENT_NOT_AUTHORIZED` (`-31005`) with the message `Request is not
  authenticated`, and logs a warning once at construction.
- **Client-side `require_guardrails` pins** (`id@major`) are checked against the
  report entry's version major.
- **A non-mapping value under an HMCP `_meta` key raises `TypeError`** in
  `call_tool`, before the request is sent.
- **`HMCPClientConnector.connect()` and `cleanup()` are serialized** by a lifecycle
  lock.
- **The reference email pattern is anchored at run starts**, so it matches in linear
  time.
- **The generated JSON Schema carries each model's description.**
- **Links to files in this repository are relative in `README.md`,
  `CONTRIBUTING.md` and `SECURITY.md`.** The PyPI long description rewrites the
  README's links to absolute GitHub URLs at build time (`hatch-fancy-pypi-readme`,
  a new build-time requirement).
- **Reference guardrail titles and descriptions no longer claim coverage**:
  `phi-redaction` and `phi-deidentification` are described as best-effort redaction
  and pseudonymization that do not by themselves de-identify data under HIPAA, and
  `patient-context-consistency` states that it checks only MRN, EMPI and
  `Patient/<id>` references. Their digests change; their versions stay `1.0.0`.

### Removed
- **Bidirectional sampling**: `@server.sampling()`, `HMCPClient` and
  `create_message` are gone, along with the sampling demos and
  `docs/specification/sampling.md`. MCP 2026-07-28 deprecates sampling, and a
  server cannot initiate requests outside the MRTR pattern.
- **HTTP+SSE transport**: `HMCPServer.run(transport="sse")` and `sse_app()` raise
  `ValueError`; the client connector is streamable-HTTP only.
- **The `experimental.hmcp` capability** and `docs/specification/context.md`,
  replaced by the two extensions and `docs/specification/patient-context.md`.
- **Never-raised authentication exceptions**: `InvalidTokenError`,
  `ClientValidationError`, `ScopeError`, `InvalidCodeError`, `InvalidPKCEError`,
  `InvalidRedirectUriError`, `TokenExpiredError` and `RefreshTokenError` are gone.
  `hmcp.shared.auth` exports `OAuthClient`, `AuthenticationError` and
  `InvalidStateError`; catch `AuthenticationError` (or `InvalidStateError`) instead.

### Fixed
- **Malformed guardrails `_meta` in a tool result** no longer turns a successful call
  into an internal error.
- **A canceled `connect()` releases its resources.**
- **The OAuth refresh token is kept when the server does not rotate it.**
- **A non-ASCII `state` no longer raises `TypeError`** during authorization-response
  parsing.

### Security
- **OAuth error messages no longer echo the authorization server's response
  body.** `OAuthClient` failures now surface only the HTTP status and, when the
  body carries a valid RFC 6749 `error` code, that code, and never the raw body,
  which may repeat the submitted `client_secret`.
- **The client connector no longer logs URL credentials or exception text.**
  Connection logs name the server host and port only (not a URL that may embed
  userinfo), and a failed connect logs the exception type name rather than its
  message, which could carry request data.
- **Guardrails now inspect a `CallToolResult`'s `_meta`.** Every key except the
  guardrails report key (which only the extension writes) is inspected; a
  tool-supplied value under that key is replaced or stripped. Previously PHI in
  `_meta` reached the client with a "pass" report.
- **Structured content and `_meta` are normalized with pydantic's serialization
  before inspection**, so tuples, sets and bytes are inspected as they appear on the
  wire; content that cannot be serialized fails closed.
- **The reference email regex no longer has quadratic backtracking.** A ~20 KB
  adversarial result previously blocked the event loop for over a second, and
  `timeout_ms` cannot interrupt synchronous evaluators.
- **Exceptions that replace another no longer chain it** (patient-context mapped
  errors, OAuth errors, and the guardrail de-id, prompt and pattern-validation
  paths), so PHI or secrets are not reachable through `__cause__` or `__context__`.
- **A malformed 2xx OAuth response no longer echoes its body** (and any echoed
  `client_secret`) through a pydantic `ValidationError`.

### Bumped
- Project version `0.0.6` → `0.1.0`.

## [0.0.6] - 2026-09-17

### Removed
- **Deleted the guardrails specification** (`docs/specification/guardrails.md`,
  plus its section and link in `docs/specification/index.md`). It documented a
  NeMo-Guardrails-backed `Guardrail` class, an `enable_guardrails=True` server
  parameter and automatic prompt-injection filtering in the sampling handler —
  none of which exist in the shipped SDK: the implementation was removed in
  `94894c8`, `nemoguardrails` is not a dependency, and the sampling handler
  forwards every message to the developer's callback unfiltered. Documenting a
  default-on protection that never ran was actively misleading for a healthcare
  SDK. Completes the cleanup started in 0.0.5. Content-level controls remain
  out of scope; see the security section of `docs/specification/sampling.md`.

### Fixed
Documentation-accuracy sweep across every tracked document. 0.0.5 removed code
but updated only some of the docs, so the rest kept teaching deleted APIs and
promising protections the SDK does not provide.

- **Deleted OAuth APIs**: removed `set_token()`, `introspect_token()` and
  `validate_client()` (all deleted in 0.0.5) from
  `src/hmcp/shared/auth/oauth_client_README.md` and `README.md`. Also corrected
  `get_client_credentials_token(scope=)` to `scopes=[...]`,
  `start_authorization_code_flow`'s real 3-tuple return, dict-subscripting of
  the `OAuthToken` pydantic model, and a refresh-token example that could not run.
- **Broken imports**: `hmcp.mcpserver.*` and `hmcp.mcpclient.*` do not exist;
  corrected to `hmcp.server.*` and `hmcp.client.*`.
- **Broken install steps** in all three READMEs: `pip install hatch` followed by
  `hatch build` builds a wheel but installs nothing, leaving `import hmcp`
  failing. Now `pip install .`, with the Python 3.11+ requirement documented.
- **Non-existent paths**: `hmcp_demo.py` (real: `hmcp_llm_demo.py`), the
  `emr_patientdata_example` directory, `multi_handoff_agent_demo.py`'s location,
  and a placeholder test path.
- **Overstated security claims**: the demos were documented as using JWT
  authentication but wire none; `sampling.md` stated that all sampling requests
  require OAuth 2.0, when enforcement is opt-in via `auth_server_provider`
  (default `None`); and audit logging, rate limiting, encryption and patient
  context were presented as shipped though none exist in `src/`. Each claim now
  states its real status.
- **Stale API and usage details**: `HMCPServerHelper` (never existed) renamed to
  `HMCPClientConnector`; `tool['name']` corrected to `tool.name` (`list_tools()`
  returns `mcp.types.Tool` objects); SSE was mislabelled as long-polling; and the
  transport list omitted `stdio`, which `HMCPServer.run` accepts and defaults to.

Reported by Syed Anas Mohiuddin, maintainer of mcp-safeguard.

### Bumped
- Project version `0.0.5` → `0.0.6`.

## [0.0.5] - 2026-08-06

### Changed
- **Upgraded the MCP dependency to `mcp[cli]==1.29.0`** (from `1.26.0`). This is
  the latest `1.x`; `mcp 2.0.0` is intentionally avoided because it removes
  `mcp.server.fastmcp`, which the SDK imports.
- **Rewrote the OAuth 2.0 client** (`src/hmcp/shared/auth/oauth_client.py`):
  - HTTP layer moved from `aiohttp` to **`httpx`**.
  - Builds on MCP's shared auth models (`mcp.shared.auth`: `OAuthMetadata`,
    `OAuthToken`, …) instead of hand-rolled request/response dicts.
  - Added authorization-server **metadata discovery** (`discover()` /
    `_ensure_metadata()`).
  - Added **dynamic client registration** (`register_client()`).
  - Added `access_token` / `refresh_token` accessor properties and a single
    internal `_token_request()` / `_require_session()` path for all token calls.
  - Removed the old `create_*_request()` dict builders and the `set_token()`,
    `validate_client()`, and `introspect_token()` methods.
- Raised the **PyJWT** floor to `>=2.10.1` in `requirements.txt` to satisfy
  mcp 1.29.0's `pyjwt` constraint.
- Pinned Black to `target-version = ["py311"]` for reproducible formatting;
  added `pytest-cov` to the test dependencies.
- (tests) Reworked `test_client_connector_streamable_http_with_auth` to drive
  the OAuth client's real async-context-manager lifecycle (mocking
  `httpx.AsyncClient.aclose`) instead of patching instance-level
  `__aenter__` / `__aexit__`.

### Removed
- Dropped the direct dependencies `aiohttp`, `python-jose`, and
  `python-multipart` from `pyproject.toml` — the rewritten OAuth client no
  longer needs them (JWT/JOSE now come transitively via `mcp`).
- **Removed guardrails from the sampling specification**
  (`docs/specification/sampling.md`): dropped the `Guardrail` usage examples,
  the "blocked by guardrails" error response and sequence diagram, and reframed
  the security section. Content-level controls (input validation,
  prompt-injection defence) are now explicitly out of scope for the protocol
  and belong in a model-agnostic layer (the calling agent's moderation pipeline
  or a network gateway). This aligns the spec with the earlier removal of the
  guardrail code.

### Bumped
- Project version `0.0.4` → `0.0.5`.
