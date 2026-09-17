# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
