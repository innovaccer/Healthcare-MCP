# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
