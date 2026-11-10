# Contributing to Healthcare Model Context Protocol (HMCP)

This document describes how to set up the project, run the checks, and open a pull
request.

## Prerequisites

- Python 3.11 or newer (the CI matrix covers 3.11–3.14).
- Git.
- **Synthetic data only:** All development, testing and demos use synthetic data.
  Real patient data is never used.

## Development setup

```bash
git clone https://github.com/innovaccer/Healthcare-MCP.git
cd Healthcare-MCP
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

`pytest.ini` points at `tests/` and enables coverage by default.

## Formatting

Formatting is enforced in CI with black:

```bash
black --target-version py311 src tests examples scripts
```

Run `black --check --target-version py311 src tests examples scripts` before pushing;
CI fails on any diff.

## Regenerating the wire schema

HMCP ships a JSON Schema for its wire shapes, generated from the Pydantic models in
`src/`. Pydantic model docstrings become the schema `description`s. After changing any
wire model, regenerate the schema and commit the result:

```bash
python scripts/generate_schema.py
```

CI runs a drift test that fails if `schema/hmcp-0.1.schema.json` does not match the
generator output.

## Contribution rules

These rules are mandatory; a pull request that violates them will be returned for
changes.

- **Synthetic data only:** Never commit or attach real patient data (PHI/PII),
  credentials, tokens or production data anywhere, including code, tests, fixtures,
  issues, pull requests, commit messages or logs.
- **Fail closed:** Security-relevant checks (patient-context binding, guardrails,
  authentication) must deny by default when a precondition is missing or a check
  errors.
- **Never leak sensitive data:** Do not put patient data, tokens or response bodies in
  log lines or exception messages. When one exception replaces another, the original
  is not chained: raise the new exception *outside* the `except` block so that both
  `__cause__` and `__context__` are `None`.
- **Tests for behavior changes:** Every behavior change requires a test that fails
  without the change.
- **Documentation updates:** Update the docs and the `CHANGELOG.md` `[Unreleased]`
  section for user-visible changes.
- **MIT license:** By contributing, a contributor agrees that their contributions are
  licensed under the MIT License (see
  [LICENSE](LICENSE)).

## Pull request process

1. Fork the repository and create a branch off `main`.
2. Make your change, add tests, and run `black` and `pytest`.
3. Regenerate the schema if you touched a wire model.
4. Update `CHANGELOG.md` and any affected docs.
5. Open a pull request against `main` and fill in the pull request template.
6. A maintainer reviews the pull request; review comments are addressed on the same
   branch.

## Reporting security issues

Do **not** open a public issue, discussion or pull request for a vulnerability. Follow
the responsible-disclosure process in
[SECURITY.md](SECURITY.md).
