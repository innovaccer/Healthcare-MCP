# Security policy

HMCP ("Healthcare-MCP", the PyPI distribution **`innovaccer-hmcp`**) is maintained by
Innovaccer.

## Reporting a vulnerability

> **Important:** Do not open a public GitHub issue, discussion or pull request for a
> vulnerability.

Report suspected vulnerabilities through Innovaccer's secure bug report form:

**https://trust-center.innovaccer.com/report-a-bug**

A report includes the following information:

- the affected product, named as **"Healthcare-MCP / innovaccer-hmcp"**, and the
  version (and, where relevant, the affected component);
- the vulnerability type and its impact;
- reproducible steps;
- a sanitized proof of concept; and
- your contact details.

> **Warning:** Never include PHI, PII, credentials or production data in a report, a
> proof of concept or any attachment. Use synthetic data.

## What to expect

Innovaccer's published targets for handling a report are:

- first response within **one business day**; and
- initial triage within **three business days**.

These are targets, not guarantees. Innovaccer asks that reporters allow reasonable
time for investigation and remediation before any public disclosure.

General security questions that are **not** vulnerability reports can be sent to
Innovaccer Enterprise Security at **enterprise-security@innovaccer.com**.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Security model

HMCP is a library, not a hardened service. The following boundaries apply:

- **In-memory stores only:** Patient-context handles and guardrail state live in
  process memory; there is no persistence and no encryption at rest.
- **No rate limiting:** The library does not throttle requests.
- **Guardrails are defense in depth, not a guarantee:** They can miss PHI, and a
  `pass` verdict does not mean content is free of PHI. Pattern evaluators run
  synchronously on the request path and cannot be interrupted by `timeout_ms`, so a
  pathological (non-linear) regular expression supplied by a deployer can block the
  event loop. Deployer-written regexes must be linear in the input size.
- **TLS and real authentication are the deployer's responsibility:** Run the server
  behind TLS and supply a real `TokenVerifier`; the demo verifiers in the examples
  accept static tokens and are not for production.

Using HMCP does not by itself make a deployment HIPAA-compliant. HMCP makes no
accuracy, completeness, compliance or patient-safety guarantee, and the implementer
is responsible for validation, compliance and outcomes. See the
[Disclaimer](README.md#disclaimer) in the README.
