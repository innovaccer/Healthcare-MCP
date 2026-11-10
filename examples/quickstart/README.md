# Quickstart

A minimal end-to-end HMCP server and client over streamable-http, using
synthetic data. The example requires no LLM and no network access beyond
localhost.

These two files are exactly the code blocks from the top-level README
("### A minimal server" and "### A minimal client"); a test keeps them
synchronized, so the README is always runnable.

## Running the example

Prerequisites: Python ≥ 3.11 and, from the repository root,
`pip install innovaccer-hmcp` (or `pip install -e .` from a clone).

In the first terminal, start the server:

```sh
python examples/quickstart/server.py
```

It listens on `http://127.0.0.1:8050/mcp`. In a second terminal, run the client:

```sh
python examples/quickstart/client.py
```

## Expected output

The client opens a patient context, calls the patient-scoped `get_observations`
tool, prints the tool's text content, then one line per guardrail report entry
(`id/stage=verdict`):

```text
{
  "patient": "Patient/test-0001"
}
guardrail report: com.innovaccer.hmcp.guardrails/patient-context-consistency/input=pass
guardrail report: com.innovaccer.hmcp.guardrails/patient-context-consistency/output=pass
```

The `demo-token-test-0001` token is demo-only. Its `patient` claim is what the
default `SmartLaunchPolicy` checks before opening the context; a production
deployment supplies real tokens from a SMART authorization server.

## Other examples

- [ehr_patient_context](../ehr_patient_context/README.md): patient context and
  guardrails end to end, with a report in each result's `_meta`.
- [custom_guardrails](../custom_guardrails/README.md): writing custom pattern,
  prompt and de-identification guardrails, with in-process stand-ins for the LLM
  and de-identification providers.
