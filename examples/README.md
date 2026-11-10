# HMCP examples

Runnable examples for HMCP 0.1.0. Every example uses synthetic data only.

> **Warning:** The examples are demo code, not production code. They use synthetic data, static demo tokens and intentionally narrow detection patterns. They show how the API works; they are not validated PHI detection and are not for use with real patient data.
>
> HMCP makes no accuracy, completeness, compliance or patient-safety guarantee, and the implementer is responsible for validation, compliance and outcomes. See the [Disclaimer](../README.md#disclaimer).

## Prerequisites

- Python ≥ 3.11.
- Install HMCP from PyPI:

  ```sh
  pip install innovaccer-hmcp
  ```

  Alternatively, install from a clone of this repository:

  ```sh
  pip install -e .
  ```

## The examples

| Example | What it shows |
|---------|---------------|
| [`quickstart/`](quickstart/README.md) | A minimal end-to-end server and client that opens a patient context and calls one patient-scoped tool. |
| [`ehr_patient_context/`](ehr_patient_context/README.md) | A runnable, LLM-free demo of the two HMCP 0.1.0 extensions: **patient context** (a server-minted handle the host opens and every patient-scoped tool must carry) and **guardrails** (declarative, best-effort checks enforced server-side, with a report in the result's `_meta`). |
| [`custom_guardrails/`](custom_guardrails/README.md) | Writing custom pattern, prompt and de-identification guardrails, with in-process stand-ins for the LLM and de-identification providers. |
| [`gravity_guardrails/`](gravity_guardrails/README.md) | Classifier (SLM) guardrails backed by Gravity guardrail configs, with an offline in-process stand-in for the Gravity service, applied to tool arguments and tool results to block prompt injection. |

## Running an example

Each example has a server and a client. Start the server in the first terminal,
then run the client in a second:

```sh
python examples/quickstart/server.py     # first terminal
python examples/quickstart/client.py     # second terminal
```

## Removed 0.0.x demos

The previous examples (an LLM demo, a multi-agent handoff demo and a set of
EMR patient-data demos) were all built on the **sampling** feature. HMCP
0.1.0 removed bidirectional sampling, so those demos were deleted rather than
patched. See the migration guide at
[`docs/specification/migration-0.1.md`](../docs/specification/migration-0.1.md)
for what replaced sampling.
