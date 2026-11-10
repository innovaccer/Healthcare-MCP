# Gravity guardrails demo

A runnable example of HMCP's `classifier` guardrail applied to tool arguments
and to tool results, backed by Gravity guardrail configs through an example
adapter. By default the example runs offline against an in-process stand-in for
Gravity, so it requires no Gravity deployment and no network access.

Everything uses synthetic data. The example requires no LLM and no network
access when it runs offline.

> **Warning:** `gravity_demo_auth.py` contains a static bearer token committed to
> the repository. It is not a secret and must never be used outside a local demo.
> A real deployment validates a signed token in `verify_token`.
>
> HMCP makes no accuracy, completeness, compliance or patient-safety guarantee, and the implementer is responsible for validation, compliance and outcomes. See the [Disclaimer](../../README.md#disclaimer).

## Overview

HMCP's `classifier` guardrail calls a deployer-supplied classifier provider for
each chunk of inspected text. This example supplies `GravityClassifier`
(`gravity_classifier.py`), an adapter that implements the provider over Gravity
guardrail configs: it posts the inspected text and a `checkType` to Gravity's
`{api_base}/guardrails/{guardrail_id}/configs/{config_id}/invoke` endpoint and
turns the returned verdict into a finding.

The server defines two guardrails over the same Gravity config:

1. **Input stage** (`prompt-injection-input`): inspects the tool arguments
   before the tool body runs. Action: `block`.
2. **Output stage** (`prompt-injection-tool-output`): inspects the tool result
   before it is returned to the host. Action: `block`.

Both definitions are mandatory, so they run on every call they apply to. Both
fail closed: a classifier error blocks the call. `fake_gravity.py` is a
deterministic in-process stand-in for Gravity, used by default so the example
runs offline.

The server also configures authentication and patient context, reusing the
static demo-token pattern from the [EHR patient-context example](../ehr_patient_context/).

## Prerequisites

Python ≥ 3.11 and, from the repository root, `pip install -e .` (or
`pip install innovaccer-hmcp`; see the [examples index](../README.md)).

## Running the example

In the first terminal, start the server:

```sh
python examples/gravity_guardrails/server.py
```

It prints `Gravity mode: offline` and listens on `http://127.0.0.1:8052/mcp`. In
a second terminal, run the client:

```sh
python examples/gravity_guardrails/client.py
```

## Running against Gravity

Set three environment variables to run against a Gravity deployment:

- `GRAVITY_API_BASE_URL`: the Gravity API base, ending in
  `/aibuilder-api/api/v1`, for example
  `https://<gravity-host>/aibuilder-api/api/v1`.
- `GRAVITY_ACCESS_TOKEN`: the bearer token for the Gravity API.
- `GRAVITY_PROMPT_INJECTION_CONFIG`: the guardrail id and config id, as
  `<guardrailId>/<configId>`, taken from the Gravity guardrail catalog.

`GRAVITY_TENANT_ID` is optional; when set, the adapter sends it as the tenant
header. When none of the three required variables is set, the server prints
`Gravity mode: offline` and uses the in-process stand-in. When all three are
set, it prints `Gravity mode: live` and uses the real endpoint. When some but
not all are set, the server raises an error that names the missing variables,
so a partial configuration never falls back to the stand-in silently.

## Stage mapping

> **Important:** Both HMCP stages use Gravity's `input` check type. An HMCP
> `output` stage inspects a tool result, which is input to the host's model. On a
> Gravity prompt-injection config whose output side is set to `allow`, an output
> check returns `allowed` without running the model (measured), so a mapping from
> the HMCP output stage to Gravity's output check would silently disable the
> guardrail for tool results.

## Expected output

The `pc_...` context handle is server-minted and differs on every run. The
client output is:

```text
opened context pc_... for Patient/test-0001
1. search_notes: 2 notes match: note-1 (follow-up visit), note-2 (referral summary from an outside clinic)
1. guardrail report: com.example.gravity-guardrails/prompt-injection-input/input=pass
1. guardrail report: com.example.gravity-guardrails/prompt-injection-tool-output/output=pass
2. search_notes blocked: is_error=True: Blocked by guardrail com.example.gravity-guardrails/prompt-injection-input
2. guardrail report: com.example.gravity-guardrails/prompt-injection-input/input=blocked
3. get_shared_note: Follow-up visit scheduled in two weeks. Continue the current care plan.
3. guardrail report: com.example.gravity-guardrails/prompt-injection-input/input=pass
3. guardrail report: com.example.gravity-guardrails/prompt-injection-tool-output/output=pass
4. get_shared_note blocked: is_error=True: Blocked by guardrail com.example.gravity-guardrails/prompt-injection-tool-output
4. guardrail report: com.example.gravity-guardrails/prompt-injection-input/input=pass
4. guardrail report: com.example.gravity-guardrails/prompt-injection-tool-output/output=blocked
```

## Data handling

> **Warning:** The Gravity service receives the inspected text, and a Gravity
> environment can log it. Use synthetic data only with this example. Before
> sending PHI, confirm the environment's retention and BAA coverage.

## Files

| File | Purpose |
|------|---------|
| `server.py` | Builds the HMCP server, the two classifier guardrails and the synthetic tools. |
| `client.py` | Drives the demo over streamable-http and prints the guardrail report. |
| `gravity_demo_auth.py` | Demo-only bearer-token verifier for synthetic Patient/test-0001. |
| `gravity_classifier.py` | Example `ClassifierProvider` backed by Gravity guardrail configs. |
| `fake_gravity.py` | Offline, deterministic stand-in for Gravity's guardrail invoke endpoint. |
