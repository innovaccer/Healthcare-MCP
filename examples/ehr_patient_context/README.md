# EHR patient context + guardrails demo

A runnable example of the two HMCP 0.1.0 extensions over streamable-http. The
example requires no LLM.

HMCP 0.1.0 removed the bidirectional **sampling** feature, so a tool can no
longer ask the host to run an LLM. In its place this demo shows:

- **Patient context:** the host (the client application, never the model)
  opens an opaque, server-minted handle for a single patient. The handle is
  bound to the caller's token, and every patient-scoped tool call must carry
  it. Tools read it with `require_patient_context()`.
- **Guardrails:** declarative definitions the server enforces on tool
  arguments and results. Each call gets a report attached to the result's
  `_meta`, so the client can see which guardrails ran and what they decided.

Everything uses synthetic data. The example requires no network access and no
LLM.

> **Warning:** `demo_auth.py` contains two static bearer tokens committed to
> the repository. They are not secrets and must never be used outside a local
> demo. A real deployment validates a signed token in `verify_token`.
>
> HMCP makes no accuracy, completeness, compliance or patient-safety guarantee, and the implementer is responsible for validation, compliance and outcomes. See the [Disclaimer](../../README.md#disclaimer).

## Running the example

Prerequisites: Python ≥ 3.11 and, from the repository root,
`pip install innovaccer-hmcp` (or `pip install -e .` from a clone); see the
[examples index](../README.md).

In the first terminal, start the server:

```sh
python examples/ehr_patient_context/ehr_server.py
```

It listens on `http://127.0.0.1:8050/mcp`. In a second terminal, run the
client:

```sh
python examples/ehr_patient_context/ehr_client.py
```

## Output reference

| # | Line | What it shows |
|---|------|---------------|
| 1 | `opened context pc_... for Patient/test-0001` | The host opened a patient context. The `pc_...` id is a server-minted opaque handle, bound to the bearer token's principal. |
| 2 | `get_medications: ...` | A patient-scoped tool call with the handle attached under the patient-context extension key. |
| 3 | `get_demographics: { ... }` | A PHI-bearing result, returned as a **keyed JSON object**. The pattern guardrail replaced the SSN, date of birth and MRN with `[REDACTED:...]` placeholders, and also the name because the fields are keyed (`name`, `dob`, ...). |
| 3 | `guardrail report: id/stage=verdict ...` | The report from the result's `_meta`: which guardrail ran, at which stage (`input`/`output`) and its verdict. A guardrail that runs at both stages appears twice, once per stage. `require_guardrails=[...]` made the client fail if this report were missing. |
| 4 | `add_note blocked: Blocked by guardrail ...` | Writing back text that already contains a `[REDACTED:...]` placeholder is refused at the input stage. |
| 5 | `lookup_with_bug blocked: Blocked by guardrail ...` | A tool with an intentional bug returned the *other* patient's record; the consistency guardrail caught the mismatched identifier and blocked the result. |
| 6 | `population_count denied: MCPError -31005 ...` | A population-scoped tool called with a single-patient token is denied by `SmartLaunchPolicy`. `-31005` is `PATIENT_NOT_AUTHORIZED`. |
| 7 | `open Patient/test-0002 denied: MCPError -31005 ...` | The token is scoped to `test-0001`, so it cannot open a context for `test-0002`. |
| 8 | `get_medications without context: MCPError -31001 ...` | A patient-scoped tool called with no context handle. `-31001` is `PATIENT_CONTEXT_REQUIRED`. |
| 9 | `closed context` | The host closed the patient context. Its handle is now in a terminal state and cannot be reused. |
| 9 | `get_medications after close: MCPError -31002 ...` | The context was closed, so reusing its handle fails. `-31002` is `PATIENT_CONTEXT_INVALID`. |

The example distinguishes two shapes of failure:

- **Blocked by a guardrail** is a normal tool result with `is_error=True` and a
  text saying which guardrail blocked it (lines 4 and 5).
- **An HMCP access error** is a JSON-RPC error, surfaced on the client as an
  `MCPError` with an HMCP code (lines 6–9).

## Guardrails in this example

`ehr_server.py` installs three definitions, all **mandatory** (they run on
every call they apply to, whether or not the client asks):

- `patient_context_consistency()`: blocks a call whose arguments or result
  contain an MRN, EMPI or `Patient/<id>` reference that does not belong to the
  active patient context (line 5).
- `redaction_placeholder_writeback()`: blocks arguments that already contain
  a `[REDACTED:...]` placeholder, to stop redacted text being written back
  (line 4).
- `demo_phi_redaction()`: an **example-local**, pattern-only redaction
  definition (`com.example.ehr-demo/phi-patterns`) over HMCP's built-in
  `PHI_IDENTIFIER_TYPES` (line 3). It is local to this demo so that no LLM is
  needed.

### Limitations of the pattern-only guardrail

`demo_phi_redaction()` is intentionally narrow, and its limitations should be
understood before conclusions are drawn from the demo:

- It **does** catch **structured identifiers by shape** (SSN, phone, email,
  dates, IP addresses, URLs, and `MRN`/`EMPI` when labeled) and **keyed
  fields** whose JSON key names are PHI (`name`, `dob`, `address`, ...).
  That is why `get_demographics()` returns a dict: with the fields keyed, the
  name is caught by the `name` key.
- It **does not** catch **names or addresses in free text**. A bare
  `"Testpatient Alpha"` inside a paragraph of prose has no pattern and no key,
  so a pattern-only guardrail would leave it untouched. A pattern guardrail is
  not a name/address detector.

Finding names and addresses in free text is the purpose of the prompt step of
the `phi_redaction()` and `phi_deidentification()` definitions (both
`cascade(pattern, prompt)`); see the next section. A de-identification
evaluator (a `deid` evaluator) appears only in a custom definition backed by a
user-supplied `DeidentificationModel`. The demo's clean step-3 output is not
evidence that a pattern-only guardrail de-identifies free text.

## Real LLM integration

The demo is LLM-free by design, but the same server can use the richer
reference guardrails. To add LLM-based PHI detection, pass a provider to
`GuardrailsConfig` and use the reference `phi_redaction()` definition:

`LLMProvider` is a **Protocol** with a single method:
`async def complete(self, *, system: str, user: str, max_tokens: int) -> str`.
Any object that has that method satisfies it, so subclassing is optional.

```python
from hmcp.server.guardrails import GuardrailsConfig
from hmcp.shared.guardrails import phi_deidentification, phi_redaction
from hmcp.shared.guardrails.evaluators.prompt import LLMProvider


class MyLLM(LLMProvider):
    async def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        # Call your model here and return its raw response.
        ...


guardrails = GuardrailsConfig(
    definitions=[phi_redaction()],
    llm=MyLLM(),
)
```

`phi_deidentification()` uses the same `llm`: its evaluator is
`cascade(pattern, prompt)`, so it needs no separate provider:

```python
guardrails = GuardrailsConfig(
    definitions=[phi_redaction(), phi_deidentification()],
    llm=MyLLM(),
)
```

A custom definition with a `deid` evaluator can instead be backed by a
[`DeidentificationModel`](../../docs/specification/guardrails.md#deidentificationmodel).

The LLM provider and any de-identification model process PHI, so they must be
covered by a BAA with zero data retention; see the `data_handling` notes on the
reference definitions.
