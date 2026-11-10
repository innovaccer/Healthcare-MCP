# Custom guardrails demo

A runnable example of the three *extension points* of the HMCP guardrails
extension. The example requires no LLM and no network access. HMCP ships a few
reference guardrails; this example shows how an adopter writes their own, using
only deterministic in-process stand-ins and synthetic data.

Everything uses synthetic data. The example requires no network access, no model
and no external service.

> **Warning:** `demo_auth.py` contains a static bearer token committed to the
> repository. It is not a secret and must never be used outside a local demo. A
> real deployment validates a signed token in `verify_token`.
>
> HMCP makes no accuracy, completeness, compliance or patient-safety guarantee, and the implementer is responsible for validation, compliance and outcomes. See the [Disclaimer](../../README.md#disclaimer).

## Overview

Each guardrail is a `GuardrailDefinition` with a different evaluator spec:

1. **Pattern** (`PatternEvaluatorSpec`): `member_id_pattern()` redacts a
   made-up member-ID format (`MBR-` followed by eight digits). It declares the
   format as an `IdentifierType` with both a `text_patterns` regex and a
   `json_keys` entry for `member_id`. Because the tool returns a **keyed JSON
   object**, both paths are exercised: the ID is redacted as the value under the
   `member_id` key *and* inside the free-text `summary`. Action: `redact`.
2. **Prompt** (`PromptEvaluatorSpec`): `prompt_injection_block()` blocks tool
   arguments that contain a fixed injection phrase (`ignore previous
   instructions`). The prompt evaluator calls an `LLMProvider`;
   `FakePromptModel` is a deterministic in-process stand-in that returns the
   verdict JSON the evaluator expects. Action: `block`.
3. **De-identification** (`DeidEvaluatorSpec`): `name_deid()` pseudonymizes a
   small fixed list of synthetic names. The deid evaluator calls a
   `DeidentificationModel`; `FakeDeidModel` is a deterministic in-process
   stand-in that reports character spans for those names. Action:
   `pseudonymize`.

Guardrails that pseudonymize need an active patient context, so the server also
configures authentication and patient context, reusing the static demo-token
pattern from the [EHR patient-context example](../ehr_patient_context/).

## Running the example

Prerequisites: Python ≥ 3.11 and, from the repository root, `pip install -e .`
(or `pip install innovaccer-hmcp`; see the [examples index](../README.md)).

In the first terminal, start the server:

```sh
python examples/custom_guardrails/server.py
```

It listens on `http://127.0.0.1:8051/mcp`. In a second terminal, run the client:

```sh
python examples/custom_guardrails/client.py
```

## Output reference

The client output is:

```text
1. opened context pc_ks3iSwnDokY0FCHbn8N96Sm0EkbcPQLN7aGsT2zpOrY for Patient/test-0001
2. get_member_record: {
  "member_id": "[REDACTED:OTHER_ID]",
  "name": "[NAME_1]",
  "summary": "Member [REDACTED:OTHER_ID] is enrolled in the demo plan."
}
2. guardrail report: com.example.custom-guardrails/prompt-injection-block/input=pass
2. guardrail report: com.example.custom-guardrails/member-id-pattern/output=redacted
2. guardrail report: com.example.custom-guardrails/name-deid/output=pseudonymized
3. submit_note blocked: is_error=True: Blocked by guardrail com.example.custom-guardrails/prompt-injection-block
3. guardrail report: com.example.custom-guardrails/prompt-injection-block/input=blocked
```

| # | Line | What it shows |
|---|------|---------------|
| 1 | `1. opened context pc_... for Patient/test-0001` | The host (the client, never the model) opened a patient context. The `pc_...` id is a server-minted opaque handle bound to the bearer token's principal. It is random per run. |
| 2 | `2. get_member_record: { ... }` | A patient-scoped tool returning a keyed JSON object. The pattern guardrail replaced the member ID with `[REDACTED:OTHER_ID]` in **both** the `member_id` field (via `json_keys`) and the free-text `summary` (via `text_patterns`). The deid guardrail replaced the synthetic name with the marker `[NAME_1]`. |
| 2 | `.../prompt-injection-block/input=pass` | The prompt guardrail ran on the call's arguments at the input stage and found nothing. |
| 2 | `.../member-id-pattern/output=redacted` | The pattern guardrail found the member ID in the result at the output stage and redacted it. |
| 2 | `.../name-deid/output=pseudonymized` | The deid guardrail found the synthetic name in the result at the output stage and pseudonymized it. |
| 3 | `3. submit_note blocked: is_error=True: Blocked by guardrail .../prompt-injection-block` | The prompt guardrail flagged the injection phrase in the arguments at the input stage, so the call was **blocked**: the server returned an error result (`is_error=True`) naming the guardrail, and the tool body never ran. |
| 3 | `.../prompt-injection-block/input=blocked` | The report entry for the blocking guardrail: id, stage `input`, verdict `blocked`. |

The guardrail report entries are written to the result's `_meta` under the
guardrails extension key. `require_guardrails=[...]` in the client made the
connector fail if any of the named guardrails did not report.

## Real model integration

`FakePromptModel` and `FakeDeidModel` are in-process stand-ins for the two provider
protocols; each docstring marks where a real client goes. The contracts are
documented in [docs/specification/guardrails.md](../../docs/specification/guardrails.md):

- **`LLMProvider`:** one method,
  `async def complete(self, *, system: str, user: str, max_tokens: int) -> str`.
  The prompt evaluator renders its `system`/`user` templates (with the inspected
  content embedded and a fresh nonce) and expects the reply to be the verdict
  JSON `{"findings": [{"text": "<exact substring>", "category": "<category>"}]}`.
  Any object with that method satisfies the protocol; subclassing is optional.
- **`DeidentificationModel`:** one method,
  `async def process_text(self, request: DeidRequest) -> Sequence[DeidTextResult | Mapping[str, Any]]`.
  The request mirrors a de-identification service's `/process/text` API: send
  `request.model_dump(mode="json")` and return one result per input text, in
  order, each carrying labeled entities with character spans. Never log or raise
  the response body.

Pass the real objects to `GuardrailsConfig(definitions=[...], mandatory=[...],
llm=..., deid=...)` exactly as `build_server()` does. The prompt provider and the
de-identification model process PHI, so they must be covered by a BAA with zero
data retention.
