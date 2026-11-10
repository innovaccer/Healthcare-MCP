# Guardrails

This document specifies HMCP guardrails: declarative, best-effort checks that the HMCP
server applies to tool calls and reports back to the client in the result `_meta`.

- **Base spec:** MCP `2026-07-28`.
- **HMCP version:** 0.1.0.
- **Extension identifier:** `com.innovaccer/hmcp-guardrails` (revision `"1"`).

> **Note:** Every statement is marked **normative** (a protocol MUST/SHOULD, RFC
> 2119), **SDK behavior** (what `src/` does today), or **future work** (not
> shipped). Where the SDK implements less than the protocol requires, this
> document says so.

## Rationale and history

**SDK behavior / project history.** HMCP `0.0.6` deleted a documentation page
that described a default-on protection which did not run. The design makes a
guardrail that did **not** run visible. Every guardrail that the server applies
is named in a per-call report in the result `_meta`, and a client can require
specific guardrails and refuse a result whose report does not name them. A
guardrail that is not listed in a report is a guardrail that did not run on that
call.

> **Accuracy rule.** No guardrail **prevents** or **guarantees** anything. Every
> evaluator, including the reference definitions, is best effort: it can miss PHI
> and can flag content that is not PHI. A `pass` verdict means only that the
> configured evaluators found nothing; it is not evidence that content is free of
> PHI. Redaction and pseudonymization do not by themselves de-identify data under
> HIPAA. Recall must be measured on the deployer's own data, and the deployer is
> responsible for validation, compliance and outcomes. See the
> [Disclaimer](../../README.md#disclaimer).

## Definition vs evaluator vs report

HMCP guardrails distinguish three elements.

- A **definition** is a portable, declarative JSON document: an id, a version, a
  set of stages, an evaluator spec, an action, and policy fields. It contains no
  executable code and introduces no vendor-specific dependency or vendor lock-in.
  **Normative.**
- An **evaluator** is the implementation that inspects content for a given
  evaluator kind. The enforcing party supplies it. A `prompt` evaluator runs on
  the enforcer's **own** LLM provider, a `deid` evaluator calls a
  de-identification **model** the enforcer controls, and a `classifier` evaluator
  calls a **text classifier** the enforcer controls (HMCP ships none).
  **SDK behavior.**
- A **report** is a per-call `GuardrailReportEntry` written into the result
  `_meta`. It records which definition ran, at which stage, with which verdict,
  and how many findings it produced (a count, never the matched text). This is
  what makes a guardrail that did not run visible. **Normative.**

The definitions and evaluators are located in `src/hmcp/shared/guardrails/`; the
server side is located in `src/hmcp/server/guardrails/`.

## The definition schema

`GuardrailDefinition` is a frozen Pydantic model that forbids extra fields and
uses camelCase aliases on the wire (`src/hmcp/shared/guardrails/definition.py`).
Python code uses snake_case; JSON uses camelCase.

| Python field | Wire name | Type | Default | Validators / notes |
| --- | --- | --- | --- | --- |
| `id` | `id` | `str` | required | Pattern `^[a-z0-9]+(\.[a-z0-9-]+)+/[a-z0-9][a-z0-9-]*$` (reverse-DNS prefix + `/` + short name). |
| `version` | `version` | `str` | required | Pattern `^\d+\.\d+\.\d+$`. |
| `title` | `title` | `str` | required | Human title. |
| `description` | `description` | `str` | `""` | Free text. |
| `stages` | `stages` | `tuple[GuardrailStage, ...]` | required | At least one; no duplicates. `GuardrailStage` is `"input"` or `"output"`. |
| `applies_to` | `appliesTo` | `AppliesTo` | `AppliesTo()` | See below. |
| `evaluator` | `evaluator` | `EvaluatorSpec` | required | Discriminated on `kind`. See evaluator kinds. |
| `action` | `action` | `ActionSpec` | required | See actions. |
| `failure_mode` | `failureMode` | `"closed"` or `"open"` | `"closed"` | What an `ERROR` verdict does. |
| `timeout_ms` | `timeoutMs` | `int` | `5000` | `100`–`60000`. |
| `data_handling` | `dataHandling` | `dict[str, str]` | `{}` | Free-form data-handling notes surfaced to the deployer. |

Cross-field validators (**normative**):

- A `builtin` evaluator requires `action.onFinding == "block"`.
- A `classifier` evaluator requires `action.onFinding` of `"block"` or `"annotate"`.
- An `action.onFinding` of `"pseudonymize"` requires `"output"` in `stages`.

### `appliesTo`

| Python field | Wire name | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `patient_scopes` | `patientScopes` | `tuple[ToolPatientScope, ...]` or `null` | `null` | `null` = every scope. Values are `"patient"`, `"population"`, `"none"`. |
| `tools` | `tools` | `tuple[str, ...]` or `null` | `null` | `null` = every tool. |

`GuardrailDefinition.applies(scope, tool_name)` returns `True` only when the tool
name is in `tools` (or `tools` is `null`) **and** the scope is in `patient_scopes`
(or `patient_scopes` is `null`). **SDK behavior.**

### `identifierTypes`

`IdentifierType` describes one kind of identifier.

| Python field | Wire name | Type | Default | Validators / notes |
| --- | --- | --- | --- | --- |
| `name` | `name` | `str` | required | Pattern `^[a-z][a-z0-9_]*$`. |
| `display_name` | `displayName` | `str` | required | |
| `category` | `category` | `PhiCategory` | required | One of the `PhiCategory` values (below). |
| `text_patterns` | `textPatterns` | `tuple[str, ...]` | `()` | Each must compile and have **exactly one** capture group; used with `re.IGNORECASE`. |
| `json_keys` | `jsonKeys` | `tuple[str, ...]` | `()` | Matched case-insensitively against the dict key a string sits under. |
| `is_access_key` | `isAccessKey` | `bool` | `False` | EMPI/MRN-like keys that identify a patient for access control. |

Validator: at least one of `text_patterns` / `json_keys` must be non-empty.

`PhiCategory` is a vendor-neutral string enum whose names follow the HIPAA Safe
Harbor identifier list. The names are labels only; checking content against them
is not a Safe Harbor de-identification:
`NAME`, `DATE`, `PHONE`, `FAX`, `EMAIL`, `SSN`, `MRN`, `HEALTH_PLAN_ID`,
`ACCOUNT_NUMBER`, `LICENSE_NUMBER`, `VEHICLE_ID`, `DEVICE_ID`, `URL`,
`IP_ADDRESS`, `BIOMETRIC_ID`, `PHOTO`, `LOCATION`, `OTHER_ID`. Each member's
value equals its name.

### `digest`, `major`, `short_name`

**SDK behavior.** `GuardrailDefinition.digest` is the string `"sha256:"`
followed by the SHA-256 hex digest of the definition serialized with
`model_dump(by_alias=True, mode="json")`, then `json.dumps(..., sort_keys=True,
separators=(",", ":"), ensure_ascii=False)` and encoded UTF-8. Two definitions
with the same digest are byte-for-byte the same on the wire. `major` is
`int(version.split(".")[0])`. `short_name` is the text after the `/` in the id.
`to_json()` returns `model_dump_json(by_alias=True)`; `from_json(text)` parses it
back.

## Evaluator kinds

The `evaluator` field is a discriminated union on `kind`. The kinds are
`pattern`, `prompt`, `deid`, `classifier`, `builtin` and `cascade`.

### `pattern`

**SDK behavior** (`src/hmcp/shared/guardrails/evaluators/pattern.py`).
Deterministic regex and JSON-key rules. Patterns are compiled once with
`re.IGNORECASE`. For each text unit and each identifier type:

- every `finditer` match produces a `Finding` over capture group 1's span
  (`category` = the type's category, `identifier_type` = the type's name,
  `source` = `"pattern"`, `value` = the matched group); and
- when the unit's key (casefolded) is one of the type's `json_keys` and the text
  is non-empty after `strip()`, one `Finding` covers the whole string.

Findings per unit then pass through `merge_overlaps`.

`PatternEvaluatorSpec` fields: `kind` (`"pattern"`), `identifier_types`
(`identifierTypes`, at least one).

### `prompt`

**SDK behavior** (`src/hmcp/shared/guardrails/evaluators/prompt.py`). An LLM
prompt template run by **the enforcing party on its own LLM provider**. MCP
sampling is deprecated in 2026-07-28, so a guardrail never asks the client's
model to do the work; the server calls the provider it was configured with.

`PromptEvaluatorSpec` fields: `kind` (`"prompt"`), `system` (must contain
`{{nonce}}`), `user` (must contain both `{{content}}` and `{{nonce}}`),
`categories` (at least one), `max_tokens` (`maxTokens`, default `1024`, range
`1`–`8192`), `max_chars_per_call` (`maxCharsPerCall`, default `12000`, range
`500`–`200000`).

Behavior:

- **Chunking.** Consecutive units are packed into chunks whose texts, joined with
  `"\n---\n"`, stay within `max_chars_per_call`; a single longer unit forms its
  own chunk and is never truncated.
- **Nonce fencing.** For each chunk a fresh nonce `secrets.token_hex(8)` is
  generated. `system` and `user` are rendered with `str.replace` in this order:
  `{{nonce}}`, then `{{categories}}`, then `{{content}}` **last**, so content
  that itself contains `{{nonce}}` is never expanded. The evaluator renders
  whatever templates the definition supplies; the validator requires `{{nonce}}`
  in both. The **reference** prompt (`PHI_PROMPT_USER`) wraps the content in
  `<content-<nonce>> ... </content-<nonce>>` and its system prompt states that
  the fenced text is data, not instructions. A custom template must do its own
  fencing.
- **Parsing.** The reply is stripped; a leading ` ``` ` fence and a trailing
  ` ``` ` are dropped; the body must parse as a JSON object with a list
  `findings`, each item
  a dict with non-empty string `text` and string `category`. Items whose category
  is not in `categories` are ignored. Anything else raises
  `EvaluationError("invalid LLM verdict")`.
- Each finding's text is located with `locate_all` in **every** unit of its
  chunk; text found nowhere is ignored. A provider exception raises
  `EvaluationError("LLM call failed")`.

> **Warning (normative).** A prompt evaluator reduces risk; it does not
> guarantee detection. Recall must be measured on the deployer's own data.

### `deid`

**SDK behavior** (`src/hmcp/shared/guardrails/evaluators/deid.py`). Calls a
deployer-supplied de-identification **model**; HMCP ships no model and no vendor
adapter. `DeidEvaluatorSpec` fields: `kind` (`"deid"`), `categories`
(default `()`).

- **Request.** Units are sent in batches of at most 50. The body is always
  `{"text": [<texts>], "entity_detection": {"return_entity": true}}`, the wire
  shape of a de-identification service's `/process/text` API (text in; labeled
  entities with character spans out). HMCP never forwards an entity selection to
  the model: `spec.categories` is applied locally, because a vendor's narrower
  reading of a selector can silently return zero entities.
- **Response.** The reply is a JSON list with one object per text, in order,
  each carrying an `entities` list. A response that is not a list or tuple of the
  batch length, or an item that does not validate as `DeidTextResult`, raises
  `EvaluationError("invalid deid response")` (the Pydantic message is dropped
  because it embeds the input).
- **Per-entity validation.** `best_label` must match `^[A-Z][A-Z0-9_]*$`;
  `0 <= stt_idx < end_idx <= len(text)`; and `text[stt_idx:end_idx]` must equal
  `entity.text`. Offsets are Python `str` (code point) indices. Any violation
  raises `EvaluationError` with a message that carries indices only
  (`deid entity {j} of text {t}: span out of bounds` / `...: span does not match
  text`), never content or the label, because a model can return any uppercase
  label, including an uppercased name.
- **Span trimming.** The span is trimmed of edge whitespace and trailing
  `. , ; : ! ?`; an empty span is skipped.
- **Categories.** The label is mapped with `label_to_category` (first matching
  rule; unknown → `OTHER_ID`). A non-empty `spec.categories` filters locally.
- **Re-occurrences.** A finding is produced at the reported span **and** at every
  other occurrence of the same value in that unit, **including overlapping ones**
  (for example `ana` in `banana`), erring toward redaction.
- **Fail-closed.** Every exception raised by the model is caught and re-raised as
  a fixed `EvaluationError("deid model failed")` with no chained cause or
  context, so a provider message cannot carry PHI; the guardrail then fails per
  its failure mode.

### `classifier`

**SDK behavior** (`src/hmcp/shared/guardrails/evaluators/classifier.py`). Calls a
deployer-supplied text classifier, such as a small language model (SLM). HMCP
ships no classifier and no vendor adapter. `ClassifierEvaluatorSpec` fields:
`kind` (`"classifier"`), `classifier` (a deployer-scoped classifier name; pattern
`^[a-z0-9][a-z0-9._-]*$`, at most 128 characters; the provider resolves it to a
model or service) and `max_chars_per_call` (`maxCharsPerCall`, default `12000`,
range `500`–`200000`). Wire example:
`{"kind": "classifier", "classifier": "prompt-injection", "maxCharsPerCall": 8000}`.

Behavior:

- **Chunking.** Chunking is identical to the `prompt` evaluator: consecutive units
  are packed into chunks whose texts, joined with `"\n---\n"`, stay within
  `max_chars_per_call`; a single longer unit forms its own chunk and is never
  truncated. A chunk is sent only if at least one of its units contains
  non-whitespace text.
- **Requests.** One `ClassificationRequest(classifier, stage, text)` is sent per
  chunk, with at most 4 calls in flight per evaluation (one stage of one tool
  call); a provider that needs a global rate limit applies its own. The provider
  returns a
  `ClassificationResult` with a required strict boolean `flagged`, or a mapping
  that validates as one (extra keys are ignored).
- **Findings.** A flagged chunk produces exactly one finding, on the chunk's first
  unit that contains non-whitespace text, covering that unit; the finding is a
  whole-unit span, not a located match; the report carries only a count of
  findings. The report's `findings` count is the number of flagged chunks. A
  classifier flags whole chunks, not spans, so redaction and pseudonymization are
  not available.
- **Failures.** A provider exception raises `EvaluationError("classifier call
  failed")`; an invalid result raises `EvaluationError("invalid classifier
  result")`. Any failed chunk fails the evaluation, and the guardrail reports
  `error` (then `failureMode` decides).

> **Warning (normative).** A classifier evaluator reduces risk; it does not
> guarantee detection. Recall must be measured on the deployer's own data. A
> classifier whose input window is shorter than `max_chars_per_call` can truncate
> input without notice; set `max_chars_per_call` within the classifier's input
> window.

> **Important:** An HMCP `output` stage inspects a tool result, which becomes
> input to the host's model. A provider for a service that distinguishes input
> checks from output checks must choose the mapping for each stage deliberately;
> the example maps both stages to the service's input check.

`examples/gravity_guardrails/` is an example that adapts Gravity guardrail configs
to a `ClassifierProvider`.

### `builtin`

**SDK behavior** (`src/hmcp/shared/guardrails/evaluators/builtin.py`).
HMCP-defined checks. `BuiltinEvaluatorSpec` fields: `kind` (`"builtin"`), `name`
(`"patient_context_consistency"` or `"redaction_placeholder"`),
`identifier_types` (`identifierTypes`, default `()`).

- `redaction_placeholder`: every match of `\[REDACTED:[A-Z_]+\]` is a finding
  with category `REDACTION_PLACEHOLDER`. Its purpose is to refuse write-back of
  already-redacted text into a system of record.
- `patient_context_consistency`: with no patient context it raises
  `EvaluationError("no patient context")`. Otherwise `allowed` is the FHIR id
  from `patient.reference` (the text after `Patient/`, uppercased) plus every
  identifier value uppercased. It flags every match of
  `\bPatient/([A-Za-z0-9\-.]{1,64})` and every match of an access-key identifier
  type (`is_access_key=True`), plus access-key `json_keys`, when the candidate is
  not in `allowed`. A value matching the active patient produces no finding.

### `cascade`

**SDK behavior** (`src/hmcp/shared/guardrails/engine.py`). Runs its steps in
order, unions the findings per unit, then `merge_overlaps`. Its `kind` string is
`"cascade(" + ",".join(step kinds) + ")"`. `CascadeEvaluatorSpec` fields: `kind`
(`"cascade"`), `steps` (at least two; each step is a `pattern`, `prompt` or
`deid` spec; a `classifier` is not a cascade step).

`build_evaluator(spec, *, llm, deid, classifier=None)` turns a spec into a
runnable evaluator and fails fast at server construction: a `prompt` spec
without an LLM provider, a `deid` spec without a de-identification model, or a
`classifier` spec without a classifier provider, raises `ValueError`.

## Actions

`ActionSpec` fields: `on_finding` (`onFinding`, one of `"block"`, `"redact"`,
`"pseudonymize"`, `"annotate"`) and `placeholder` (default
`"[REDACTED:{category}]"`; it must contain `{category}` and no other `{...}`
field). With `n` = total findings after merging (**SDK behavior**):

- **block:** `n > 0` gives `blocked`, else `pass`. The tool's content is
  discarded at the output stage.
- **redact:** replace each finding via `redact(text, findings, placeholder)`;
  `redacted` if `n > 0`, else `pass`.
- **annotate:** content is unchanged; `annotated` if `n > 0`, else `pass`. The
  signal is the report entry.
- **pseudonymize at the output stage:** requires a patient context (else
  `EvaluationError("no patient context")`). Each finding is replaced with
  `vault.marker_for(context_id, expires_at, category, value)` right-to-left;
  `pseudonymized` if `n > 0`, else `pass`. Marker creation is bounded by
  `timeout_ms` exactly like evaluation; a timeout yields the verdict `error`.
- **pseudonymize at the input stage is re-identification:** the evaluator is
  **not** run. One regex, with the bracketed alternative first,
  `\[ ?([A-Z_]+_\d+) ?\]|\b([A-Z][A-Z_]*_\d+)\b`, matches two kinds of marker. A
  bracketed marker is resolved with
  `vault.resolve(context_id, "[" + group + "]")`; if it is unresolved the verdict
  is `blocked` and nothing is replaced. A bare token such as `NAME_1` (models
  often drop the brackets when copying a marker into a tool argument) is resolved
  as `"[" + token + "]"`: if the vault knows it in this context it is replaced;
  if not it is left untouched, is not counted and never blocks, so legitimate
  text such as `ICD_10` or `LINE_1` passes. If every bracketed marker resolves,
  the text is replaced and the verdict is `reidentified`. `findings` is the
  number of bracketed markers plus resolved bare markers; the verdict is `pass`
  when that is 0. No patient context raises
  `EvaluationError("no patient context")`. Re-identification is bounded by
  `timeout_ms` exactly like evaluation; a timeout yields the verdict `error`.

### The redaction invariant

**SDK behavior / normative.** A verdict of `redacted`, `pseudonymized` or
`reidentified` means **every** finding's characters are absent from the returned
object. The runtime applies every replacement through a setter that replaces the
slot (a root-level scalar is walked through a one-element holder so
its setter is not a no-op). If any replacement cannot be applied, the verdict is
`error`, never `redacted`.

## What content is inspected

**SDK behavior** (`src/hmcp/shared/guardrails/engine.py`). Text units are
extracted as follows.

Input (`run_input`):

- Walk `arguments` recursively. A `str` becomes a text unit with `key` = the
  nearest dict key. An `int` (not `bool`) becomes a text unit of its decimal
  string; if replaced, its slot becomes a string. Lists and dicts recurse. Other
  types (including `bool`, floats, `None`) are ignored.
- A root value that is itself a `str` or `int` (a non-dict `arguments`) is walked
  through a one-element holder so a replacement has a real setter; if a
  replacement is needed but cannot be observed by the caller, the run fails
  closed with `error`.

Output (`run_output`):

- Each `result.content` block: a `TextContent` contributes its text; an
  `EmbeddedResource` whose resource is `TextResourceContents` contributes that
  text; **any other block type** (image, audio, blob, resource link) raises
  `EvaluationError("uninspectable content")`.
- `result.structured_content` and `result._meta` are walked like input. Both are
  first normalized with pydantic's own serialization
  (`pydantic_core.to_jsonable_python`), so what is inspected is exactly what goes
  on the wire: tuples, sets and frozensets become arrays, `bytes` become UTF-8
  strings, and non-string dict keys become their wire strings. A value that
  cannot be serialized, or nesting deeper than the serializer allows, raises
  `EvaluationError` and the guardrail fails closed with verdict `error`.
- `result._meta` is inspected for **every key except** the guardrails extension
  key `com.innovaccer/hmcp-guardrails`. That key is excluded because the
  extension always replaces it with its own report (or strips it when no
  guardrail applied), so a tool can never forge or smuggle anything under it.
- **JSON text is walked with keys.** Because `MCPServer` serializes a
  dict-returning tool's output as JSON **text** (with `structured_content`
  possibly `null`), any `TextContent` / `TextResourceContents` whose text parses
  (`json.loads`) as a JSON object or array is walked like structured content:
  keys are available to `json_keys` rules and every string/int leaf is a unit.
  If anything changed, the block is re-serialized with
  `json.dumps(obj, indent=2, ensure_ascii=False)`. Re-serializing can change
  formatting and numbers (for example `1.50` → `1.5`), collapses duplicate keys
  to the last value, and writes a non-finite number parsed from non-strict input
  (for example `1e400`) as `Infinity`, which strict JSON parsers reject, so a
  changed block is not byte-identical to the original. Text that does not parse
  as an object or array is a single plain-text unit.

> **Important:** String dict keys are also units. The same walk applies to
> `arguments`, `structured_content`, `_meta` and JSON text. Every **string dict
> key** is also a text unit, with no `key` of its own (so `json_keys` rules never
> apply to a key), evaluated after the value it holds. A **non-string key** can
> occur only in the input `arguments` (in `structured_content`, `_meta` and JSON
> text every key is already a string): an **int key** (not `bool`) is a unit too,
> as its decimal string, exactly like an int value, and a replacement renames it
> to the new string; every other non-string key (`bool`, float, `None`, …) is not
> a unit, and the value under it is walked with no `key`, so `json_keys` rules do
> not apply to it. A replacement renames a key in place and keeps key order;
> markers in keys are re-identified at the input stage like values; keys are sent
> to a `prompt` evaluator like values, so a finding on a key renames it. If a
> renamed key would collide with an existing key (for example two SSN keys that
> both redact to `[REDACTED:SSN]`), the run fails with `error`; replacements
> already applied are not rolled back, and a `closed` guardrail then blocks the
> call or result. Under `failure_mode: "open"` the call or result instead
> proceeds with the replacements applied before the collision, so it is partially
> replaced.

### What is not inspected (v1 limits)

**SDK behavior.** HMCP 0.1.0 inspects `tools/call` arguments and results only.

- `resources/read` and `prompts/get` are **not** inspected.
- A tool result that is not a `CallToolResult` (for example an MRTR
  `InputRequiredResult`) is **not** inspected: the interceptor returns it
  unchanged.
- The guardrails extension key in `_meta` is **not** inspected: the extension
  always replaces or strips it, so nothing under it reaches the client.
- Non-text content blocks are an error, not silently skipped.
- Cross-call state, conversation history and the client's own model I/O are out
  of scope.

## Negotiation, mandatory guardrails and `appliesTo`

**Normative.** Extensions are reverse-DNS identifiers advertised by the server in
[`server/discover`](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
and declared per request under `clientCapabilities.extensions`, per the base
spec's
[extension negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning.md#extension-negotiation).
The guardrails identifier is `com.innovaccer/hmcp-guardrails`; its revision is
`"1"`. **SDK behavior:** `GuardrailsExtension.settings()` publishes `revision`
and an `available` list; each entry has `id`, `version`, `digest`, `stages`, a
`definition` resource URI (`hmcp://guardrails/<short-name>`), `enforcedBy:
"server"` and a `mandatory` boolean. The same definitions are published as
`application/json` resources at those URIs.

**Requesting guardrails (normative).** A client puts a `require` list under the
extension key in the request `_meta`:

```json
{
  "method": "tools/call",
  "params": {
    "name": "get_observations",
    "arguments": { "patient": "Patient/test-0001" },
    "_meta": {
      "com.innovaccer/hmcp-guardrails": {
        "require": ["com.innovaccer.hmcp.guardrails/phi-redaction", "com.innovaccer.hmcp.guardrails/phi-deidentification@1"]
      }
    }
  }
}
```

Each entry is either `"<id>"` or `"<id>@<major>"`. **SDK behavior:** the server
validates the list **before the tool runs**. A non-dict container, or a `require`
value that is not a list of strings, raises HMCP error code `-31011`
(`GUARDRAIL_UNAVAILABLE`) with no `guardrail` key. An unknown id, a non-digit
major, or a major that differs from the definition's `major` also raise `-31011`,
with the offending string echoed in the error `data` under `guardrail` (capped at
200 characters). The tool never runs.

**Mandatory guardrails (normative).** A server may mark definitions mandatory.
A mandatory definition runs on every call it `applies` to. **SDK behavior:**
`mandatory` is a list of ids supplied to `GuardrailsConfig`; a mandatory id that
is not in `definitions` raises `ValueError` at construction.

**`appliesTo` (SDK behavior).** Mandatory definitions are filtered by
`applies(scope, tool_name)`. A **required** definition is applied regardless of
`appliesTo`, because the client asked for it explicitly.

**No bypass by omission (normative / SDK behavior).** An interceptor runs even
when a client does not declare the extension. A client that omits the
declaration does not thereby skip a mandatory guardrail.

## The report in the result `_meta`

**Normative.** After the call, the server writes one report entry per definition
per stage that ran into the result `_meta` under the guardrails extension key. A
`GuardrailReportEntry` has exactly these wire fields (camelCase): `id`,
`version`, `digest`, `stage` (`"input"` / `"output"`), `verdict`, `evaluator`,
`findings` (a count, never text) and `durationMs`. `verdict` is one of `pass`,
`redacted`, `pseudonymized`, `reidentified`, `annotated`, `blocked`, `error`.

```json
{
  "_meta": {
    "com.innovaccer/hmcp-guardrails": {
      "report": [
        {
          "id": "com.innovaccer.hmcp.guardrails/phi-redaction",
          "version": "1.0.0",
          "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
          "stage": "output",
          "verdict": "redacted",
          "evaluator": "cascade(pattern,prompt)",
          "findings": 3,
          "durationMs": 42
        }
      ]
    }
  }
}
```

A blocked call returns a `CallToolResult` with `isError: true`, a single text
block `Blocked by guardrail <id>`, and the same report. At the output stage a
block discards the tool's content entirely. **SDK behavior:** the report is
appended for every definition that ran, whether it passed, redacted, blocked or
errored. The value under the guardrails key comes only from the extension: when
a guardrail ran, the extension overwrites any tool-supplied value there with its
own `report`, and when none ran it strips a tool-supplied key, so a tool can
never forge a report.

**Client verification (normative).** For each required entry, the result report
must contain a matching entry; otherwise the client raises
`GuardrailReportMissing`, whose message names the required string only. An
unpinned `"<id>"` matches a report entry on id alone. A pinned `"<id>@<major>"`
additionally requires the entry's `version` to have the same leading-integer
major (the digits before the first `.`); a missing or non-string `version` does
not satisfy a pin, and a malformed pin is never satisfied. This check is applied
**even when the result is an error**, so a failed call cannot silently skip a
required guardrail.

## Failure modes

**SDK behavior.** A guardrail fails closed by default.

- `timeout_ms` bounds the awaited model calls (a `prompt` evaluator's LLM call
  and a `deid` evaluator's model call) and the vault / pseudonymization and
  re-identification steps, each wrapped in `asyncio.wait_for(timeout_ms / 1000)`.
  It **cannot** interrupt a synchronous evaluator (`pattern`, `builtin`), which
  runs to completion on the event loop; deployer-written regexes must therefore
  be linear in the input size.
- `EvaluationError`, a timeout, `VaultFull`, or any other exception yields the
  verdict `error` with `findings` = 0 and content unchanged, except after a
  dict-key collision, where replacements applied before the collision stay
  applied (a `closed` guardrail then blocks the call or result, so they are not
  returned). Model and provider exceptions surface only as fixed-message
  evaluation errors (for example `LLM call failed`, `invalid LLM verdict`,
  `deid model failed`, `invalid deid response`, `classifier call failed`,
  `invalid classifier result`) and never carry or chain the
  provider's message. The only log line
  is `guardrail %s failed: %s` with the definition id and the exception type
  name; content, finding values, prompts and provider replies are never logged.
- With `failureMode: "closed"` (the default), an `error` verdict blocks the call
  and the client sees a blocked result. With `failureMode: "open"`, the call
  proceeds and the report still carries the `error` verdict.
- Requesting a guardrail the server cannot apply is `-31011`
  (`GUARDRAIL_UNAVAILABLE`), before the tool runs.
- A missing report for a required guardrail is `GuardrailReportMissing` on the
  client.

## The four reference guardrails

**SDK behavior** (`src/hmcp/shared/guardrails/reference.py`). HMCP ships four
definitions, all version `1.0.0`, all with `failureMode: "closed"`. Their full
ids share the prefix `com.innovaccer.hmcp.guardrails/`. Retrieve them with
`reference_definitions()` (keyed by id) or the individual factory functions.

| Short name | Id suffix | Stages | Evaluator | Action | Timeout | What it does |
| --- | --- | --- | --- | --- | --- | --- |
| `phi-redaction` | `/phi-redaction` | output | `cascade(pattern, prompt)` | `redact` | 15000 ms | Attempts to redact identifiers in the HIPAA Safe Harbor categories from tool results: deterministic patterns first, then an LLM prompt for names, locations, dates and other identifiers in free text. Best effort only: it can miss identifiers, a pass verdict does not mean the result is free of PHI, and redaction does not by itself de-identify data under HIPAA. |
| `phi-deidentification` | `/phi-deidentification` | output, input | `cascade(pattern, prompt)` | `pseudonymize` | 15000 ms | Replaces PHI found in tool results with markers such as `[NAME_1]` and restores them in later tool arguments inside the same patient context; unknown **bracketed** markers are blocked, and bare known markers are restored too. Best effort only: it can miss identifiers, and pseudonymization does not by itself de-identify data under HIPAA. |
| `patient-context-consistency` | `/patient-context-consistency` | input, output | `builtin` `patient_context_consistency` (mrn, empi) | `block` | 2000 ms | Blocks a call whose arguments or result contain an MRN, EMPI or `Patient/<id>` reference that is not one of the active patient context's identifiers; other references to a different patient, such as a name, are not detected. `appliesTo.patientScopes` is `["patient"]`. |
| `redaction-placeholder-writeback` | `/redaction-placeholder-writeback` | input | `builtin` `redaction_placeholder` | `block` | 1000 ms | Blocks tool arguments that contain `[REDACTED:...]` placeholders, to stop redacted text being written back to a system of record. |

`PHI_IDENTIFIER_TYPES` (used by the first two) defines these identifier types
(name → category, patterns, JSON keys, access key):

| name | category | text patterns | json keys | access key |
| --- | --- | --- | --- | --- |
| `ssn` | `SSN` | `\b(\d{3}-\d{2}-\d{4})\b` | `ssn`, `social_security_number` | no |
| `phone` | `PHONE` | `(?<!\d)(\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})(?!\d)` | `phone`, `phone_number`, `telephone`, `mobile` | no |
| `email` | `EMAIL` | `(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b` | `email`, `email_address` | no |
| `mrn` | `MRN` | `\bMRN[:#\s]*([A-Z0-9-]{4,20})\b` | `mrn`, `medical_record_number` | yes |
| `empi` | `OTHER_ID` | `\bEMPI[:#\s]*([A-Z0-9-]{4,30})\b` | `empi` | yes |
| `date` | `DATE` | `(?<!\d)(\d{1,2}/\d{1,2}/\d{2,4})(?!\d)`, `(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)` | `dob`, `birth_date`, `birthdate`, `date_of_birth` | no |
| `ip_address` | `IP_ADDRESS` | `\b((?:\d{1,3}\.){3}\d{1,3})\b` | — | no |
| `url` | `URL` | `\b(https?://[^\s<>"']+)` | — | no |
| `person_name` | `NAME` | — | `name`, `patient_name`, `first_name`, `last_name`, `given_name`, `family_name`, `full_name` | no |
| `address` | `LOCATION` | — | `address`, `street`, `address_line`, `city`, `zip`, `zip_code`, `postal_code` | no |

### Limits of the reference set

**SDK behavior / non-goal.** These definitions reduce risk; they do not
guarantee detection or removal.

- The `date` pattern also matches the date part of an ISO timestamp.
- The `email` pattern is anchored so a match may start only at the beginning of a
  run of local-part characters (a negative lookbehind), which keeps matching
  linear without length caps; an over-long local part is redacted in full.
- Patterns err toward **over-redaction**: the `ip_address` pattern treats any
  dotted quad as an IP address, so a version string such as `1.2.3.4` is
  redacted. Over-redaction is intentional for a redaction guardrail.
- `person_name` and `address` have no text patterns, so they only match when the
  string sits under a matching JSON key; free-text names rely on the prompt
  evaluator.
- The prompt evaluator's recall depends on the model and must be measured on the
  deployer's own data.

## Provider integration and data handling

**SDK behavior.** Three protocols integrate with the evaluators: `LLMProvider` for
`prompt` evaluators, `DeidentificationModel` for `deid` evaluators and
`ClassifierProvider` for `classifier` evaluators.

```python
class LLMProvider(Protocol):
    async def complete(self, *, system: str, user: str, max_tokens: int) -> str: ...

class DeidentificationModel(Protocol):
    async def process_text(
        self, request: DeidRequest
    ) -> Sequence[DeidTextResult | Mapping[str, Any]]: ...

class ClassifierProvider(Protocol):
    async def classify(
        self, request: ClassificationRequest
    ) -> ClassificationResult | Mapping[str, Any]: ...
```

**SDK behavior.** For a `prompt` evaluator, HMCP builds the `system` and `user`
strings itself: it renders the definition's templates with a fresh per-call
nonce and frames the inspected content between them, then calls `complete` with
the rendered strings. `complete` returns the model's reply as a string; the
provider must not log the content it is given, and any exception it raises fails
the evaluation: the guardrail reports an `error` verdict and then fails per its
`failureMode`. A definition is validated when it is constructed: an
`IdentifierType.textPatterns` regex that does not compile (or does not have
exactly one capture group), or an `ActionSpec.placeholder` that lacks
`{category}` (or adds another `{...}` field), raises `ValueError` before any call
runs.

### `DeidentificationModel`

**SDK behavior** (`src/hmcp/shared/guardrails/evaluators/deid.py`). HMCP ships
no de-identification model and no vendor adapter: a deployer implements
`DeidentificationModel` against its own self-hosted or BAA-covered model. The
four reference guardrails need none: they use only patterns, prompts and
builtins.

The request and response are the wire shape of a de-identification service's
`/process/text` API (text in; labeled entities with character spans out), so an
implementation can be a thin HTTP client. The types are frozen Pydantic models
whose field names are that service's wire names in snake_case:

| Type | Fields | Notes |
| --- | --- | --- |
| `DeidEntityDetection` | `return_entity: Literal[True] = True` | Extra fields forbidden. |
| `DeidRequest` | `text: list[str]` (kept out of `repr`), `entity_detection: DeidEntityDetection = DeidEntityDetection()` | Extra fields forbidden. `text` is unit text, i.e. PHI. The body is always `{"text": [<texts>], "entity_detection": {"return_entity": true}}`. |
| `DeidEntityLocation` | `stt_idx: StrictInt`, `end_idx: StrictInt` | Extra fields ignored. |
| `DeidEntity` | `text: StrictStr` (kept out of `repr`), `best_label: StrictStr`, `labels: dict[str, float] = {}`, `location: DeidEntityLocation` | Extra fields ignored. |
| `DeidTextResult` | `entities: list[DeidEntity]` | `entities` is required; extra fields ignored. |

All eight names (`DeidEntity`, `DeidEntityDetection`, `DeidEntityLocation`,
`DeidEvaluator`, `DeidRequest`, `DeidTextResult`, `DeidentificationModel` and
`label_to_category`) are exported from **both**
`hmcp.shared.guardrails.evaluators` and `hmcp.shared.guardrails`.

A `deid` evaluator sends units in batches of at most 50 and expects a JSON list,
one object per text in order, each with an `entities` list of
`{"text", "best_label", "labels", "location": {"stt_idx", "end_idx"}}`. Extra
keys such as `processed_text`, `entities_present`, `characters_processed` and
`languages_detected` are ignored, and `process_text` may return a raw JSON list
as is. Offsets are Python `str` (code point) indices. HMCP never forwards an
entity selection to the model.

Fail-closed rules:

- A response that is not a list or tuple of the batch length, or an item that
  does not validate as `DeidTextResult`, raises
  `EvaluationError("invalid deid response")`.
- A `best_label` that does not match `^[A-Z][A-Z0-9_]*$`, an out-of-range span,
  or `text[stt_idx:end_idx] != entity.text` raises `EvaluationError`; messages
  carry indices only, never content or the label, because a model can return any
  uppercase label, including an uppercased name.
- Every exception raised by the model is caught and re-raised as a fixed
  `EvaluationError("deid model failed")` with no chained cause or context, so a
  provider message cannot carry PHI; the guardrail then fails per its failure
  mode.

`label_to_category` maps `best_label` to a `PhiCategory` by the first matching
rule (`NAME*`→NAME; `DATE*`/`DOB`/`AGE`→DATE; `PHONE_NUMBER`→PHONE;
`FAX_NUMBER`→FAX; `EMAIL_ADDRESS`→EMAIL; `SSN`→SSN;
`MEDICAL_RECORD_NUMBER`→MRN; `HEALTHCARE_NUMBER`→HEALTH_PLAN_ID;
`BANK_ACCOUNT`/`CREDIT_CARD`/`ROUTING_NUMBER`/`ACCOUNT_NUMBER`→ACCOUNT_NUMBER;
`DRIVER_LICENSE`/`PASSPORT_NUMBER`→LICENSE_NUMBER; `VEHICLE_ID`→VEHICLE_ID;
`DEVICE_ID`→DEVICE_ID; `URL`→URL; `IP_ADDRESS`→IP_ADDRESS; `LOCATION*`→LOCATION;
anything else→OTHER_ID).

A minimal implementation is a thin HTTP client:

```python
# Deployer code, not shipped by HMCP.
from collections.abc import Mapping, Sequence
from typing import Any

import httpx2

from hmcp.shared.guardrails.evaluators import DeidRequest, DeidTextResult


class HttpDeidModel:
    """POSTs the request to a de-identification service's /process/text endpoint and returns its JSON list."""

    def __init__(self, base_url: str, token: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token

    async def process_text(
        self, request: DeidRequest
    ) -> Sequence[DeidTextResult | Mapping[str, Any]]:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        async with httpx2.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/process/text",
                json=request.model_dump(mode="json"),
                headers=headers,
            )
        if response.status_code >= 400:
            # Never include the response body: it embeds PHI.
            raise RuntimeError(f"deid model HTTP {response.status_code}")
        return response.json()
```

For example, `HttpDeidModel("https://deid.internal.example.com", token=...)`; pass
it as `GuardrailsConfig(deid=...)` for definitions that use a `deid` evaluator.

### `ClassifierProvider`

**SDK behavior** (`src/hmcp/shared/guardrails/evaluators/classifier.py`). HMCP
ships no classifier and no vendor adapter: a deployer implements
`ClassifierProvider` against its own self-hosted or BAA-covered classifier
service, such as a small language model (SLM). The classifier name in a
`classifier` evaluator is resolved by the provider to a model or service.

A `classify` call carries a `ClassificationRequest(classifier, stage, text)`; the
provider returns a `ClassificationResult` with a required strict boolean
`flagged`, or a mapping that validates as one (extra keys are ignored). The
provider must not log the text it is given or raise it, and it raises for an
unknown classifier name. Any exception it raises fails the evaluation: the
guardrail reports an `error` verdict and then fails per its `failureMode`.

### PHI and BAA data handling

**Normative.** Content sent to a `prompt` evaluator is PHI: the model must be
covered by a BAA with zero data retention. Content sent to a `deid` evaluator
goes to a self-hosted or BAA-covered de-identification model. Content sent to a
`classifier` evaluator is PHI: the classifier service must be self-hosted or
covered by a BAA, with retention appropriate for PHI. Markers created by
pseudonymization are re-identified only inside the patient context that produced
them, and PHI must not outlive that context: the in-memory vault drops a
context's entries at its `expires_at`. The `dataHandling` field of a definition
carries these notes to the deployer; the reference definitions set
`promptEvaluatorModel` accordingly. HMCP never logs content,
finding values, prompts, replies or response bodies; log lines carry only a
definition id and an exception type name.

**Future work.** A persistent, cross-process marker vault is not shipped in v1;
the only vault is the process-local `InMemoryMarkerVault`.

## Server integration

**SDK behavior.** Pass a `GuardrailsConfig` to `HMCPServer`. When one is
supplied, a `GuardrailsExtension` is installed inside the patient-context
extension, so guardrails see the resolved patient context. It is reachable
through the `HMCPServer.guardrails` property.

```python
from hmcp.server.hmcp_server import HMCPServer
from hmcp.server.guardrails import GuardrailsConfig
from hmcp.shared.guardrails import phi_deidentification, phi_redaction
from hmcp.shared.guardrails.evaluators.prompt import LLMProvider


class MyLLMProvider:
    """Runs on your own BAA-covered model."""

    async def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        raise NotImplementedError


server = HMCPServer(
    "my-hmcp-server",
    guardrails=GuardrailsConfig(
        definitions=[phi_redaction(), phi_deidentification()],
        mandatory=["com.innovaccer.hmcp.guardrails/phi-redaction"],
        llm=MyLLMProvider(),
    ),
)
```

`GuardrailsConfig` fields: `definitions` (a sequence of
`GuardrailDefinition`), `mandatory` (ids, default `()`), `llm` (an `LLMProvider`
or `None`), `deid` (a `DeidentificationModel` or `None`), `vault` (a
`MarkerVault`, default `InMemoryMarkerVault`) and `classifier` (a
`ClassifierProvider` or `None`, default `None`). A prompt definition without an
`llm`, a deid definition without a `deid` model, or a classifier definition
without a `classifier` provider, fails fast at construction.

## Client integration

**SDK behavior.** `HMCPClientConnector` declares both extensions by default and
`call_tool` accepts a `require_guardrails` list. It merges the ids into the
request `_meta` and, after the call, verifies the report. An unpinned `"<id>"`
is satisfied by a report entry with that id; a pinned `"<id>@<major>"` also
requires the entry's `version` to have that major. If a requirement is not
satisfied, it raises `GuardrailReportMissing` (the message names the required
string only), including when the result is an error.

```python
from hmcp.client.client_connector import HMCPClientConnector
from hmcp.client.patient_context import GuardrailReportMissing

PHI_REDACTION_ID = "com.innovaccer.hmcp.guardrails/phi-redaction"


async def fetch(connector: HMCPClientConnector) -> None:
    handle = await connector.open_patient_context("Patient/test-0001")
    try:
        result = await connector.call_tool(
            "get_observations",
            {"patient": "Patient/test-0001"},
            patient_context=handle,
            require_guardrails=[PHI_REDACTION_ID],
        )
    except GuardrailReportMissing:
        # The server returned no report entry for a required guardrail.
        raise
    finally:
        await connector.close_patient_context(handle)
```

## Related pages

- [HMCP overview](./index.md)
- [Patient context](./patient-context.md): how a context is opened and how the
  guardrails interceptor sees it.
- [Migrating to 0.1.0](./migration-0.1.md): sampling removal and other changes.
