"""Guardrail runtime engine for HMCP 0.1.0.

The runtime applies one :class:`GuardrailDefinition` to a tool call. At the
input stage it walks and may MUTATE the arguments in place (redaction and
re-identification); at the output stage it returns a new ``CallToolResult``
when the content changes. Every run yields a ``GuardrailReportEntry``. Every
failure fails closed and never logs content: only the definition id and the
exception type name are recorded.
"""

import asyncio
import json
import logging
import re
from time import monotonic
from typing import Any, Callable, Sequence

from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    TextContent,
    TextResourceContents,
)
from pydantic_core import to_jsonable_python

from hmcp.shared.constants import GUARDRAILS_EXTENSION
from hmcp.shared.types import (
    GuardrailReportEntry,
    GuardrailStage,
    GuardrailVerdict,
)

from .definition import (
    BuiltinEvaluatorSpec,
    CascadeEvaluatorSpec,
    DeidEvaluatorSpec,
    EvaluatorSpec,
    GuardrailDefinition,
    PatternEvaluatorSpec,
    PromptEvaluatorSpec,
)
from .evaluators.base import (
    EvaluationContext,
    EvaluationError,
    Evaluator,
    TextUnit,
)
from .evaluators.builtin import (
    PatientContextConsistencyEvaluator,
    RedactionPlaceholderEvaluator,
)
from .evaluators.deid import DeidEvaluator, DeidentificationModel
from .evaluators.pattern import PatternEvaluator
from .evaluators.prompt import LLMProvider, PromptEvaluator
from .findings import Finding, merge_overlaps, redact
from .vault import MarkerVault

logger = logging.getLogger(__name__)

# Markers look like "[NAME_1]"; a resolver accepts optional spaces inside the
# brackets, and also a bare "NAME_1" because models often drop the brackets when
# copying a marker into a tool argument.
_MARKER = re.compile(r"\[ ?([A-Z_]+_\d+) ?\]|\b([A-Z][A-Z_]*_\d+)\b")

_Emit = Callable[[TextUnit, Callable[[str], None]], None]
_Setter = Callable[[str], None]


def _noop_setter(_text: str) -> None:
    """Top-level setter for the one-element holder, which is never replaced."""


class CascadeEvaluator:
    """Run each step in order, union the findings per unit, then merge overlaps."""

    def __init__(self, evaluators: Sequence[Evaluator]) -> None:
        self._evaluators = tuple(evaluators)
        self.kind = "cascade(" + ",".join(e.kind for e in self._evaluators) + ")"

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        combined: list[list[Finding]] = [[] for _ in units]
        for evaluator in self._evaluators:
            step = await evaluator.find(units, ctx)
            if len(step) != len(units):
                raise EvaluationError("invalid evaluator result")
            for index, findings in enumerate(step):
                combined[index].extend(findings)
        return [
            merge_overlaps(findings, text=unit.text)
            for unit, findings in zip(units, combined)
        ]


def build_evaluator(
    spec: EvaluatorSpec,
    *,
    llm: LLMProvider | None,
    deid: DeidentificationModel | None,
) -> Evaluator:
    """Turn a declarative spec into a runnable evaluator; fail fast if unusable."""
    if isinstance(spec, PatternEvaluatorSpec):
        return PatternEvaluator(spec)
    if isinstance(spec, PromptEvaluatorSpec):
        if llm is None:
            raise ValueError("prompt evaluator requires an LLM provider")
        return PromptEvaluator(spec, llm)
    if isinstance(spec, DeidEvaluatorSpec):
        if deid is None:
            raise ValueError("deid evaluator requires a de-identification model")
        return DeidEvaluator(spec, deid)
    if isinstance(spec, BuiltinEvaluatorSpec):
        if spec.name == "redaction_placeholder":
            return RedactionPlaceholderEvaluator()
        if spec.name == "patient_context_consistency":
            return PatientContextConsistencyEvaluator(spec.identifier_types)
        raise ValueError(f"unknown builtin evaluator {spec.name!r}")
    if isinstance(spec, CascadeEvaluatorSpec):
        return CascadeEvaluator(
            [build_evaluator(step, llm=llm, deid=deid) for step in spec.steps]
        )
    raise ValueError(f"unknown evaluator kind {getattr(spec, 'kind', None)!r}")


def _walk(
    value: Any,
    key: str | None,
    set_value: _Setter,
    emit: _Emit,
) -> None:
    """Walk a JSON-like value, emitting a TextUnit + setter for every string/int.

    Lists and dicts are mutated in place by the setters handed down here, so
    the caller's object identity is preserved. bool is ignored (not an int).
    Dict keys are walked too, post-order after their value, as units with
    ``key=None`` so json_keys rules never apply to a key; a renamed key is
    replaced in place and any collision fails closed. str and int keys (not
    bool) are units (an int key is evaluated as text); any other key is
    ignored, and the value under a non-str key is walked with ``key=None``.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, str):
        emit(TextUnit(value, key), set_value)
    elif isinstance(value, int):
        # An int is evaluated as text; if replaced, its slot becomes a str.
        emit(TextUnit(str(value), key), set_value)
    elif isinstance(value, dict):
        for item_key, item_value in list(value.items()):
            key_is_str = isinstance(item_key, str)
            _walk(
                item_value,
                item_key if key_is_str else None,
                lambda new, d=value, k=item_key: d.__setitem__(k, new),
                emit,
            )
            if key_is_str:
                # Post-order: the value's setters run before this key is renamed.
                emit(
                    TextUnit(item_key, None),
                    lambda new, d=value, k=item_key: _rename_key(d, k, new),
                )
            elif isinstance(item_key, int) and not isinstance(item_key, bool):
                # An int key is evaluated as text, like an int value; a
                # replacement renames it to the new str. (bool is excluded.)
                emit(
                    TextUnit(str(item_key), None),
                    lambda new, d=value, k=item_key: _rename_key(d, k, new),
                )
    elif isinstance(value, list):
        for index, item_value in enumerate(value):
            _walk(
                item_value,
                key,
                lambda new, l=value, i=index: l.__setitem__(i, new),
                emit,
            )


def _rename_key(d: dict, old: str | int, new: str) -> None:
    """Rename a dict key in place, keeping key order; any collision fails closed."""
    if new == old:
        return
    if old not in d or new in d:
        raise EvaluationError("cannot apply replacement")
    items = list(d.items())
    d.clear()
    for k, v in items:
        d[new if k == old else k] = v


def _extract_units(value: Any) -> tuple[list[TextUnit], list[_Setter], bool]:
    """Return ``(units, setters, root_is_scalar)`` for a JSON-like input value.

    The root value is wrapped in a one-element holder so that a root-level
    str/int gets a real setter rather than a no-op. A dict or list is still
    mutated in place (identity preserved); a scalar root can only be replaced
    inside the holder, which the caller of ``run_input`` cannot observe, so
    ``root_is_scalar`` lets the caller fail closed when a replacement is made.
    """
    units: list[TextUnit] = []
    setters: list[_Setter] = []

    def emit(unit: TextUnit, setter: _Setter) -> None:
        units.append(unit)
        setters.append(setter)

    _walk([value], None, _noop_setter, emit)
    root_is_scalar = isinstance(value, (str, int)) and not isinstance(value, bool)
    return units, setters, root_is_scalar


def _apply_replacements(
    replacements: Sequence[str | None], setters: Sequence[_Setter]
) -> int:
    """Apply each replacement via its setter; return how many were applied."""
    applied = 0
    for replacement, setter in zip(replacements, setters):
        if replacement is not None:
            setter(replacement)
            applied += 1
    return applied


class GuardrailRuntime:
    """Apply one guardrail definition to tool arguments and results."""

    def __init__(
        self,
        definition: GuardrailDefinition,
        *,
        llm: LLMProvider | None = None,
        deid: DeidentificationModel | None = None,
        vault: MarkerVault,
    ) -> None:
        self._definition = definition
        self._vault = vault
        self._evaluator = build_evaluator(definition.evaluator, llm=llm, deid=deid)
        self.kind = self._evaluator.kind

    async def run_input(
        self, arguments: dict[str, Any] | None, ctx: EvaluationContext
    ) -> GuardrailReportEntry:
        t0 = monotonic()
        try:
            units, setters, root_is_scalar = _extract_units(arguments)
            if self._definition.action.on_finding == "pseudonymize":
                verdict, findings, applied = await asyncio.wait_for(
                    self._reidentify(units, setters, ctx),
                    self._definition.timeout_ms / 1000,
                )
            else:
                results = await self._evaluate(units, ctx)
                verdict, findings, applied = await self._apply(
                    units, setters, results, ctx
                )
            if root_is_scalar and applied:
                # A scalar root cannot be handed back to the caller: fail closed.
                raise EvaluationError("cannot apply replacement")
        except Exception as exc:  # noqa: BLE001 - fail closed on ANY error
            logger.warning(
                "guardrail %s failed: %s", self._definition.id, type(exc).__name__
            )
            return self._entry("input", GuardrailVerdict.ERROR, 0, t0)
        return self._entry("input", verdict, findings, t0)

    async def run_output(
        self, result: CallToolResult, ctx: EvaluationContext
    ) -> tuple[CallToolResult, GuardrailReportEntry]:
        t0 = monotonic()
        try:
            units, setters, finalize = _extract_output(result)
            results = await self._evaluate(units, ctx)
            verdict, findings, _applied = await asyncio.wait_for(
                self._apply(units, setters, results, ctx),
                self._definition.timeout_ms / 1000,
            )
            new_result = finalize()
        except Exception as exc:  # noqa: BLE001 - fail closed on ANY error
            logger.warning(
                "guardrail %s failed: %s", self._definition.id, type(exc).__name__
            )
            return result, self._entry("output", GuardrailVerdict.ERROR, 0, t0)
        return new_result, self._entry("output", verdict, findings, t0)

    async def _evaluate(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        """Run the evaluator under the definition's timeout.

        ``timeout_ms`` can only interrupt an evaluator that yields to the event
        loop (prompt, deid). A synchronous evaluator (pattern, builtin) runs to
        completion on the event loop and cannot be interrupted by this timeout;
        it must therefore be kept linear in the input size.
        """
        results = await asyncio.wait_for(
            self._evaluator.find(units, ctx), self._definition.timeout_ms / 1000
        )
        if len(results) != len(units):
            raise EvaluationError("invalid evaluator result")
        return [
            merge_overlaps(findings, text=unit.text)
            for unit, findings in zip(units, results)
        ]

    async def _apply(
        self,
        units: Sequence[TextUnit],
        setters: Sequence[_Setter],
        results: Sequence[Sequence[Finding]],
        ctx: EvaluationContext,
    ) -> tuple[GuardrailVerdict, int, int]:
        findings = sum(len(f) for f in results)
        action = self._definition.action.on_finding

        if action == "block":
            return (
                (GuardrailVerdict.BLOCKED if findings else GuardrailVerdict.PASS),
                findings,
                0,
            )
        if action == "annotate":
            return (
                (GuardrailVerdict.ANNOTATED if findings else GuardrailVerdict.PASS),
                findings,
                0,
            )
        if action == "redact":
            replacements = [
                (
                    redact(
                        unit.text, unit_findings, self._definition.action.placeholder
                    )
                    if unit_findings
                    else None
                )
                for unit, unit_findings in zip(units, results)
            ]
            applied = _apply_replacements(replacements, setters)
            return (
                (GuardrailVerdict.REDACTED if findings else GuardrailVerdict.PASS),
                findings,
                applied,
            )
        if action == "pseudonymize":
            replacements = await self._pseudonymize(units, results, ctx)
            applied = _apply_replacements(replacements, setters)
            return (
                (GuardrailVerdict.PSEUDONYMIZED if findings else GuardrailVerdict.PASS),
                findings,
                applied,
            )
        raise EvaluationError("unknown action")

    async def _pseudonymize(
        self,
        units: Sequence[TextUnit],
        results: Sequence[Sequence[Finding]],
        ctx: EvaluationContext,
    ) -> list[str | None]:
        patient_context = ctx.patient_context
        if patient_context is None:
            raise EvaluationError("no patient context")

        replacements: list[str | None] = []
        for unit, unit_findings in zip(units, results):
            if not unit_findings:
                replacements.append(None)
                continue
            text = unit.text
            # Right-to-left so earlier spans stay valid as markers change length.
            for finding in sorted(unit_findings, key=lambda f: f.start, reverse=True):
                marker = await self._vault.marker_for(
                    patient_context.context_id,
                    patient_context.expires_at,
                    finding.category,
                    finding.value,
                )
                text = text[: finding.start] + marker + text[finding.end :]
            replacements.append(text)
        return replacements

    async def _reidentify(
        self,
        units: Sequence[TextUnit],
        setters: Sequence[_Setter],
        ctx: EvaluationContext,
    ) -> tuple[GuardrailVerdict, int, int]:
        patient_context = ctx.patient_context
        if patient_context is None:
            raise EvaluationError("no patient context")

        matches_per_unit = [list(_MARKER.finditer(unit.text)) for unit in units]
        total = sum(len(matches) for matches in matches_per_unit)
        if total == 0:
            return GuardrailVerdict.PASS, 0, 0

        # Resolve every match before replacing anything. A bracketed marker MUST
        # resolve; an unknown bare token is legitimate text and passes through.
        # ``n`` counts bracketed matches plus resolved bare matches.
        n = 0
        blocked = False
        resolved_per_unit: list[list[tuple[re.Match[str], str]]] = []
        for matches in matches_per_unit:
            resolved: list[tuple[re.Match[str], str]] = []
            for match in matches:
                bracketed = match.group(1) is not None
                token = match.group(1) or match.group(2)
                value = await self._vault.resolve(
                    patient_context.context_id, "[" + token + "]"
                )
                if bracketed:
                    n += 1
                    if value is None:
                        blocked = True
                    else:
                        resolved.append((match, value))
                elif value is not None:
                    n += 1
                    resolved.append((match, value))
            resolved_per_unit.append(resolved)

        if blocked:
            return GuardrailVerdict.BLOCKED, n, 0
        if n == 0:
            return GuardrailVerdict.PASS, 0, 0

        replacements: list[str | None] = []
        for unit, resolved in zip(units, resolved_per_unit):
            if not resolved:
                replacements.append(None)
                continue
            text = unit.text
            for match, value in sorted(
                resolved, key=lambda item: item[0].start(), reverse=True
            ):
                text = text[: match.start()] + value + text[match.end() :]
            replacements.append(text)

        applied = _apply_replacements(replacements, setters)
        return GuardrailVerdict.REIDENTIFIED, n, applied

    def _entry(
        self,
        stage: GuardrailStage,
        verdict: GuardrailVerdict,
        findings: int,
        t0: float,
    ) -> GuardrailReportEntry:
        return GuardrailReportEntry(
            id=self._definition.id,
            version=self._definition.version,
            digest=self._definition.digest,
            stage=stage,
            verdict=verdict,
            evaluator=self.kind,
            findings=findings,
            duration_ms=int((monotonic() - t0) * 1000),
        )


def _parse_json_container(text: str) -> Any:
    """Return the parsed value only when ``text`` is a JSON object or array."""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _replace_block_text(block: Any, text: str) -> Any:
    """Copy a TextContent / EmbeddedResource block with new text."""
    if isinstance(block, TextContent):
        return block.model_copy(update={"text": text})
    return block.model_copy(
        update={"resource": block.resource.model_copy(update={"text": text})}
    )


def _normalise(value: Any) -> Any:
    """Convert a value to the JSON-reachable form pydantic serialises on the wire.

    Uses pydantic's own ``to_jsonable_python`` so tuples, sets and frozensets
    become lists and bytes become str exactly as ``CallToolResult.model_dump(
    mode="json")`` produces. A value that cannot be serialised (an unknown type,
    over-deep nesting or a circular reference) raises ``EvaluationError`` so the
    guardrail fails closed.
    """
    normalise_failed = False
    normalised: Any = None
    try:
        normalised = to_jsonable_python(value)
    except Exception:  # noqa: BLE001 - fail closed on ANY serialisation error
        normalise_failed = True
    if normalise_failed:
        # Raised OUTSIDE the except block so __context__ stays None.
        raise EvaluationError("uninspectable content") from None
    return normalised


def _extract_output(
    result: CallToolResult,
) -> tuple[list[TextUnit], list[_Setter], Callable[[], CallToolResult]]:
    """Extract text units from a result plus a finalizer that rebuilds it.

    A text block whose text parses as a JSON object/array is walked like
    structured content (so json_keys rules see the keys) and re-serialised with
    ``json.dumps(obj, indent=2, ensure_ascii=False)`` only when a replacement
    was applied. Any other text is a single plain-text unit. The result's
    ``_meta`` is walked the same way so PHI there is inspected too. Only
    ``GUARDRAILS_EXTENSION`` is excluded, because the guardrails extension always
    replaces that key (non-empty report) or strips it (empty report); every other
    key, including ``PATIENT_CONTEXT_EXTENSION``, is tool-controlled and is
    inspected. Content blocks are copied lazily (never mutated in place);
    structured content and ``_meta`` are normalised to the JSON form pydantic
    serialises on the wire (tuples/sets/frozensets become lists, bytes become
    str) before walking, so a non-JSON value cannot hide PHI. Any block we cannot
    inspect is an error.
    """
    units: list[TextUnit] = []
    setters: list[_Setter] = []
    blocks = list(result.content)
    state = {"changed": False}
    json_blocks: list[dict[str, Any]] = []

    def register_plain_text(index: int, block: Any, text: str) -> None:
        units.append(TextUnit(text, None))

        def set_text(new: str, i: int = index, b: Any = block) -> None:
            blocks[i] = _replace_block_text(b, new)
            state["changed"] = True

        setters.append(set_text)

    def register_json_text(index: int, block: Any, obj: Any) -> None:
        entry: dict[str, Any] = {
            "index": index,
            "block": block,
            "obj": obj,
            "changed": False,
        }
        json_blocks.append(entry)

        def emit(unit: TextUnit, setter: _Setter) -> None:
            units.append(unit)

            def mark(new: str, inner: _Setter = setter) -> None:
                inner(new)
                entry["changed"] = True
                state["changed"] = True

            setters.append(mark)

        _walk(obj, None, _noop_setter, emit)

    for index, block in enumerate(result.content):
        if isinstance(block, TextContent):
            text = block.text
        elif isinstance(block, EmbeddedResource) and isinstance(
            block.resource, TextResourceContents
        ):
            text = block.resource.text
        else:
            raise EvaluationError("uninspectable content")

        parsed = _parse_json_container(text)
        if parsed is None:
            register_plain_text(index, block, text)
        else:
            register_json_text(index, block, parsed)

    structured = [_normalise(result.structured_content)]

    def emit(unit: TextUnit, setter: _Setter) -> None:
        units.append(unit)

        def mark(text: str, inner: _Setter = setter) -> None:
            inner(text)
            state["changed"] = True

        setters.append(mark)

    # Wrap the root in a one-element holder so a bare str/int structured_content
    # gets a real setter; a dict/list is still mutated in place inside the holder.
    _walk(structured, None, _noop_setter, emit)

    # ``_meta`` is inspected like structured content so PHI there cannot bypass
    # guardrails. Only the guardrails key is excluded: the extension always
    # either replaces it (non-empty report) or strips it (empty report), so a
    # tool value there never reaches the client. Every other key, including the
    # patient-context key, is tool-controlled and is inspected.
    original_meta = result.meta
    excluded_meta_keys = (GUARDRAILS_EXTENSION,)
    if isinstance(original_meta, dict):
        meta: dict[str, Any] | None = _normalise(
            {k: v for k, v in original_meta.items() if k not in excluded_meta_keys}
        )
        _walk(meta, None, _noop_setter, emit)
    else:
        meta = None

    def finalize() -> CallToolResult:
        if not state["changed"]:
            return result
        for entry in json_blocks:
            if entry["changed"]:
                new_text = json.dumps(entry["obj"], indent=2, ensure_ascii=False)
                blocks[entry["index"]] = _replace_block_text(entry["block"], new_text)
        update: dict[str, Any] = {
            "content": blocks,
            "structured_content": structured[0],
        }
        if meta is not None and original_meta is not None:
            rebuilt_meta = dict(meta)
            rebuilt_meta.update(
                {k: v for k, v in original_meta.items() if k in excluded_meta_keys}
            )
            update["meta"] = rebuilt_meta
        return result.model_copy(update=update)

    return units, setters, finalize
