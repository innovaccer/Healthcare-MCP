"""Guardrails MCP extension for HMCP 0.1.0.

Runs the guardrail runtime inside an MCP :class:`Extension` that intercepts
every ``tools/call``. It enforces the configured mandatory
guardrails, honours a per-request ``require`` list (never skipping a required
guardrail silently) and attaches a report to the result's ``_meta`` so a
guardrail that did not run is visible to the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from mcp.server.extension import Extension, ResourceBinding
from mcp.server.mcpserver.resources.types import TextResource
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent

from hmcp.shared.constants import (
    GUARDRAILS_EXTENSION,
    GUARDRAILS_REVISION,
    GUARDRAIL_RESOURCE_PREFIX,
    META_REPORT,
    META_REQUIRE,
)
from hmcp.shared.errors import HMCPErrorCode, hmcp_error
from hmcp.shared.guardrails import (
    EvaluationContext,
    GuardrailDefinition,
    GuardrailRuntime,
    InMemoryMarkerVault,
    MarkerVault,
)
from hmcp.shared.guardrails.evaluators.deid import DeidentificationModel
from hmcp.shared.guardrails.evaluators.prompt import LLMProvider
from hmcp.shared.types import (
    GuardrailReportEntry,
    GuardrailVerdict,
    ToolPatientScope,
    current_patient_context,
)


@dataclass
class GuardrailsConfig:
    """Everything the guardrails extension needs to run.

    ``mandatory`` lists ids applied to every call they apply to (in addition to
    anything the client requires). ``llm`` and ``deid`` are only needed when a
    definition uses a prompt or deid evaluator; a missing model fails fast at
    extension construction.
    """

    definitions: Sequence[GuardrailDefinition]
    mandatory: Sequence[str] = ()
    llm: LLMProvider | None = None
    deid: DeidentificationModel | None = None
    vault: MarkerVault = field(default_factory=InMemoryMarkerVault)


class GuardrailsExtension(Extension):
    """Enforce guardrail definitions in a ``tools/call`` interceptor."""

    identifier = GUARDRAILS_EXTENSION

    def __init__(
        self, config: GuardrailsConfig, scope_for: Callable[[str], ToolPatientScope]
    ) -> None:
        self._config = config
        self._scope_for = scope_for
        self._definitions = list(config.definitions)
        self._by_id = {definition.id: definition for definition in self._definitions}
        if len(self._by_id) != len(self._definitions):
            raise ValueError("duplicate guardrail id")
        # The short name is the published resource URI; two definitions sharing
        # one would publish one URI whose JSON can disagree with the other's
        # advertised digest.
        short_names = [definition.short_name for definition in self._definitions]
        if len(set(short_names)) != len(short_names):
            raise ValueError("duplicate guardrail short name")
        self._mandatory = frozenset(config.mandatory)
        for mandatory_id in self._mandatory:
            if mandatory_id not in self._by_id:
                raise ValueError(f"mandatory guardrail {mandatory_id!r} is not defined")
        # One runtime per definition; build_evaluator fails fast on a prompt
        # definition without an LLM or a deid definition without a provider.
        self._runtimes = {
            definition.id: GuardrailRuntime(
                definition, llm=config.llm, deid=config.deid, vault=config.vault
            )
            for definition in self._definitions
        }

    def settings(self) -> dict[str, Any]:
        return {
            "revision": GUARDRAILS_REVISION,
            "available": [
                {
                    "id": definition.id,
                    "version": definition.version,
                    "digest": definition.digest,
                    "stages": list(definition.stages),
                    "definition": GUARDRAIL_RESOURCE_PREFIX + definition.short_name,
                    "enforcedBy": "server",
                    "mandatory": definition.id in self._mandatory,
                }
                for definition in self._definitions
            ],
        }

    def resources(self) -> Sequence[ResourceBinding]:
        return [
            ResourceBinding(
                resource=TextResource(
                    uri=GUARDRAIL_RESOURCE_PREFIX + definition.short_name,
                    name=definition.short_name,
                    title=definition.title,
                    mime_type="application/json",
                    text=definition.to_json(),
                )
            )
            for definition in self._definitions
        ]

    async def intercept_tool_call(self, params, ctx, call_next):
        required = self._required(params)
        scope = self._scope_for(params.name)
        applicable = self._applicable(scope, params.name, required)
        patient_context = current_patient_context()
        report: list[GuardrailReportEntry] = []

        for definition in applicable:
            if "input" not in definition.stages:
                continue
            entry = await self._runtimes[definition.id].run_input(
                params.arguments,
                EvaluationContext(patient_context, "input", params.name),
            )
            report.append(entry)
            if self._is_blocking(entry, definition):
                return self._blocked(definition, report)

        result = await call_next(ctx)
        if not isinstance(result, CallToolResult):
            # e.g. MRTR InputRequiredResult: not inspected in v1.
            return result

        for definition in applicable:
            if "output" not in definition.stages:
                continue
            result, entry = await self._runtimes[definition.id].run_output(
                result,
                EvaluationContext(patient_context, "output", params.name),
            )
            report.append(entry)
            if self._is_blocking(entry, definition):
                return self._blocked(definition, report)

        if report:
            result = result.model_copy(
                update={
                    "meta": {
                        **(result.meta or {}),
                        GUARDRAILS_EXTENSION: {
                            META_REPORT: [
                                entry.model_dump(by_alias=True, mode="json")
                                for entry in report
                            ]
                        },
                    }
                }
            )
        elif result.meta and GUARDRAILS_EXTENSION in result.meta:
            # No guardrail applied: a tool must never be able to forge anything
            # under the extension key, so strip any tool-supplied value.
            result = result.model_copy(
                update={
                    "meta": {
                        key: value
                        for key, value in result.meta.items()
                        if key != GUARDRAILS_EXTENSION
                    }
                }
            )
        return result

    def _required(self, params) -> list[str]:
        """Validate and return the requested guardrail ids.

        Rejects a malformed ``require`` value, an unknown id, a non-digit major
        and a major that does not match the definition -- before the tool runs.
        """
        meta = params.meta or {}
        guardrail_meta = meta.get(GUARDRAILS_EXTENSION)
        if guardrail_meta is None:
            return []
        # A non-dict container (str/list/int/...) is a malformed request, not an
        # internal error: reject it as unavailable before the tool runs.
        if not isinstance(guardrail_meta, dict):
            raise hmcp_error(HMCPErrorCode.GUARDRAIL_UNAVAILABLE)
        raw = guardrail_meta.get(META_REQUIRE, [])
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise hmcp_error(HMCPErrorCode.GUARDRAIL_UNAVAILABLE)
        for item in raw:
            base_id, sep, major_text = item.partition("@")
            definition = self._by_id.get(base_id)
            if definition is None:
                raise self._unavailable(item)
            # Cap the length before int(): a very long digit string would raise
            # ValueError from CPython's int-string conversion limit.
            if sep and not (
                major_text.isascii() and major_text.isdigit() and len(major_text) <= 6
            ):
                raise self._unavailable(item)
            if sep and int(major_text) != definition.major:
                raise self._unavailable(item)
        return list(raw)

    def _unavailable(self, item: str) -> MCPError:
        return hmcp_error(
            HMCPErrorCode.GUARDRAIL_UNAVAILABLE,
            data={"guardrail": item[:200]},
        )

    def _applicable(
        self, scope: ToolPatientScope, tool_name: str, required: Sequence[str]
    ) -> list[GuardrailDefinition]:
        required_ids = {item.partition("@")[0] for item in required}
        return [
            definition
            for definition in self._definitions
            if (
                definition.id in self._mandatory
                and definition.applies(scope, tool_name)
            )
            or definition.id in required_ids
        ]

    @staticmethod
    def _is_blocking(
        entry: GuardrailReportEntry, definition: GuardrailDefinition
    ) -> bool:
        if entry.verdict is GuardrailVerdict.BLOCKED:
            return True
        return (
            entry.verdict is GuardrailVerdict.ERROR
            and definition.failure_mode == "closed"
        )

    @staticmethod
    def _blocked(
        definition: GuardrailDefinition, report: Sequence[GuardrailReportEntry]
    ) -> CallToolResult:
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"Blocked by guardrail {definition.id}")
            ],
            is_error=True,
            meta={
                GUARDRAILS_EXTENSION: {
                    META_REPORT: [
                        entry.model_dump(by_alias=True, mode="json") for entry in report
                    ]
                }
            },
        )
