"""Tests for the guardrails MCP extension. Synthetic data only.

Uses a plain ``MCPServer`` with the extension plus the in-memory ``Client``;
``scope_for`` is backed by a dict so no ``hmcp.server.hmcp_server`` import is
needed. Fake LLM/deid providers only; no network.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import pytest
from mcp.client import Client
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError

from hmcp.server.guardrails import GuardrailsConfig, GuardrailsExtension
from hmcp.shared.constants import (
    GUARDRAILS_EXTENSION,
    GUARDRAILS_REVISION,
    GUARDRAIL_RESOURCE_PREFIX,
    META_REPORT,
    META_REQUIRE,
)
from hmcp.shared.errors import HMCPErrorCode
from hmcp.shared.guardrails.definition import (
    ActionSpec,
    AppliesTo,
    ClassifierEvaluatorSpec,
    GuardrailDefinition,
    PatternEvaluatorSpec,
    PhiCategory,
    PromptEvaluatorSpec,
)
from hmcp.shared.guardrails.reference import (
    PHI_IDENTIFIER_TYPES,
    phi_redaction,
    redaction_placeholder_writeback,
)
from hmcp.shared.types import ToolPatientScope

PHI_REDACTION = phi_redaction()
REPORT_KEYS = {
    "id",
    "version",
    "digest",
    "stage",
    "verdict",
    "evaluator",
    "findings",
    "durationMs",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope_for(
    *,
    default: ToolPatientScope = "none",
    overrides: dict[str, ToolPatientScope] | None = None,
) -> Callable[[str], ToolPatientScope]:
    table = overrides or {}

    def scope_for(name: str) -> ToolPatientScope:
        return table.get(name, default)

    return scope_for


class FakeLLM:
    """A fake prompt-evaluator LLM over a fixed finding list."""

    def __init__(
        self, findings: Sequence[dict[str, str]] = (), *, raises: bool = False
    ) -> None:
        self._findings = list(findings)
        self._raises = raises

    async def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        if self._raises:
            raise RuntimeError("fake llm down")
        found = [item for item in self._findings if item["text"] in user]
        return json.dumps({"findings": found})


def _pattern_definition(
    *,
    gid: str,
    stages: tuple[str, ...] = ("input",),
    action: str = "redact",
    applies_tools: tuple[str, ...] | None = None,
) -> GuardrailDefinition:
    return GuardrailDefinition(
        id=gid,
        version="1.0.0",
        title="test pattern",
        stages=stages,
        applies_to=AppliesTo(tools=applies_tools) if applies_tools else AppliesTo(),
        evaluator=PatternEvaluatorSpec(
            kind="pattern", identifier_types=PHI_IDENTIFIER_TYPES
        ),
        action=ActionSpec(on_finding=action),
    )


def _prompt_definition(*, gid: str, failure_mode: str) -> GuardrailDefinition:
    return GuardrailDefinition(
        id=gid,
        version="1.0.0",
        title="test prompt",
        stages=("input",),
        evaluator=PromptEvaluatorSpec(
            kind="prompt",
            system="Find PHI in <c-{{nonce}}>: {{categories}}",
            user="<c-{{nonce}}>{{content}}</c-{{nonce}}>",
            categories=(PhiCategory.NAME,),
        ),
        action=ActionSpec(on_finding="redact"),
        failure_mode=failure_mode,
    )


class FakeClassifier:
    """A fake classifier provider over a fixed verdict; records its requests."""

    def __init__(self, *, flagged: bool = False, raises: bool = False) -> None:
        self._flagged = flagged
        self._raises = raises
        self.requests: list[Any] = []

    async def classify(self, request: Any) -> Any:
        self.requests.append(request)
        if self._raises:
            raise RuntimeError("classifier down")
        return {"flagged": self._flagged}


def _classifier_definition(
    *,
    gid: str,
    stages: tuple[str, ...] = ("input",),
    failure_mode: str = "closed",
) -> GuardrailDefinition:
    return GuardrailDefinition(
        id=gid,
        version="1.0.0",
        title="test classifier",
        stages=stages,
        evaluator=ClassifierEvaluatorSpec(kind="classifier", classifier="toxicity"),
        action=ActionSpec(on_finding="block"),
        failure_mode=failure_mode,
    )


def _report(result: Any) -> list[dict[str, Any]]:
    return result.meta[GUARDRAILS_EXTENSION][META_REPORT]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_duplicate_ids_rejected() -> None:
    definition = _pattern_definition(gid="com.example.test/dup")
    with pytest.raises(ValueError):
        GuardrailsExtension(
            GuardrailsConfig(definitions=[definition, definition]), _scope_for()
        )


def test_mandatory_id_not_configured_rejected() -> None:
    definition = _pattern_definition(gid="com.example.test/one")
    with pytest.raises(ValueError):
        GuardrailsExtension(
            GuardrailsConfig(
                definitions=[definition], mandatory=["com.example.test/missing"]
            ),
            _scope_for(),
        )


def test_prompt_definition_without_llm_rejected() -> None:
    with pytest.raises(ValueError):
        GuardrailsExtension(GuardrailsConfig(definitions=[PHI_REDACTION]), _scope_for())


def test_duplicate_short_names_rejected() -> None:
    first = _pattern_definition(gid="com.example.test/shared")
    second = _pattern_definition(gid="com.example.other/shared")
    assert first.id != second.id
    assert first.short_name == second.short_name
    with pytest.raises(ValueError):
        GuardrailsExtension(GuardrailsConfig(definitions=[first, second]), _scope_for())


# ---------------------------------------------------------------------------
# Settings and definition resources
# ---------------------------------------------------------------------------


async def test_settings_and_definition_resource_round_trip() -> None:
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[PHI_REDACTION], llm=FakeLLM(), mandatory=[PHI_REDACTION.id]
        ),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def demo() -> str:
        return "ok"

    async with Client(server) as client:
        capabilities = client.server_capabilities
        settings = capabilities.extensions[GUARDRAILS_EXTENSION]
        assert settings["revision"] == GUARDRAILS_REVISION
        assert len(settings["available"]) == 1
        entry = settings["available"][0]
        assert entry["id"] == PHI_REDACTION.id
        assert entry["version"] == PHI_REDACTION.version
        assert entry["digest"] == PHI_REDACTION.digest
        assert entry["stages"] == ["output"]
        assert entry["definition"] == (
            GUARDRAIL_RESOURCE_PREFIX + PHI_REDACTION.short_name
        )
        assert entry["enforcedBy"] == "server"
        assert entry["mandatory"] is True

        read = await client.read_resource(entry["definition"])

    text = read.contents[0].text
    round_tripped = GuardrailDefinition.from_json(text)
    assert round_tripped.digest == entry["digest"]
    assert round_tripped.id == PHI_REDACTION.id


# ---------------------------------------------------------------------------
# require validation (before the tool runs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("require", "has_guardrail_data"),
    [
        (["com.example.test/unknown"], True),
        ([PHI_REDACTION.id + "@2"], True),
        ([PHI_REDACTION.id + "@x"], True),
        ([PHI_REDACTION.id + "@" + "9" * 5000], True),
        # A string is iterable: without the list-type guard it would be walked
        # char by char and rejected as an unknown id (with guardrail data), so
        # asserting the ABSENCE of that data distinguishes the two paths.
        ("not-a-list", False),
    ],
)
async def test_bad_require_is_unavailable_and_tool_not_called(
    require: Any, has_guardrail_data: bool
) -> None:
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[PHI_REDACTION], llm=FakeLLM()), _scope_for()
    )
    server = MCPServer("t", extensions=[extension])
    calls: list[str] = []

    @server.tool()
    async def demo(text: str) -> str:
        calls.append(text)
        return "ok"

    async with Client(server) as client:
        meta = {GUARDRAILS_EXTENSION: {META_REQUIRE: require}}
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("demo", {"text": "x"}, meta=meta)

    assert excinfo.value.code == int(HMCPErrorCode.GUARDRAIL_UNAVAILABLE)
    assert ("guardrail" in (excinfo.value.data or {})) is has_guardrail_data
    assert calls == []


@pytest.mark.parametrize("container", ["oops", ["x"], 5, 3.14])
async def test_non_dict_guardrail_meta_is_unavailable_and_tool_not_called(
    container: Any,
) -> None:
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[PHI_REDACTION], llm=FakeLLM()), _scope_for()
    )
    server = MCPServer("t", extensions=[extension])
    calls: list[str] = []

    @server.tool()
    async def demo(text: str) -> str:
        calls.append(text)
        return "ok"

    async with Client(server) as client:
        meta = {GUARDRAILS_EXTENSION: container}
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("demo", {"text": "x"}, meta=meta)

    assert excinfo.value.code == int(HMCPErrorCode.GUARDRAIL_UNAVAILABLE)
    assert calls == []


# ---------------------------------------------------------------------------
# Mandatory guardrails
# ---------------------------------------------------------------------------


async def test_mandatory_writeback_blocks_redaction_placeholder() -> None:
    definition = redaction_placeholder_writeback()
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[definition], mandatory=[definition.id]),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])
    calls: list[str] = []

    @server.tool()
    async def note(note: str) -> str:
        calls.append(note)
        return "saved"

    async with Client(server) as client:
        result = await client.call_tool("note", {"note": "[REDACTED:NAME]"})

    assert result.is_error is True
    assert result.content[0].text == f"Blocked by guardrail {definition.id}"
    assert _report(result)[0]["verdict"] == "blocked"
    assert calls == []


async def test_input_redaction_mutates_params_in_place_reaching_tool() -> None:
    """IN-PLACE GUARD: proves the SDK passes the same params object to the tool.

    The interceptor mutates ``params.arguments`` in place; the tool must receive
    the redacted value. If a future SDK copies ``params`` before the handler,
    this test MUST go red.
    """
    definition = _pattern_definition(
        gid="com.example.test/ssn-redact", stages=("input",), action="redact"
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[definition], mandatory=[definition.id]),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])
    seen: list[str] = []

    @server.tool()
    async def record(text: str) -> str:
        seen.append(text)
        return "ok"

    async with Client(server) as client:
        result = await client.call_tool("record", {"text": "SSN 900-12-3456"})

    assert seen == ["SSN [REDACTED:SSN]"]
    assert result.is_error is False
    assert _report(result)[0]["verdict"] == "redacted"


async def test_required_phi_redaction_cascade() -> None:
    llm = FakeLLM([{"text": "Testpatient Alpha", "category": "NAME"}])
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[PHI_REDACTION], llm=llm), _scope_for()
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def chart() -> str:
        return "Testpatient Alpha SSN 900-12-3456"

    async with Client(server) as client:
        meta = {GUARDRAILS_EXTENSION: {META_REQUIRE: [PHI_REDACTION.id]}}
        result = await client.call_tool("chart", {}, meta=meta)

    text = result.content[0].text
    assert "[REDACTED:NAME]" in text
    assert "[REDACTED:SSN]" in text
    assert "Testpatient Alpha" not in text
    assert "900-12-3456" not in text
    entry = _report(result)[0]
    assert entry["verdict"] == "redacted"
    assert entry["evaluator"] == "cascade(pattern,prompt)"
    assert entry["findings"] >= 2


async def test_output_block_discards_tool_content() -> None:
    definition = _pattern_definition(
        gid="com.example.test/ssn-block", stages=("output",), action="block"
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[definition], mandatory=[definition.id]),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def chart() -> str:
        return "SSN 900-12-3456"

    async with Client(server) as client:
        result = await client.call_tool("chart", {})

    assert result.is_error is True
    assert result.content[0].text == f"Blocked by guardrail {definition.id}"
    assert "900-12-3456" not in result.content[0].text
    entry = _report(result)[0]
    assert entry["verdict"] == "blocked"
    assert entry["stage"] == "output"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_failure_mode_open_reports_error_and_succeeds() -> None:
    definition = _prompt_definition(gid="com.example.test/open", failure_mode="open")
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[definition],
            llm=FakeLLM(raises=True),
            mandatory=[definition.id],
        ),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def chart(text: str) -> str:
        return "ok"

    async with Client(server) as client:
        result = await client.call_tool("chart", {"text": "anything"})

    assert result.is_error is False
    assert result.content[0].text == "ok"
    assert _report(result)[0]["verdict"] == "error"


async def test_failure_mode_closed_blocks_on_error() -> None:
    definition = _prompt_definition(
        gid="com.example.test/closed", failure_mode="closed"
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[definition],
            llm=FakeLLM(raises=True),
            mandatory=[definition.id],
        ),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def chart(text: str) -> str:
        return "ok"

    async with Client(server) as client:
        result = await client.call_tool("chart", {"text": "anything"})

    assert result.is_error is True
    assert _report(result)[0]["verdict"] == "error"


# ---------------------------------------------------------------------------
# Applicability and report shape
# ---------------------------------------------------------------------------


async def test_not_applicable_definition_leaves_no_report() -> None:
    definition = _pattern_definition(
        gid="com.example.test/only-other",
        stages=("input",),
        action="block",
        applies_tools=("other_tool",),
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[definition], mandatory=[definition.id]),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def demo(text: str) -> str:
        return "ok"

    async with Client(server) as client:
        result = await client.call_tool("demo", {"text": "SSN 900-12-3456"})

    assert GUARDRAILS_EXTENSION not in (result.meta or {})


async def test_report_entries_are_camel_case_and_carry_no_content() -> None:
    definition = _pattern_definition(
        gid="com.example.test/ssn-report", stages=("input",), action="redact"
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[definition], mandatory=[definition.id]),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def demo(text: str) -> str:
        return "ok"

    async with Client(server) as client:
        result = await client.call_tool("demo", {"text": "SSN 900-12-3456"})

    entry = _report(result)[0]
    assert "durationMs" in entry
    assert "duration_ms" not in entry
    assert set(entry) == REPORT_KEYS
    assert all("900-12-3456" not in str(value) for value in entry.values())


# ---------------------------------------------------------------------------
# Classifier guardrails
# ---------------------------------------------------------------------------


async def test_classifier_input_flagged_blocks_before_tool_runs() -> None:
    definition = _classifier_definition(
        gid="com.example.test/classifier-input", stages=("input",)
    )
    classifier = FakeClassifier(flagged=True)
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[definition],
            classifier=classifier,
            mandatory=[definition.id],
        ),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])
    calls: list[str] = []

    @server.tool()
    async def note(text: str) -> str:
        calls.append(text)
        return "saved"

    async with Client(server) as client:
        result = await client.call_tool("note", {"text": "anything"})

    assert result.is_error is True
    assert result.content[0].text == f"Blocked by guardrail {definition.id}"
    entry = _report(result)[0]
    assert entry["evaluator"] == "classifier"
    assert entry["verdict"] == "blocked"
    assert calls == []


async def test_classifier_output_flagged_blocks_after_tool_runs() -> None:
    definition = _classifier_definition(
        gid="com.example.test/classifier-output", stages=("output",)
    )
    classifier = FakeClassifier(flagged=True)
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[definition],
            classifier=classifier,
            mandatory=[definition.id],
        ),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])
    calls: list[str] = []

    @server.tool()
    async def chart() -> str:
        calls.append("ran")
        return "anything"

    async with Client(server) as client:
        result = await client.call_tool("chart", {})

    assert calls == ["ran"]
    assert result.is_error is True
    assert result.content[0].text == f"Blocked by guardrail {definition.id}"


async def test_classifier_raising_provider_closed_blocks() -> None:
    definition = _classifier_definition(
        gid="com.example.test/classifier-closed",
        stages=("input",),
        failure_mode="closed",
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[definition],
            classifier=FakeClassifier(raises=True),
            mandatory=[definition.id],
        ),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def chart(text: str) -> str:
        return "ok"

    async with Client(server) as client:
        result = await client.call_tool("chart", {"text": "anything"})

    assert result.is_error is True
    assert _report(result)[0]["verdict"] == "error"


async def test_classifier_raising_provider_open_returns_tool_result() -> None:
    definition = _classifier_definition(
        gid="com.example.test/classifier-open",
        stages=("input",),
        failure_mode="open",
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[definition],
            classifier=FakeClassifier(raises=True),
            mandatory=[definition.id],
        ),
        _scope_for(),
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def chart(text: str) -> str:
        return "ok"

    async with Client(server) as client:
        result = await client.call_tool("chart", {"text": "anything"})

    assert result.is_error is False
    assert result.content[0].text == "ok"
    assert _report(result)[0]["verdict"] == "error"


async def test_classifier_definition_without_provider_rejected() -> None:
    definition = _classifier_definition(gid="com.example.test/classifier-missing")
    with pytest.raises(
        ValueError, match="classifier evaluator requires a classifier provider"
    ):
        GuardrailsExtension(GuardrailsConfig(definitions=[definition]), _scope_for())
