"""ASGI-layer tests: PHI in a tool result's ``_meta`` is inspected.

A tool can return a ``CallToolResult`` with arbitrary ``_meta``. Guardrails must
inspect it exactly like structured content, and a tool must never be able to
forge the guardrails report. These tests drive the real streamable-http app
through an in-process ``httpx2.ASGITransport`` (no network, no server process).

Synthetic data only.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, Callable

import httpx2
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from hmcp.server.guardrails import GuardrailsConfig, GuardrailsExtension
from hmcp.shared.constants import (
    GUARDRAILS_EXTENSION,
    META_REPORT,
    PATIENT_CONTEXT_EXTENSION,
)
from hmcp.shared.guardrails.definition import (
    ActionSpec,
    AppliesTo,
    GuardrailDefinition,
    PatternEvaluatorSpec,
)
from hmcp.shared.guardrails.reference import PHI_IDENTIFIER_TYPES, phi_redaction
from hmcp.shared.types import ToolPatientScope

PHI_REDACTION = phi_redaction()
_SERVER_URL = "http://127.0.0.1:8050/mcp"
_SSN = "123-45-6789"


class _FakeLLM:
    """A prompt evaluator that finds nothing; the pattern step does the work."""

    async def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        return '{"findings": []}'


def _scope_for(_name: str) -> ToolPatientScope:
    return "none"


@contextlib.asynccontextmanager
async def _client(server: MCPServer):
    transport = httpx2.ASGITransport(app=server.streamable_http_app())
    async with server.session_manager.run():
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:8050"
        ) as http:
            async with Client(
                streamable_http_client(_SERVER_URL, http_client=http)
            ) as client:
                yield client


def _redaction_server() -> MCPServer:
    extension = GuardrailsExtension(
        GuardrailsConfig(
            definitions=[PHI_REDACTION],
            llm=_FakeLLM(),
            mandatory=[PHI_REDACTION.id],
        ),
        _scope_for,
    )
    return MCPServer("t", extensions=[extension])


def _report(result: Any) -> list[dict[str, Any]]:
    return (result.meta or {})[GUARDRAILS_EXTENSION][META_REPORT]


async def _run(result_factory: Callable[[], CallToolResult]) -> CallToolResult:
    server = _redaction_server()

    @server.tool()
    async def tool() -> CallToolResult:
        return result_factory()

    async with _client(server) as client:
        return await client.call_tool("tool", {})


# ---------------------------------------------------------------------------
# PHI in _meta is redacted, and the report reflects it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"note": f"ssn {_SSN}"},  # flat string
        {"outer": {"note": f"ssn {_SSN}"}},  # nested in a dict
        {"ssn": _SSN},  # under a json_keys key
    ],
)
async def test_phi_in_meta_is_redacted_like_structured_content(
    payload: dict[str, Any],
) -> None:
    meta_result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")], meta=payload
        )
    )
    structured_result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")], structured_content=payload
        )
    )

    # The engine's report for the equivalent structured content is computed, not
    # assumed: the meta run must agree with it.
    expected = _report(structured_result)[0]
    got = _report(meta_result)[0]
    assert got["verdict"] == expected["verdict"]
    assert got["findings"] == expected["findings"]
    assert got["findings"] >= 1

    # And the PHI never reaches the client in _meta.
    assert _SSN not in json.dumps(meta_result.meta)


async def test_flat_meta_string_value_is_redacted() -> None:
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")],
            meta={"note": f"ssn {_SSN}"},
        )
    )

    assert result.meta["note"] == "ssn [REDACTED:SSN]"


async def test_nested_meta_dict_value_is_redacted() -> None:
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")],
            meta={"outer": {"note": f"ssn {_SSN}"}},
        )
    )

    assert result.meta["outer"]["note"] == "ssn [REDACTED:SSN]"


async def test_meta_json_keys_key_is_redacted() -> None:
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")],
            meta={"ssn": _SSN},
        )
    )

    assert result.meta["ssn"] == "[REDACTED:SSN]"


async def test_phi_under_patient_context_meta_key_is_redacted() -> None:
    payload = {PATIENT_CONTEXT_EXTENSION: {"note": f"ssn {_SSN}"}}
    meta_result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")], meta=payload
        )
    )
    structured_result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")], structured_content=payload
        )
    )

    expected = _report(structured_result)[0]
    got = _report(meta_result)[0]
    assert got["verdict"] == expected["verdict"]
    assert got["findings"] == expected["findings"]
    assert got["findings"] >= 1

    assert meta_result.meta[PATIENT_CONTEXT_EXTENSION]["note"] == "ssn [REDACTED:SSN]"
    assert _SSN not in json.dumps(meta_result.meta)


# ---------------------------------------------------------------------------
# Non-JSON values are normalised to the wire form before inspection
# ---------------------------------------------------------------------------


_ODD_VALUES = [
    ("tuple", lambda: {"t": ("a", f"ssn {_SSN}")}),
    ("set", lambda: {"s": {f"ssn {_SSN}"}}),
    ("frozenset", lambda: {"f": frozenset({f"ssn {_SSN}"})}),
    ("bytes", lambda: {"b": f"ssn {_SSN}".encode()}),
]


@pytest.mark.parametrize("label,factory", _ODD_VALUES)
async def test_odd_meta_value_is_redacted(label: str, factory: Callable) -> None:
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")], meta=factory()
        )
    )

    entry = _report(result)[0]
    assert entry["verdict"] != "pass"
    assert entry["findings"] >= 1
    assert _SSN not in json.dumps(result.meta)


@pytest.mark.parametrize("label,factory", _ODD_VALUES)
async def test_odd_structured_content_value_is_redacted(
    label: str, factory: Callable
) -> None:
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content=factory(),
        )
    )

    entry = _report(result)[0]
    assert entry["verdict"] != "pass"
    assert entry["findings"] >= 1
    assert _SSN not in json.dumps(result.structured_content)


def _deep(depth: int) -> Any:
    value: Any = f"ssn {_SSN}"
    for _ in range(depth):
        value = [value]
    return value


async def test_deep_meta_fails_closed_with_guardrail_block() -> None:
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")], meta={"d": _deep(400)}
        )
    )

    assert result.is_error is True
    assert _report(result)[0]["verdict"] == "error"


async def test_normal_meta_and_structured_content_unchanged() -> None:
    meta = {"note": "clean text", "n": 5}
    structured = {"rows": [{"v": "clean"}], "count": 2}
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")],
            meta=meta,
            structured_content=structured,
        )
    )

    assert {key: result.meta[key] for key in meta} == meta
    assert result.structured_content == structured
    assert _report(result)[0]["verdict"] == "pass"


# ---------------------------------------------------------------------------
# A tool cannot forge the guardrails report
# ---------------------------------------------------------------------------


async def test_tool_supplied_guardrails_meta_is_stripped_when_none_applies() -> None:
    definition = GuardrailDefinition(
        id="com.example.test/other-tool-only",
        version="1.0.0",
        title="only applies to another tool",
        stages=("output",),
        applies_to=AppliesTo(tools=("other_tool",)),
        evaluator=PatternEvaluatorSpec(
            kind="pattern", identifier_types=PHI_IDENTIFIER_TYPES
        ),
        action=ActionSpec(on_finding="redact"),
    )
    extension = GuardrailsExtension(
        GuardrailsConfig(definitions=[definition], mandatory=[definition.id]),
        _scope_for,
    )
    server = MCPServer("t", extensions=[extension])

    @server.tool()
    async def tool() -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            meta={
                GUARDRAILS_EXTENSION: {
                    META_REPORT: [{"forged": True}],
                    "note": f"ssn {_SSN}",
                },
                "note": "hello",
            },
        )

    async with _client(server) as client:
        result = await client.call_tool("tool", {})

    assert GUARDRAILS_EXTENSION not in (result.meta or {})
    assert result.meta["note"] == "hello"
    assert _SSN not in json.dumps(result.meta)


async def test_phi_under_guardrails_meta_key_is_replaced_when_applied() -> None:
    result = await _run(
        lambda: CallToolResult(
            content=[TextContent(type="text", text="ok")],
            meta={GUARDRAILS_EXTENSION: {"note": f"ssn {_SSN}"}},
        )
    )

    # The extension replaces the tool's value with the real report, so the PHI
    # the tool hid under the guardrails key never reaches the client.
    assert _SSN not in json.dumps(result.meta)
    report = _report(result)
    assert report
    assert "note" not in json.dumps(report)


# ---------------------------------------------------------------------------
# Control: a result with no _meta still works
# ---------------------------------------------------------------------------


async def test_result_without_meta_is_unchanged() -> None:
    result = await _run(
        lambda: CallToolResult(content=[TextContent(type="text", text="ok")])
    )

    assert result.content[0].text == "ok"
    entry = _report(result)[0]
    assert entry["verdict"] == "pass"
    assert entry["findings"] == 0
