"""Client that drives the EHR demo server over streamable-http.

Start the server first (see ``ehr_server.py``), then run::

    python examples/ehr_patient_context/ehr_client.py

:func:`run_demo` performs each step in order and returns the lines it prints,
so a test can assert on them without capturing stdout.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.shared.exceptions import MCPError

from hmcp.client.client_connector import HMCPClientConnector
from hmcp.shared.constants import GUARDRAILS_EXTENSION, META_REPORT

from demo_auth import SERVER_URL

# DEMO ONLY: the bearer token for synthetic Patient/test-0001.
DEMO_TOKEN = "demo-token-test-0001"

# The example-local guardrail the client asks the server to enforce.
PHI_PATTERN_GUARDRAIL = "com.example.ehr-demo/phi-patterns"


def _guardrail_summary(result: Any) -> str:
    """Summarise the guardrail report attached to a tool result's ``_meta``.

    The guardrails extension writes ``_meta[GUARDRAILS_EXTENSION][META_REPORT]``
    as a list of entries, each with the guardrail ``id``, the ``stage`` it ran
    at ("input" or "output") and its ``verdict``. A guardrail that runs at both
    stages appears twice, once per stage.
    """

    guardrails = (result.meta or {}).get(GUARDRAILS_EXTENSION) or {}
    report = guardrails.get(META_REPORT) or []
    if not report:
        return "no guardrail report"
    return ", ".join(
        f"{entry['id']}/{entry['stage']}={entry['verdict']}" for entry in report
    )


async def run_demo(connector: HMCPClientConnector) -> list[str]:
    """Connect, run every demo step, disconnect, and return the printed lines."""

    lines: list[str] = []

    def emit(line: str) -> None:
        lines.append(line)

    # The connector authenticates with the bearer token given in its headers.
    await connector.connect()
    try:
        # 1. Open a patient context. The server mints an opaque handle bound to
        #    this token's principal; the host (this client), not the model,
        #    decides which patient to open.
        handle = await connector.open_patient_context("Patient/test-0001")
        emit(f"1. opened context {handle.context_id} for Patient/test-0001")

        # 2. A patient-scoped tool with the handle attached.
        result = await connector.call_tool("get_medications", patient_context=handle)
        emit(f"2. get_medications: {result.content[0].text}")

        # 3. A PHI-bearing result, with a required guardrail. require_guardrails
        #    makes the connector verify the guardrail reported on this call.
        result = await connector.call_tool(
            "get_demographics",
            patient_context=handle,
            require_guardrails=[PHI_PATTERN_GUARDRAIL],
        )
        # Print the returned text exactly as the server sent it.
        emit(f"3. get_demographics: {result.content[0].text}")
        emit(f"3. guardrail report: {_guardrail_summary(result)}")

        # 4. Writing back a redaction placeholder is blocked at the input
        #    stage: the server returns an error result, not a normal one.
        result = await connector.call_tool(
            "add_note", {"text": "[REDACTED:NAME] called"}, patient_context=handle
        )
        emit(f"4. add_note blocked: {result.content[0].text}")

        # 5. This tool returns the wrong patient; consistency blocks it.
        result = await connector.call_tool("lookup_with_bug", patient_context=handle)
        emit(f"5. lookup_with_bug blocked: {result.content[0].text}")

        # 6. A population tool with a single-patient token is denied. The
        #    server rejects the call with an MCPError carrying an HMCP code.
        try:
            await connector.call_tool("population_count")
        except MCPError as exc:
            emit(f"6. population_count denied: MCPError {exc.code} {exc.message}")

        # 7. The token is scoped to test-0001, so opening test-0002 is denied.
        try:
            await connector.open_patient_context("Patient/test-0002")
        except MCPError as exc:
            emit(f"7. open Patient/test-0002 denied: MCPError {exc.code} {exc.message}")

        # 8. A patient-scoped tool with no handle at all is refused.
        try:
            await connector.call_tool("get_medications")
        except MCPError as exc:
            emit(
                "8. get_medications without context: "
                f"MCPError {exc.code} {exc.message}"
            )

        # 9. Close the context, then reuse the now-closed handle.
        await connector.close_patient_context(handle)
        emit("9. closed context")
        try:
            await connector.call_tool("get_medications", patient_context=handle)
        except MCPError as exc:
            emit(f"9. get_medications after close: MCPError {exc.code} {exc.message}")
    finally:
        await connector.cleanup()

    return lines


async def main() -> None:
    connector = HMCPClientConnector(
        SERVER_URL,
        headers={"Authorization": f"Bearer {DEMO_TOKEN}"},
    )
    for line in await run_demo(connector):
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
