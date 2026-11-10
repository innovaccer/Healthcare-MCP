import asyncio
from typing import Any

from hmcp.client.client_connector import HMCPClientConnector
from hmcp.shared.constants import GUARDRAILS_EXTENSION, META_REPORT

DEMO_TOKEN = "demo-token-test-0001"


def _report_entries(result: Any) -> list[str]:
    guardrails = (result.meta or {}).get(GUARDRAILS_EXTENSION) or {}
    report = guardrails.get(META_REPORT) or []
    return [f"{e['id']}/{e['stage']}={e['verdict']}" for e in report]


async def run(connector: HMCPClientConnector) -> None:
    handle = await connector.open_patient_context(
        "Patient/test-0001", purpose_of_use="TREAT"
    )
    try:
        result = await connector.call_tool(
            "get_observations",
            patient_context=handle,
            require_guardrails=[
                "com.innovaccer.hmcp.guardrails/patient-context-consistency"
            ],
        )
        print(result.content[0].text)
        for entry in _report_entries(result):
            print(f"guardrail report: {entry}")
    finally:
        await connector.close_patient_context(handle)


async def main() -> None:
    async with HMCPClientConnector(
        url="http://127.0.0.1:8050/mcp",
        headers={"Authorization": f"Bearer {DEMO_TOKEN}"},
    ) as connector:
        await run(connector)


if __name__ == "__main__":
    asyncio.run(main())
