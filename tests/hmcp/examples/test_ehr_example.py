"""Smoke test for the EHR patient-context + guardrails example.

It imports the real ``build_server`` and ``run_demo`` from
``examples/ehr_patient_context/`` and drives them end to end over
streamable-http using an in-process ``httpx2.ASGITransport`` (so there is no
network and no server process). The point is to keep the example runnable as
the SDK and HMCP evolve.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import httpx2

from hmcp.client.client_connector import HMCPClientConnector

# Add the example directory so ``demo_auth``, ``ehr_server`` and ``ehr_client``
# import as top-level modules, exactly as they do when run as scripts.
EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "ehr_patient_context"
sys.path.insert(0, str(EXAMPLE_DIR))

from demo_auth import SERVER_URL  # noqa: E402
from ehr_client import run_demo  # noqa: E402
from ehr_server import build_server  # noqa: E402


@contextlib.asynccontextmanager
async def _running(server):
    """Run the server's session manager and yield an in-process transport."""

    app = server.streamable_http_app()
    transport = httpx2.ASGITransport(app=app)
    async with server.session_manager.run():
        yield transport


async def test_ehr_example_demo():
    server = build_server()
    async with _running(server) as transport:
        connector = HMCPClientConnector(
            SERVER_URL,
            headers={"Authorization": "Bearer demo-token-test-0001"},
            transport=transport,
        )
        lines = await run_demo(connector)

    joined = "\n".join(lines)
    # Step 3 is the get_demographics call and its guardrail report.
    step3 = "\n".join(line for line in lines if line.startswith("3."))

    # Guardrail redaction is visible, and none of the raw step-3 fields leak:
    # the name (caught by its key), the SSN, the date of birth and the MRN.
    assert "[REDACTED:SSN]" in step3
    for raw in ("Testpatient Alpha", "900-12-3456", "1980-01-01", "TEST-0001"):
        assert raw not in step3

    # Report lines carry the stage. The client-required pattern guardrail
    # reported "redacted" at the output stage...
    assert "com.example.ehr-demo/phi-patterns/output=redacted" in step3
    # ...and consistency ran at both stages, so both appear.
    assert (
        "com.innovaccer.hmcp.guardrails/patient-context-consistency/input=pass" in step3
    )
    assert (
        "com.innovaccer.hmcp.guardrails/patient-context-consistency/output=pass"
        in step3
    )

    # Both blocked steps surfaced as error results.
    assert "add_note blocked" in joined
    assert "lookup_with_bug blocked" in joined

    # Each expected HMCP error code appeared, -31005 twice.
    assert joined.count("-31005") == 2
    assert "-31001" in joined
    assert "-31002" in joined
