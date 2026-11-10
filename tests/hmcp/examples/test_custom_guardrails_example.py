"""Smoke test for the custom-guardrails example.

It imports the real ``build_server`` and ``run_demo`` from
``examples/custom_guardrails/`` and drives them end to end over streamable-http
using an in-process ``httpx2.ASGITransport`` (so there is no network and no
server process). The point is to keep the example runnable as the SDK and HMCP
evolve, and to prove each of the three custom guardrails still fires.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import httpx2

# Add the example directory so ``demo_auth`` imports as a top-level module,
# exactly as it does when the scripts run.
EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "custom_guardrails"
sys.path.insert(0, str(EXAMPLE_DIR))


def _load_example_module(name: str, filename: str) -> ModuleType:
    """Load an example module from its file under a unique name.

    The example's modules are named ``server`` and ``client``, which clash with
    the ``tests/hmcp/server`` and ``tests/hmcp/client`` packages pytest puts on
    ``sys.path`` when it collects the full suite. Loading them by path keeps
    both names intact for anyone running the example as a script.
    """

    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_client = _load_example_module("custom_guardrails_client", "client.py")
_server = _load_example_module("custom_guardrails_server", "server.py")

from demo_auth import SERVER_URL  # noqa: E402

MEMBER_ID_PATTERN_ID = _client.MEMBER_ID_PATTERN_ID
NAME_DEID_ID = _client.NAME_DEID_ID
PROMPT_INJECTION_ID = _client.PROMPT_INJECTION_ID
build_server = _server.build_server
run_demo = _client.run_demo


@contextlib.asynccontextmanager
async def _running(server):
    """Run the server's session manager and yield an in-process transport."""

    app = server.streamable_http_app()
    transport = httpx2.ASGITransport(app=app)
    async with server.session_manager.run():
        yield transport


async def test_custom_guardrails_example_demo():
    server = build_server()
    async with _running(server) as transport:
        lines = await run_demo(
            SERVER_URL,
            headers={"Authorization": "Bearer demo-token-test-0001"},
            transport=transport,
        )

    joined = "\n".join(lines)
    # Step 2 is the get_member_record call and its guardrail report.
    step2 = "\n".join(line for line in lines if line.startswith("2."))
    # Step 3 is the blocked submit_note call and its guardrail report.
    step3 = "\n".join(line for line in lines if line.startswith("3."))

    # The pattern guardrail redacted the member ID by BOTH the JSON key path
    # (the member_id field) and the text pattern (inside the free-text summary).
    assert "[REDACTED:OTHER_ID]" in step2
    assert '"member_id": "[REDACTED:OTHER_ID]"' in step2
    assert f"{MEMBER_ID_PATTERN_ID}/output=redacted" in step2

    # The deid guardrail pseudonymized the synthetic name.
    assert f"{NAME_DEID_ID}/output=pseudonymized" in step2
    assert "[NAME_1]" in step2

    # The prompt guardrail blocked the injection input, and the result is an
    # error result naming the guardrail that blocked it.
    assert "is_error=True" in step3
    assert "Blocked by guardrail" in step3
    assert PROMPT_INJECTION_ID in step3
    assert f"{PROMPT_INJECTION_ID}/input=blocked" in step3

    # No synthetic sensitive value appears anywhere in the captured output.
    for secret in ("MBR-10000001", "Jordan Fields", "Priya Nair"):
        assert secret not in joined
