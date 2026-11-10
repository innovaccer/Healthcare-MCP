"""Smoke test for the quickstart example and its README blocks.

The two Python fences in the top-level README (under "### A minimal server" and
"### A minimal client") are the source of truth for this example: the test
asserts each file equals its fence (ignoring only a trailing newline), then
drives the server and client end to end over an in-process
``httpx2.ASGITransport``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import httpx2

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = ROOT / "examples" / "quickstart"
README = ROOT / "README.md"

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


def _fence_under(heading: str) -> str:
    """Return the text of the first python fence after ``heading`` in README.md."""

    text = README.read_text(encoding="utf-8")
    start = text.index(heading)
    match = re.search(r"```python\n(.*?)\n```", text[start:], re.DOTALL)
    assert match is not None, f"no python fence under {heading!r}"
    return match.group(1)


_server = _load_example_module("quickstart_server", "server.py")
_client = _load_example_module("quickstart_client", "client.py")


def test_quickstart_files_match_readme():
    server_file = (EXAMPLE_DIR / "server.py").read_text(encoding="utf-8")
    client_file = (EXAMPLE_DIR / "client.py").read_text(encoding="utf-8")

    assert _fence_under("### A minimal server") == server_file.rstrip("\n")
    assert _fence_under("### A minimal client") == client_file.rstrip("\n")


@contextlib.asynccontextmanager
async def _running(server):
    """Run the server's session manager and yield an in-process transport."""

    app = server.streamable_http_app()
    transport = httpx2.ASGITransport(app=app)
    async with server.session_manager.run():
        yield transport


async def test_quickstart_example_runs(capsys):
    from hmcp.client.client_connector import HMCPClientConnector

    async with _running(_server.server) as transport:
        connector = HMCPClientConnector(
            _server.SERVER_URL,
            headers={"Authorization": f"Bearer {_server.DEMO_TOKEN}"},
            transport=transport,
        )
        async with connector:
            await _client.run(connector)

    printed = capsys.readouterr().out
    # The client prints the tool's text content and one line per guardrail
    # report entry, not the raw CallToolResult repr.
    assert "Patient/test-0001" in printed
    assert (
        "guardrail report: "
        "com.innovaccer.hmcp.guardrails/patient-context-consistency/input=pass"
        in printed
    )
    assert (
        "guardrail report: "
        "com.innovaccer.hmcp.guardrails/patient-context-consistency/output=pass"
        in printed
    )
    assert "digest" not in printed
    assert "durationMs" not in printed
