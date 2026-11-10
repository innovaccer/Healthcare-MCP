"""Tests for :class:`hmcp.client.client_connector.HMCPClientConnector`.

The connector is exercised end to end against a plain
``mcp.server.mcpserver.MCPServer`` served over ``streamable_http_app()`` with an
in-process ``httpx2.ASGITransport`` passed as the connector's ``transport``
seam. All values are synthetic.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import httpx2
import mcp.types as types
import pytest
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import Context, MCPServer
from pydantic import BaseModel

from hmcp.client.client_connector import HMCPClientConnector
from hmcp.shared.constants import GUARDRAILS_EXTENSION, PATIENT_CONTEXT_EXTENSION

BASE = "http://127.0.0.1:8000"
MCP_URL = f"{BASE}/mcp"


# ---------------------------------------------------------------------------
# Test server
# ---------------------------------------------------------------------------


class OpenParams(types.RequestParams):
    patient: str


class EchoExtension(Extension):
    """A custom extension method reachable through ``send_request``."""

    identifier = "com.example/hmcp-echo"

    def settings(self) -> dict[str, Any]:
        return {"revision": "1"}

    def methods(self):
        async def handler(ctx, params: OpenParams):
            return {"contextId": "pc_test", "patient": params.patient}

        return [
            MethodBinding(
                method="hmcp/patientContext/open",
                params_type=OpenParams,
                handler=handler,
            )
        ]


class OpenResult(BaseModel):
    contextId: str
    patient: str
    model_config = {"extra": "allow"}


def build_server(**kwargs: Any) -> tuple[MCPServer, dict[str, Any]]:
    """Build an MCPServer with a tool, resource, prompt and extension method."""
    server = MCPServer("c3-test", extensions=[EchoExtension()], **kwargs)
    seen: dict[str, Any] = {}

    @server.tool()
    async def echo(text: str, ctx: Context) -> str:
        seen["meta"] = dict(ctx.request_context.meta or {})
        return f"echo:{text}"

    @server.resource("test://greeting")
    def greeting() -> str:
        return "hello resource"

    @server.prompt()
    def greet(name: str) -> str:
        return f"Hello, {name}"

    return server, seen


@contextlib.asynccontextmanager
async def running(server: MCPServer):
    """Run the server's session manager and yield an in-process transport."""
    app = server.streamable_http_app()
    transport = httpx2.ASGITransport(app=app)
    async with server.session_manager.run():
        yield transport


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


async def test_connect_list_tools_and_call_tool():
    server, seen = build_server()
    async with running(server) as transport:
        connector = HMCPClientConnector(url=MCP_URL, transport=transport)
        await connector.connect()
        try:
            assert connector.connected is True
            assert connector.server_info is not None
            assert connector.server_info.name == "c3-test"

            tools = await connector.list_tools()
            assert [t.name for t in tools] == ["echo"]

            result = await connector.call_tool("echo", {"text": "hi"}, meta={"k": "v"})
            assert isinstance(result, types.CallToolResult)
            assert result.content[0].text == "echo:hi"
            # The request meta is visible to the tool via ctx.request_context.meta.
            assert seen["meta"]["k"] == "v"
        finally:
            await connector.cleanup()
        assert connector.connected is False


async def test_resources_and_prompts_roundtrip():
    server, _ = build_server()
    async with running(server) as transport:
        async with HMCPClientConnector(url=MCP_URL, transport=transport) as connector:
            resources = await connector.list_resources()
            assert [str(r.uri) for r in resources] == ["test://greeting"]

            read = await connector.read_resource("test://greeting")
            assert isinstance(read, types.ReadResourceResult)
            assert read.contents[0].text == "hello resource"

            prompts = await connector.list_prompts()
            assert [p.name for p in prompts] == ["greet"]

            prompt = await connector.get_prompt("greet", {"name": "Ada"})
            assert isinstance(prompt, types.GetPromptResult)
            assert prompt.messages[0].content.text == "Hello, Ada"


async def test_send_request_reaches_extension_method():
    server, _ = build_server()
    async with running(server) as transport:
        async with HMCPClientConnector(url=MCP_URL, transport=transport) as connector:
            result = await connector.send_request(
                "hmcp/patientContext/open",
                OpenParams(patient="Patient/1"),
                OpenResult,
            )
            assert isinstance(result, OpenResult)
            assert result.contextId == "pc_test"
            assert result.patient == "Patient/1"


# ---------------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------------


async def test_methods_before_connect_raise():
    connector = HMCPClientConnector(url=MCP_URL)
    for call in (
        connector.list_tools(),
        connector.call_tool("echo", {"text": "x"}),
        connector.list_resources(),
        connector.read_resource("test://greeting"),
        connector.list_prompts(),
        connector.get_prompt("greet", {"name": "x"}),
        connector.send_request(
            "hmcp/patientContext/open", OpenParams(patient="Patient/1"), OpenResult
        ),
    ):
        with pytest.raises(RuntimeError, match="Not connected"):
            await call


async def test_cleanup_twice_is_safe():
    server, _ = build_server()
    async with running(server) as transport:
        connector = HMCPClientConnector(url=MCP_URL, transport=transport)
        await connector.connect()
        await connector.cleanup()
        # A second cleanup is a no-op and must not raise.
        await connector.cleanup()
        assert connector.connected is False


async def test_cleanup_before_connect_is_safe():
    connector = HMCPClientConnector(url=MCP_URL)
    await connector.cleanup()
    assert connector.connected is False


class _CountingTransport(httpx2.AsyncBaseTransport):
    """Wraps a transport and counts the requests that reach it."""

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self._inner = inner
        self.requests = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        return await self._inner.handle_async_request(request)


@pytest.mark.parametrize(
    "meta, kwargs",
    [
        ({PATIENT_CONTEXT_EXTENSION: "x"}, {"patient_context": "pc_1"}),
        ({PATIENT_CONTEXT_EXTENSION: ["x"]}, {"patient_context": "pc_1"}),
        ({GUARDRAILS_EXTENSION: "x"}, {"require_guardrails": ["com.example/x"]}),
        ({GUARDRAILS_EXTENSION: 7}, {"require_guardrails": ["com.example/x"]}),
        # A non-mapping value is rejected even without the matching kwarg.
        ({PATIENT_CONTEXT_EXTENSION: "x"}, {}),
        ({PATIENT_CONTEXT_EXTENSION: ["x"]}, {}),
        ({GUARDRAILS_EXTENSION: "x"}, {}),
        ({GUARDRAILS_EXTENSION: 7}, {}),
    ],
)
async def test_non_mapping_extension_meta_raises_before_request(meta, kwargs):
    server, seen = build_server()
    async with running(server) as inner:
        counter = _CountingTransport(inner)
        async with HMCPClientConnector(url=MCP_URL, transport=counter) as connector:
            # connect() issues its own requests; only the tool call matters here.
            counter.requests = 0
            with pytest.raises(TypeError, match="must be a mapping"):
                await connector.call_tool("echo", {"text": "x"}, meta=meta, **kwargs)
            # The request never reached the server.
            assert counter.requests == 0
            assert seen == {}


async def test_concurrent_connect_establishes_one_session(monkeypatch):
    import hmcp.client.client_connector as client_connector

    server, _ = build_server()
    async with running(server) as transport:
        created: list[Any] = []
        real_client = client_connector.Client

        def counting_client(*args: Any, **kwargs: Any) -> Any:
            client = real_client(*args, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(client_connector, "Client", counting_client)

        connector = HMCPClientConnector(url=MCP_URL, transport=transport)
        await asyncio.gather(connector.connect(), connector.connect())
        try:
            assert connector.connected is True
            # Both calls shared one session: only one Client was constructed.
            assert len(created) == 1
        finally:
            await connector.cleanup()
        assert connector.connected is False


class _BlockingTransport(httpx2.AsyncBaseTransport):
    """Blocks the first request of the MCP handshake until cancelled."""

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self._inner = inner
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._blocked = False

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if not self._blocked:
            self._blocked = True
            self.started.set()
            await self.release.wait()
        return await self._inner.handle_async_request(request)


async def test_cancelled_connect_cleans_up_entered_resources(monkeypatch):
    server, _ = build_server()
    async with running(server) as inner:
        blocking = _BlockingTransport(inner)
        connector = HMCPClientConnector(url=MCP_URL, transport=blocking)

        cleanup_calls: list[bool] = []
        real_cleanup = HMCPClientConnector.cleanup

        async def spy_cleanup(self) -> None:
            cleanup_calls.append(True)
            await real_cleanup(self)

        monkeypatch.setattr(HMCPClientConnector, "cleanup", spy_cleanup)

        task = asyncio.create_task(connector.connect())
        # Wait until the HTTP client is entered and the handshake is in flight.
        await blocking.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The lock-held cleanup ran; the public cleanup() was never called.
        assert cleanup_calls == []
        assert connector.connected is False
        assert connector._http is None
        assert connector._client is None
        assert connector.session is None


async def test_context_manager_connects_and_cleans_up():
    server, _ = build_server()
    async with running(server) as transport:
        connector = HMCPClientConnector(url=MCP_URL, transport=transport)
        async with connector as entered:
            assert entered is connector
            assert connector.connected is True
            result = await connector.call_tool("echo", {"text": "hi"})
            assert result.content[0].text == "echo:hi"
        assert connector.connected is False
        assert connector.client is None


async def test_context_manager_cleans_up_when_body_raises():
    server, _ = build_server()
    async with running(server) as transport:
        connector = HMCPClientConnector(url=MCP_URL, transport=transport)
        with pytest.raises(RuntimeError, match="body boom"):
            async with connector:
                assert connector.connected is True
                raise RuntimeError("body boom")
        # The body's exception propagated and cleanup still ran.
        assert connector.connected is False
        assert connector.client is None


# ---------------------------------------------------------------------------
# Authentication (OAuthClient faked)
# ---------------------------------------------------------------------------


class _FakeToken:
    def __init__(self, access_token: str) -> None:
        self.access_token = access_token


class FakeOAuthClient:
    """Fake async-context-manager OAuthClient returning a fixed token."""

    last_kwargs: dict[str, Any] = {}
    token = "tok-good"

    def __init__(self, **kwargs: Any) -> None:
        FakeOAuthClient.last_kwargs = kwargs

    async def __aenter__(self) -> "FakeOAuthClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def get_client_credentials_token(self, scopes: Any = None) -> _FakeToken:
        return _FakeToken(self.token)


class _Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        if token == "tok-good":
            return AccessToken(
                token=token,
                client_id="app-1",
                scopes=["read"],
                resource=MCP_URL,
                subject="svc",
            )
        return None


async def test_oauth_token_sent_and_accepted(monkeypatch):
    monkeypatch.setattr("hmcp.client.client_connector.OAuthClient", FakeOAuthClient)
    server, _ = build_server(
        token_verifier=_Verifier(),
        auth=AuthSettings(
            issuer_url="https://auth.example.com",
            resource_server_url=MCP_URL,
            validate_token_resource=True,
        ),
    )
    async with running(server) as transport:
        connector = HMCPClientConnector(
            url=MCP_URL,
            client_id="test-client",
            client_secret="test-secret",
            scopes=["read"],
            transport=transport,
        )
        await connector.connect()
        try:
            tools = await connector.list_tools()
            assert [t.name for t in tools] == ["echo"]
        finally:
            await connector.cleanup()

    # The OAuth client was constructed with the auth server URL and the seam.
    assert FakeOAuthClient.last_kwargs["server_url"] == BASE
    assert FakeOAuthClient.last_kwargs["transport"] is transport


async def test_connector_without_credentials_is_rejected():
    server, _ = build_server(
        token_verifier=_Verifier(),
        auth=AuthSettings(
            issuer_url="https://auth.example.com",
            resource_server_url=MCP_URL,
            validate_token_resource=True,
        ),
    )
    async with running(server) as transport:
        connector = HMCPClientConnector(url=MCP_URL, transport=transport)
        with pytest.raises(Exception):
            await connector.connect()
        await connector.cleanup()


# ---------------------------------------------------------------------------
# Log hygiene
# ---------------------------------------------------------------------------


async def test_userinfo_in_url_is_not_logged(caplog):
    caplog.set_level(logging.DEBUG)
    # The mcp SDK and httpx2 log the endpoint/request URL verbatim (including
    # userinfo) at DEBUG/INFO; those third-party loggers are outside this
    # module. Silence them so this test isolates the connector's own logging.
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    server, _ = build_server()
    async with running(server) as transport:
        connector = HMCPClientConnector(
            url="http://user:secret@127.0.0.1:8000/mcp",
            transport=transport,
            debug=True,
        )
        await connector.connect()
        try:
            # The connector's debug line carries the host/port only.
            assert "127.0.0.1" in caplog.text
        finally:
            await connector.cleanup()
    assert "secret" not in caplog.text


class ExplodingOAuthClient:
    """Fake OAuthClient whose token call raises a sentinel-bearing error."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "ExplodingOAuthClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def get_client_credentials_token(self, scopes: Any = None):
        raise RuntimeError("SENTINEL-SECRET")


async def test_oauth_failure_message_is_not_logged(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(
        "hmcp.client.client_connector.OAuthClient", ExplodingOAuthClient
    )
    connector = HMCPClientConnector(
        url=MCP_URL,
        client_id="test-client",
        client_secret="test-secret",
        scopes=["read"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        await connector.connect()
    # The original error still propagates to the caller...
    assert "SENTINEL-SECRET" in str(excinfo.value)
    # ...but its text never reaches the logs.
    assert "SENTINEL-SECRET" not in caplog.text
