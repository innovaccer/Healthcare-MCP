#!/usr/bin/env python3
"""
HMCP client connector: a helper for talking to an HMCP server over MCP.

This module provides :class:`HMCPClientConnector`, which abstracts away the
low-level details of establishing a streamable-HTTP connection, optional
client-credentials authentication, and session management. It exposes simple
methods to call tools, read resources, fetch prompts and send custom
extension requests.

Usage:
    from hmcp.client.client_connector import HMCPClientConnector

    connector = HMCPClientConnector(url="http://localhost:8050/mcp")
    await connector.connect()
    tools = await connector.list_tools()
    result = await connector.call_tool("tool_name", {"param": "value"})
    await connector.cleanup()

HMCP 0.1.0 is streamable-HTTP only; the deprecated HTTP+SSE transport and the
bidirectional sampling feature have both been removed.
"""

from __future__ import annotations

import asyncio
import logging
from asyncio import Lock
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any, Sequence
from urllib.parse import urlsplit

import httpx2
import mcp.types as types
from mcp.client import Client
from mcp.client.extension import ClientExtension
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from hmcp.client.patient_context import (
    GuardrailReportMissing,
    Identifier,
    PatientContextCloseParams,
    PatientContextCloseResult,
    PatientContextGetParams,
    PatientContextGetResult,
    PatientContextHandle,
    PatientContextOpenParams,
    PatientContextOpenResult,
    PatientRef,
    context_id_of,
)
from hmcp.shared.auth import OAuthClient
from hmcp.shared.constants import (
    GUARDRAILS_EXTENSION,
    METHOD_PATIENT_CONTEXT_CLOSE,
    METHOD_PATIENT_CONTEXT_GET,
    METHOD_PATIENT_CONTEXT_OPEN,
    META_CONTEXT_ID,
    META_REPORT,
    META_REQUIRE,
    PATIENT_CONTEXT_EXTENSION,
)

logger = logging.getLogger(__name__)


class _PatientContextClientExtension(ClientExtension):
    """Default client extension declaration for patient context."""

    identifier = PATIENT_CONTEXT_EXTENSION

    def settings(self) -> dict[str, Any]:
        return {}


class _GuardrailsClientExtension(ClientExtension):
    """Default client extension declaration for guardrails."""

    identifier = GUARDRAILS_EXTENSION

    def settings(self) -> dict[str, Any]:
        return {}


class HMCPClientConnector:
    """
    A helper that simplifies interactions with an HMCP server.

    The connector manages a streamable-HTTP connection, optional OAuth
    client-credentials authentication, and the underlying MCP client/session.

    When ``client_id`` and ``client_secret`` are given, a client-credentials
    token is fetched once at :meth:`connect` and sent as a static bearer header
    on every request; it is never refreshed. If the token later expires, the
    server's 401 surfaces as an ``MCPError`` and :attr:`connected` stays
    ``True``. To recover, call :meth:`cleanup` and then :meth:`connect` again
    (or re-enter the connector as a context manager).

    The connector is an async context manager: ``async with
    HMCPClientConnector(...) as connector:`` connects on entry and cleans up on
    exit, including when the body raises.
    """

    def __init__(
        self,
        url: str,
        *,
        debug: bool = False,
        client_id: str | None = None,
        client_secret: str | None = None,
        scopes: list[str] | None = None,
        headers: dict[str, str] | None = None,
        extensions: Sequence[ClientExtension] | None = None,
        timeout: float = 30.0,
        transport: httpx2.AsyncBaseTransport | None = None,
        auth_server_url: str | None = None,
    ) -> None:
        """
        Initialize the connector.

        Args:
            url: The full MCP endpoint, e.g. ``http://host:port/mcp``.
            debug: Whether to enable debug logging.
            client_id: OAuth client id (client-credentials flow).
            client_secret: OAuth client secret (client-credentials flow).
            scopes: OAuth scopes to request.
            headers: Extra HTTP headers to send with every request.
            extensions: Client extensions to advertise to the server.
            timeout: HTTP timeout in seconds.
            transport: Optional httpx2 transport (test seam); passed to
                ``httpx2.AsyncClient`` and to the OAuth client.
            auth_server_url: Base URL of the authorization server. Defaults to
                the scheme and host of ``url``.
        """
        self.server_url = url
        self.debug = debug
        self._client_id = client_id
        self._client_secret = client_secret
        self._scopes = scopes
        self._headers = dict(headers or {})
        self._extensions = self._merge_extensions(extensions)
        self._timeout = timeout
        self._transport = transport

        if auth_server_url is None:
            parts = urlsplit(url)
            auth_server_url = f"{parts.scheme}://{parts.netloc}"
        self.auth_server_url = auth_server_url

        # A library configures only its own logger, never the root logger.
        if debug:
            logging.getLogger("hmcp").setLevel(logging.DEBUG)

        # Connection state
        self._client: Client | None = None
        self._session = None
        self._http: httpx2.AsyncClient | None = None
        self._exit_stack = AsyncExitStack()
        self._lifecycle_lock = Lock()
        self.connected = False

    @staticmethod
    def _merge_extensions(
        extensions: Sequence[ClientExtension] | None,
    ) -> list[ClientExtension]:
        """Return caller extensions plus the default ones, de-duplicated by id."""
        merged: list[ClientExtension] = list(extensions) if extensions else []
        present = {ext.identifier for ext in merged}
        for default in (_PatientContextClientExtension(), _GuardrailsClientExtension()):
            if default.identifier not in present:
                merged.append(default)
                present.add(default.identifier)
        return merged

    @property
    def client(self) -> Client | None:
        """The underlying MCP client, or ``None`` before :meth:`connect`."""
        return self._client

    @property
    def session(self) -> ClientSession | None:
        """The underlying MCP client session, or ``None`` before connect."""
        return self._session

    @property
    def server_capabilities(self) -> types.ServerCapabilities | None:
        """Capabilities advertised by the connected server."""
        return self._client.server_capabilities if self._client else None

    @property
    def server_info(self) -> types.Implementation | None:
        """Implementation info advertised by the connected server."""
        return self._client.server_info if self._client else None

    async def connect(self) -> None:
        """
        Connect to the HMCP server.

        Establishes a streamable-HTTP connection. If ``client_id`` and
        ``client_secret`` were supplied, a client-credentials token is obtained
        from the authorization server and sent as a static bearer header; it is
        never refreshed. A later 401 surfaces as an ``MCPError`` and leaves
        :attr:`connected` ``True``; call :meth:`cleanup` and then
        :meth:`connect` again (or re-enter the context manager) to recover.
        Concurrent calls are serialised: the first establishes the session and
        the rest observe it already connected.

        Returns:
            None.

        Raises:
            Exception: If the connection fails.
        """
        async with self._lifecycle_lock:
            if self.connected:
                logger.warning("Already connected to server")
                return

            try:
                headers = dict(self._headers)
                if self._client_id and self._client_secret:
                    async with OAuthClient(
                        client_id=self._client_id,
                        client_secret=self._client_secret,
                        scopes=self._scopes,
                        server_url=self.auth_server_url,
                        transport=self._transport,
                    ) as oauth_client:
                        token = await oauth_client.get_client_credentials_token(
                            scopes=self._scopes
                        )
                    headers["Authorization"] = f"Bearer {token.access_token}"

                parts = urlsplit(self.server_url)
                logger.debug(
                    "Connecting to HMCP server at %s:%s", parts.hostname, parts.port
                )

                self._http = await self._exit_stack.enter_async_context(
                    httpx2.AsyncClient(
                        headers=headers,
                        timeout=self._timeout,
                        transport=self._transport,
                    )
                )

                transport_cm = streamable_http_client(
                    self.server_url, http_client=self._http
                )
                self._client = await self._exit_stack.enter_async_context(
                    Client(transport_cm, extensions=self._extensions)
                )
                self._session = self._client.session
                self.connected = True

                info = self._client.server_info
                if info is not None:
                    logger.info(f"Connected to {info.name} v{info.version}")

            except BaseException as e:
                logger.error("Failed to connect to HMCP server: %s", type(e).__name__)
                await self._cleanup_locked()
                raise

    async def cleanup(self) -> None:
        """
        Clean up the connection to the HMCP server.

        Safe to call more than once, and safe to call before connecting.

        Returns:
            None.
        """
        async with self._lifecycle_lock:
            await self._cleanup_locked()

    async def _cleanup_locked(self) -> None:
        """Tear down the session and HTTP client; the lifecycle lock is held."""
        if not self.connected and self._client is None and self._http is None:
            return

        # Mark as disconnected first to prevent new operations.
        self.connected = False
        self._session = None
        self._client = None
        self._http = None

        try:
            await self._exit_stack.aclose()
        except asyncio.CancelledError:
            logger.debug("Exit stack cleanup was cancelled during shutdown")
        except Exception as e:
            logger.debug(
                "Non-critical error during exit stack cleanup: %s",
                type(e).__name__,
            )
        finally:
            self._exit_stack = AsyncExitStack()

        logger.debug("Cleaned up HMCP server connection")

    def _ensure_connected(self) -> None:
        """Raise ``RuntimeError`` unless a session is established."""
        if not self.connected or self._client is None or self._session is None:
            raise RuntimeError("Not connected")

    async def list_tools(self) -> list[types.Tool]:
        """
        List all available tools on the server.

        Returns:
            The tools advertised by the server.

        Raises:
            RuntimeError: If not connected to the server.
        """
        self._ensure_connected()
        result: types.ListToolsResult = await self._session.list_tools()
        return list(result.tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
        patient_context: PatientContextHandle | str | None = None,
        require_guardrails: Sequence[str] | None = None,
    ) -> types.CallToolResult:
        """
        Call a tool on the HMCP server.

        Args:
            name: The name of the tool to call.
            arguments: Optional arguments to pass to the tool.
            meta: Optional request ``_meta`` to send with the call.
            patient_context: Optional patient context (handle or id) to attach
                under the patient-context extension key.
            require_guardrails: Optional guardrail ids that must report on the
                result.

        Returns:
            The server's :class:`mcp.types.CallToolResult`.

        Raises:
            RuntimeError: If not connected to the server.
            TypeError: If a caller-supplied extension ``_meta`` value is present
                but not a mapping.
            GuardrailReportMissing: If a required guardrail did not report.
        """
        self._ensure_connected()

        merged: dict[str, Any] = dict(meta or {})
        # Any present extension value must be a mapping, whether or not this
        # call merges into it: a malformed value is a caller error and is never
        # forwarded to the server.
        for extension_key in (PATIENT_CONTEXT_EXTENSION, GUARDRAILS_EXTENSION):
            existing = merged.get(extension_key)
            if existing is not None and not isinstance(existing, Mapping):
                raise TypeError(
                    f"_meta[{extension_key!r}] must be a mapping, "
                    f"not {type(existing).__name__}"
                )
        if patient_context is not None:
            context_meta = dict(merged.get(PATIENT_CONTEXT_EXTENSION) or {})
            context_meta[META_CONTEXT_ID] = context_id_of(patient_context)
            merged[PATIENT_CONTEXT_EXTENSION] = context_meta
        if require_guardrails is not None:
            guardrail_meta = dict(merged.get(GUARDRAILS_EXTENSION) or {})
            guardrail_meta[META_REQUIRE] = list(require_guardrails)
            merged[GUARDRAILS_EXTENSION] = guardrail_meta

        result = await self._session.call_tool(name, arguments, meta=merged)

        if require_guardrails:
            self._verify_guardrail_reports(result, require_guardrails)
        return result

    @staticmethod
    def _verify_guardrail_reports(
        result: types.CallToolResult, required: Sequence[str]
    ) -> None:
        """Raise ``GuardrailReportMissing`` for any required id with no report.

        Applied even when the tool result is an error, so a failed call cannot
        silently skip a required guardrail. A requirement may pin a major
        version (``id@2``); it is then satisfied only by a report entry with
        that id whose ``version`` has the same major. A missing or non-string
        version does not satisfy a pinned requirement.
        """
        guardrails = (result.meta or {}).get(GUARDRAILS_EXTENSION)
        if not isinstance(guardrails, dict):
            guardrails = {}
        report = guardrails.get(META_REPORT)
        if not isinstance(report, list):
            report = []
        entries = [
            entry
            for entry in report
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]
        for required_id in required:
            guardrail_id, sep, major_text = required_id.partition("@")
            if not sep:
                matched = any(entry["id"] == guardrail_id for entry in entries)
            else:
                major = HMCPClientConnector._major_of_pin(major_text)
                matched = major is not None and any(
                    entry["id"] == guardrail_id
                    and HMCPClientConnector._version_major(entry.get("version"))
                    == major
                    for entry in entries
                )
            if not matched:
                raise GuardrailReportMissing(required_id)

    @staticmethod
    def _major_of_pin(major_text: str) -> int | None:
        """Return the integer major of a ``@<major>`` pin, or ``None``.

        Mirrors the server's rule: a pin major is a short ASCII digit string.
        A malformed pin has no major and therefore cannot be satisfied.
        """
        if not (major_text.isascii() and major_text.isdigit() and len(major_text) <= 6):
            return None
        return int(major_text)

    @staticmethod
    def _version_major(version: Any) -> int | None:
        """Return the leading integer major of a version string, or ``None``.

        The major is the digits before the first ``.``. A missing or non-string
        version has no major.
        """
        if not isinstance(version, str):
            return None
        head = version.partition(".")[0]
        if not (head.isascii() and head.isdigit()):
            return None
        try:
            return int(head)
        except ValueError:
            return None

    async def open_patient_context(
        self,
        patient_reference: str,
        *,
        purpose_of_use: str = "TREAT",
        identifiers: Sequence[tuple[str, str]] = (),
        encounter: str | None = None,
        correlation_id: str | None = None,
    ) -> PatientContextHandle:
        """
        Open a patient context on the server.

        Args:
            patient_reference: FHIR-relative reference, e.g. ``Patient/123``.
            purpose_of_use: HL7 v3 PurposeOfUse code (default ``TREAT``).
            identifiers: Optional ``(system, value)`` identifier pairs.
            encounter: Optional FHIR-relative encounter reference.
            correlation_id: Optional caller correlation id.

        Returns:
            A :class:`PatientContextHandle` for the newly opened context.

        Raises:
            RuntimeError: If not connected to the server.
            ValueError: If the server's ``expiresAt`` has no UTC offset.
        """
        params = PatientContextOpenParams(
            patient=PatientRef(
                reference=patient_reference,
                identifiers=[Identifier(system=s, value=v) for s, v in identifiers],
            ),
            encounter=encounter,
            purpose_of_use=purpose_of_use,
            correlation_id=correlation_id,
        )
        result: PatientContextOpenResult = await self.send_request(
            METHOD_PATIENT_CONTEXT_OPEN, params, PatientContextOpenResult
        )
        if result.expires_at.tzinfo is None or result.expires_at.utcoffset() is None:
            raise ValueError("Patient context expires_at must include a UTC offset")
        return PatientContextHandle(
            context_id=result.context_id,
            status=result.status,
            expires_at=result.expires_at,
        )

    async def get_patient_context(
        self, handle: PatientContextHandle | str
    ) -> PatientContextGetResult:
        """
        Fetch the current state of a patient context.

        Args:
            handle: A :class:`PatientContextHandle` or a bare context id.

        Returns:
            The server's :class:`PatientContextGetResult`, carrying the context
            id, status and expiry, plus the bound patient, encounter and
            purpose of use when the server reports them.

        Raises:
            RuntimeError: If not connected to the server.
        """
        params = PatientContextGetParams(context_id=context_id_of(handle))
        result: PatientContextGetResult = await self.send_request(
            METHOD_PATIENT_CONTEXT_GET, params, PatientContextGetResult
        )
        return result

    async def close_patient_context(self, handle: PatientContextHandle | str) -> None:
        """
        Close a patient context on the server.

        Args:
            handle: A :class:`PatientContextHandle` or a bare context id.

        Returns:
            None. A successful close is confirmed by the server; its
            :class:`PatientContextCloseResult` is not surfaced to the caller.

        Raises:
            RuntimeError: If not connected to the server.
        """
        params = PatientContextCloseParams(context_id=context_id_of(handle))
        await self.send_request(
            METHOD_PATIENT_CONTEXT_CLOSE, params, PatientContextCloseResult
        )

    async def list_resources(self) -> list[types.Resource]:
        """
        List all available resources on the server.

        Returns:
            The resources advertised by the server.

        Raises:
            RuntimeError: If not connected to the server.
        """
        self._ensure_connected()
        result: types.ListResourcesResult = await self._session.list_resources()
        return list(result.resources)

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        """
        Read a resource from the server.

        Args:
            uri: The URI of the resource to read.

        Returns:
            The server's :class:`mcp.types.ReadResourceResult`.

        Raises:
            RuntimeError: If not connected to the server.
        """
        self._ensure_connected()
        return await self._session.read_resource(uri)

    async def list_prompts(self) -> list[types.Prompt]:
        """
        List all available prompts on the server.

        Returns:
            The prompts advertised by the server.

        Raises:
            RuntimeError: If not connected to the server.
        """
        self._ensure_connected()
        result: types.ListPromptsResult = await self._session.list_prompts()
        return list(result.prompts)

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> types.GetPromptResult:
        """
        Get a prompt from the server.

        Args:
            name: The name of the prompt.
            arguments: Optional arguments to pass to the prompt.

        Returns:
            The server's :class:`mcp.types.GetPromptResult`.

        Raises:
            RuntimeError: If not connected to the server.
        """
        self._ensure_connected()
        return await self._session.get_prompt(name, arguments)

    async def send_request(
        self, method: str, params: types.RequestParams, result_type: type[BaseModel]
    ) -> BaseModel:
        """
        Send a custom (extension) request to the server.

        Args:
            method: The custom method name, e.g. ``hmcp/patientContext/open``.
            params: The request parameters.
            result_type: Pydantic model used to parse the response.

        Returns:
            The parsed result, an instance of ``result_type``.

        Raises:
            RuntimeError: If not connected to the server.
        """
        self._ensure_connected()
        request = types.Request[Any, str](method=method, params=params)
        return await self._session.send_request(request, result_type)

    async def __aenter__(self) -> "HMCPClientConnector":
        """Enter the async context manager, connecting to the server.

        Returns:
            This connector, once :meth:`connect` has completed.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager, cleaning up the connection.

        Cleanup runs whether the body completed or raised, and the body's
        exception, if any, is not suppressed.
        """
        await self.cleanup()
