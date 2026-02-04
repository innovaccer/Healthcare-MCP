#!/usr/bin/env python3
"""
HMCP Server Helper: A utility class to simplify interactions with HMCP servers

This module provides a helper class that makes it easier to interact with HMCP servers
by abstracting away the low-level details of establishing connections, authentication,
and managing server sessions. It provides simple methods to call tools, access resources,
and use the sampling capabilities of HMCP servers.

Usage:
    from hmcp_server_helper import HMCPClientConnector

    # Create a helper for an existing HMCP server
    helper = HMCPClientConnector(host="localhost", port=8050)

    # Connect to the server
    await helper.connect()

    # Send a message for sampling
    result = await helper.create_message("Your message here")

    # Call a tool
    tool_result = await helper.call_tool("tool_name", {"param": "value"})

    # Clean up when done
    await helper.cleanup()
"""

from __future__ import annotations
import asyncio
import logging
from asyncio import Lock
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
import httpx
from hmcp.shared.auth import OAuthClient
from hmcp.client.hmcp_client import HMCPClient
from mcp.client.sse import sse_client
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    SamplingMessage,
    TextContent,
    ErrorData,
    CallToolResult,
    ListToolsResult,
    ListPromptsResult,
    GetPromptResult,
    ListResourcesResult,
    ReadResourceResult,
)
from datetime import timedelta

# Configure logging
logger = logging.getLogger(__name__)


class HMCPClientConnector:
    """
    A helper class that simplifies interactions with HMCP servers.

    This class provides convenient methods to connect to an HMCP server,
    call its tools, access its resources, and use its sampling capabilities.
    It handles authentication and connection management automatically.
    """

    def __init__(
        self,
        url: str,
        debug: bool = False,
        client_id: str = None,
        client_secret: str = None,
        scopes: List[str] = None,
    ):
        """
        Initialize the HMCP Server Helper.

        Args:
            url: url to the mcp server
            auth_config: Optional authentication configuration
            client_id: Client ID to use for authentication
            client_secret: Client secret to use for authentication
            debug: Whether to enable debug logging
        """
        self.server_url = url
        self.debug = debug

        if client_id and client_secret and scopes:
            self.oauth_client = OAuthClient(
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
                server_url=self.server_url,
            )
        else:
            self.oauth_client = None

        # Set up logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level)

        # Initialize connection components
        self.session = None
        self.session_id = None
        self.client = None
        self.exit_stack = AsyncExitStack()
        self._cleanup_lock = Lock()
        self.connected = False
        self.server_info = None

    async def connect(
        self,
        transport: str = "streamable-http",
        timeout: int = 30,
        sse_read_timeout: int = 5,
        headers: Dict[str, str] = {},
    ) -> Dict[str, Any]:
        """
        Connect to the HMCP server.

        This method establishes a connection to the HMCP server using the SSE transport,
        initializes the client session, and returns the server information.

        Returns:
            Dict containing the server information including name, version, etc.

        Raises:
            Exception: If the connection fails
        """
        if self.connected:
            logger.warning("Already connected to server")
            return self._get_server_info()

        try:
            auth_headers = {}
            if self.oauth_client:
                async with self.oauth_client:
                    # Set the token in the OAuth client
                    await self.oauth_client.set_client_credentials_token()
                    auth_headers = self.oauth_client.get_auth_header()

            headers = {**auth_headers, **headers}

            logger.debug(f"Connecting to HMCP server at {self.server_url}/sse")

            # Setup streams and session using AsyncExitStack to properly manage cleanup
            conn_transport = await self.exit_stack.enter_async_context(
                self._create_streams(
                    headers,
                    transport=transport,
                    timeout=timeout,
                    sse_read_timeout=sse_read_timeout,
                )
            )

            # Streamable HTTP transport returns (read_stream, write_stream, get_session_id)
            # SSE transport returns (read_stream, write_stream)
            if transport == "streamable-http":
                read_stream, write_stream, get_session_id = conn_transport
                # Store session ID callback for later use
                self.get_session_id = get_session_id
            else:
                read_stream, write_stream = conn_transport
                self.get_session_id = None

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize the session
            init_result = await self.session.initialize()

            # Create the HMCP client
            self.client = HMCPClient(self.session)
            self.connected = True

            logger.info(
                f"Connected to {init_result.serverInfo.name} v{init_result.serverInfo.version}"
            )

            # Note: Session ID callback was removed in MCP SDK v1.26.0
            # Session management is now handled internally by the transport layer

            # Store server info
            self.server_info = {
                "name": init_result.serverInfo.name,
                "version": init_result.serverInfo.version,
                "protocolVersion": init_result.protocolVersion,
                "capabilities": init_result.capabilities,
            }

            return self.server_info

        except Exception as e:
            logger.error(f"Failed to connect to HMCP server: {str(e)}")
            logger.exception("Full traceback:")  # Add detailed traceback
            # Ensure proper cleanup on failure
            await self.cleanup()
            raise

    def _create_streams(
        self,
        headers: Dict[str, str],
        transport: str,
        timeout: int,
        sse_read_timeout: int,
    ):
        """
        Create the streams to connect to the server.
        """
        if transport == "streamable-http":
            return self._create_streamable_http_streams(
                headers, timeout, sse_read_timeout
            )
        elif transport == "sse":
            return self._create_sse_streams(headers, timeout, sse_read_timeout)
        else:
            raise ValueError(f"Invalid transport: {transport}")

    def _create_streamable_http_streams(
        self, headers: Dict[str, str], timeout: int, sse_read_timeout: int
    ):
        """
        Create the Streamable HTTP streams to connect to the server.
        Uses httpx.AsyncClient for proper timeout and header configuration.
        """
        # Create httpx client with timeout and headers
        http_client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout, read=60 * sse_read_timeout),
            follow_redirects=True,
        )
        return streamable_http_client(
            url=f"{self.server_url}/mcp/",
            http_client=http_client,
        )

    def _create_sse_streams(
        self, headers: Dict[str, str], timeout: int, sse_read_timeout: int
    ):
        """
        Create the SSE streams to connect to the server.
        This is a helper method for connect().

        Returns:
            An async context manager that yields a tuple of (read_stream, write_stream)
        """
        return sse_client(
            url=f"{self.server_url}/sse/",
            headers=headers,
            timeout=float(timeout),  # HTTP timeout in seconds
            # SSE read timeout in seconds (default 5 minutes = 300 seconds)
            sse_read_timeout=float(60 * sse_read_timeout),
        )

    async def cleanup(self) -> None:
        """
        Clean up the connection to the HMCP server.

        This method closes any open connections and frees resources.
        It should be called when the helper is no longer needed.
        """
        if not self.connected:
            return

        async with self._cleanup_lock:
            try:
                # First mark as disconnected to prevent new operations
                self.connected = False

                # Store references to clear
                session = self.session
                client = self.client

                # Clear references first
                self.session = None
                self.client = None

                # Close the exit stack if it exists
                if hasattr(self, "exit_stack"):
                    try:
                        await self.exit_stack.aclose()
                    except asyncio.CancelledError:
                        logger.debug("Exit stack cleanup was cancelled during shutdown")
                    except Exception as e:
                        logger.debug(
                            f"Non-critical error during exit stack cleanup: {str(e)}"
                        )

                # Explicitly close session if it exists
                if session:
                    try:
                        await session.close()
                    except asyncio.CancelledError:
                        logger.debug("Session cleanup was cancelled during shutdown")
                    except Exception as e:
                        logger.debug(
                            f"Non-critical error during session cleanup: {str(e)}"
                        )

                logger.debug("Cleaned up HMCP server connection")

            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                # Don't re-raise the error to ensure cleanup completes

    def _ensure_connected(self) -> None:
        """
        Ensure that the helper is connected to the server.

        Raises:
            RuntimeError: If not connected to the server
        """
        if not self.connected or not self.client or not self.session:
            raise RuntimeError("Not connected to HMCP server. Call connect() first.")

    def _get_server_info(self) -> Dict[str, Any]:
        """
        Get basic server information.

        Returns:
            Dict containing server information
        """
        self._ensure_connected()
        if self.server_info:
            return self.server_info

        return {
            "name": "Unknown",
            "version": "Unknown",
            "capabilities": {},
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools on the server.

        Returns:
            List of tools with their names, descriptions, and schemas

        Raises:
            RuntimeError: If not connected to the server
        """
        self._ensure_connected()

        try:
            result: ListToolsResult = await self.session.list_tools()

            logger.info(f"Client Connector ListToolsResult: {result}")
            # Convert to a more user-friendly format
            tools_list = [tool for tool in result.tools]
            return tools_list

        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            raise

    async def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a tool on the HMCP server.

        Args:
            tool_name: The name of the tool to call
            arguments: Optional arguments to pass to the tool

        Returns:
            The result of the tool call

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If the tool call returns an error
        """
        self._ensure_connected()

        try:
            result: CallToolResult = await self.session.call_tool(tool_name, arguments)
            logger.info(f"CallToolResult: {result}")
            # Convert to a dictionary for easier use
            if hasattr(result, "content") and result.content is not None:
                return result.content
            else:
                return []

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            raise

    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List all available resources on the server.

        Returns:
            List of resources

        Raises:
            RuntimeError: If not connected to the server
        """
        self._ensure_connected()

        try:
            result: ListResourcesResult = await self.session.list_resources()

            # Convert to a more user-friendly format
            resources_list = []
            for resource in result.resources:
                resource_dict = {
                    "uri": resource.uri,
                    "title": getattr(resource, "title", None),
                    "description": getattr(resource, "description", None),
                }
                resources_list.append(resource_dict)

            return resources_list

        except Exception as e:
            logger.error(f"Error listing resources: {str(e)}")
            raise

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the server.

        Args:
            uri: The URI of the resource to read

        Returns:
            The content of the resource

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If the resource cannot be read
        """
        self._ensure_connected()

        try:
            result: ReadResourceResult = await self.session.read_resource(uri)

            # Convert to a dictionary for easier use
            return {
                "content": result.content,
                "contentType": getattr(result, "contentType", None),
            }

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {str(e)}")
            raise

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """
        List all available prompts on the server.

        Returns:
            List of prompts

        Raises:
            RuntimeError: If not connected to the server
        """
        self._ensure_connected()

        try:
            result: ListPromptsResult = await self.session.list_prompts()

            # Convert to a more user-friendly format
            prompts_list = []
            for prompt in result.prompts:
                prompt_dict = {
                    "name": prompt.name,
                    "description": getattr(prompt, "description", None),
                }
                prompts_list.append(prompt_dict)

            return prompts_list

        except Exception as e:
            logger.error(f"Error listing prompts: {str(e)}")
            raise

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Get a prompt from the server.

        Args:
            name: The name of the prompt
            arguments: Optional arguments to pass to the prompt

        Returns:
            The prompt content

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If the prompt cannot be retrieved
        """
        self._ensure_connected()

        try:
            result: GetPromptResult = await self.session.get_prompt(name, arguments)

            return result.prompt

        except Exception as e:
            logger.error(f"Error getting prompt {name}: {str(e)}")
            raise

    async def create_message(
        self,
        message: str,
        role: str = "user",
        messages_history: Optional[List[Dict[str, Any]]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a message using the server's sampling capability.

        Args:
            message: The message content
            role: The role of the message sender (default: "user")
            messages_history: Optional history of previous messages
            model_params: Optional parameters for the model

        Returns:
            The response from the server containing role, content, model, stopReason

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If message creation fails
        """
        self._ensure_connected()

        try:
            # Convert message to the format expected by HMCP
            text_content = TextContent(type="text", text=message)

            # Prepare message history
            all_messages = []

            if messages_history:
                for msg in messages_history:
                    if isinstance(msg, dict) and "content" in msg and "role" in msg:
                        if isinstance(msg["content"], str):
                            all_messages.append(
                                SamplingMessage(
                                    role=msg["role"],
                                    content=TextContent(
                                        type="text", text=msg["content"]
                                    ),
                                )
                            )
                        elif isinstance(msg["content"], TextContent):
                            all_messages.append(
                                SamplingMessage(
                                    role=msg["role"], content=msg["content"]
                                )
                            )

            # Add the current message
            if message:
                all_messages.append(SamplingMessage(role=role, content=text_content))

            # Use the client to create a message
            result = await self.client.create_message(messages=all_messages)

            # Handle error response
            if isinstance(result, ErrorData):
                error_msg = (
                    f"Error creating message: {result.message} (code: {result.code})"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Extract and return the result
            content = None
            if hasattr(result, "content"):
                content = getattr(result.content, "text", result.content)

            response = {
                "role": getattr(result, "role", "assistant"),
                "content": content,
                "model": getattr(result, "model", None),
                "stopReason": getattr(result, "stopReason", None),
            }

            return response

        except Exception as e:
            if not isinstance(e, ValueError):
                logger.error(f"Error creating message: {str(e)}")
            raise

    async def __aenter__(self) -> "HMCPClientConnector":
        """
        Enter the async context manager.

        Returns:
            The helper instance
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the async context manager.
        """
        await self.cleanup()
