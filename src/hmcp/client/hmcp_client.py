"""
Extended MCP Client with support for bidirectional sampling communication.

This module extends the standard MCP ClientSession to add direct support
for sending sampling/createMessage requests to an HMCP Server.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import mcp.types as types
from mcp import ClientSession
from pydantic import RootModel

# Configure logging for the HMCP client module
logger = logging.getLogger(__name__)

new_client_request_base = RootModel[
    types.PingRequest
    | types.InitializeRequest
    | types.CompleteRequest
    | types.SetLevelRequest
    | types.GetPromptRequest
    | types.ListPromptsRequest
    | types.ListResourcesRequest
    | types.ListResourceTemplatesRequest
    | types.ReadResourceRequest
    | types.SubscribeRequest
    | types.UnsubscribeRequest
    | types.CallToolRequest
    | types.ListToolsRequest
    | types.CreateMessageRequest
]

types.ClientRequest = type("ClientRequest", (new_client_request_base,), {})


class HMCPClient:
    """
    HMCP Client extends the MCP Client with direct sampling capability.

    This wrapper allows the client to send CreateMessageRequest messages to servers
    that implement the HMCP protocol with bidirectional sampling support. The HMCP
    client provides methods for generating text from models exposed through HMCP
    servers.
    """

    def __init__(self, session: ClientSession):
        """
        Initialize the HMCP Client.

        Args:
            session: An existing MCP ClientSession that's already connected to a server
        """
        # Store the provided ClientSession for communication with the server
        self.session = session

    async def create_message(
        self,
        messages: List[types.SamplingMessage],
        maxTokens: int = 1000,
        temperature: Optional[float] = None,
        topP: Optional[float] = None,
        stop: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[types.CreateMessageResult, types.ErrorData]:
        """
        Send a sampling/createMessage request to the server.

        This method allows the client to request text generation from the server
        by providing a conversation history and sampling parameters.

        Args:
            messages: List of messages representing the conversation history
            maxTokens: Maximum number of tokens to generate in the response
            temperature: Optional temperature for controlling randomness (0-1)
            topP: Optional top-p for nucleus sampling (0-1)
            stop: Optional list of stop sequences that will halt generation
            metadata: Optional metadata to include with the request

        Returns:
            The server's response as a CreateMessageResult or ErrorData if an error occurred
        """
        logger.info(f"Sending sampling request with {len(messages)} messages")

        # Create the request parameters with properly named camelCase parameters
        params = types.CreateMessageRequestParams(
            messages=messages,
            maxTokens=maxTokens,
            temperature=temperature,
            topP=topP,
            stop=stop,
            metadata=metadata,
        )

        # Construct the request object with the sampling method and parameters
        createMessageRequest = types.CreateMessageRequest(
            method="sampling/createMessage", params=params
        )

        # Log the request details for debugging purposes
        logger.info(f"Sending sampling request: {createMessageRequest}")

        # Send the request to the server and await the response
        return await self.session.send_request(
            types.ClientRequest(createMessageRequest), types.CreateMessageResult
        )
