#!/usr/bin/env python3
"""
Example demonstrating how to run a standalone HMCP server.

This example shows how to:
1. Create and configure an HMCP server
2. Define custom sampling handlers
3. Run the server with SSE transport
4. Handle basic message processing
"""

import logging
import os
from typing import Any
from hmcp.server.hmcp_server import HMCPServer
from mcp.shared.context import RequestContext
from dotenv import load_dotenv
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Run the HMCP Standalone Server"""

    try:
        # Create and configure server
        server = HMCPServer(
            name="HMCP Standalone Server",
            host=os.getenv("STANDALONE_SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("STANDALONE_SERVER_PORT", "8060")),
            debug=os.getenv("STANDALONE_SERVER_DEBUG",
                            "false").lower() == "true",
            instructions="This is a standalone HMCP server for demonstration purposes.",

        )

        @server.sampling()
        async def handle_gateway_sampling(
            context: RequestContext[Any, Any],
            params: types.CreateMessageRequestParams,
        ) -> types.CreateMessageResult:
            """Handle gateway requests.

            Example:
                @hmcp_server.sampling()
                async def handle_sampling(context, params):
                    # The response is already available in context.server_info["response"]
                    return types.CreateMessageResult(
                        model="gateway-agent",
                        role="assistant",
                        content=types.TextContent(
                            type="text",
                            text=context.server_info["response"]
                        ),
                        stopReason="endTurn"
                    )
            """
            logger.info(f"STANDALONE: Processing response from server")

            # Extract the latest message content
            latest_message = params.messages[-1]
            message_content = ""
            if isinstance(latest_message.content, list):
                message_content = "".join(
                    [
                        content.text
                        for content in latest_message.content
                        if isinstance(content, types.TextContent)
                    ]
                )
            elif isinstance(latest_message.content, types.TextContent):
                message_content = latest_message.content.text

            return types.CreateMessageResult(
                model="standalone-agent",
                role="assistant",
                content=types.TextContent(
                    type="text",
                    text=f"STANDALONE SERVER: Received your message: {message_content}"
                ),
                stopReason="endTurn"
            )

        # Start the server
        logger.info(
            f"Starting HMCP Standalone Server on {os.getenv('STANDALONE_SERVER_HOST', '127.0.0.1')}:{os.getenv('STANDALONE_SERVER_PORT', '8060')}"
        )
        server.run(transport="streamable-http")

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
