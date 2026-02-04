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
            name="EMR Writeback Server",
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=8050,
            debug=True,
            instructions="This is a standalone HMCP server for demonstration purposes. It is used to write back data to the EMR.",

        )

        @server.sampling()
        async def handle_emr_writeback_sampling(
            context: RequestContext[Any, Any],
            params: types.CreateMessageRequestParams,
        ) -> types.CreateMessageResult:
            """Handle EMR writeback requests.

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
            """Handle EMR writeback requests."""
            logger.info(
                f"EMR WRITEBACK: Received sampling request with {len(params.messages)} messages"
            )

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

            logger.info(
                f"EMR WRITEBACK: Processing message: {message_content}")

            # Check if this is the first request (clinical data blob without patient ID)
            if (
                "patient_id" not in message_content.lower()
                and "clinical_data" in message_content.lower()
            ):
                # Need more information - request patient ID
                logger.info(
                    "EMR WRITEBACK: Patient ID missing, requesting additional information"
                )
                return types.CreateMessageResult(
                    model="emr-writeback-agent",
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text="Additional information required: Need patient_id to associate with this clinical data.",
                    ),
                    stopReason="endTurn",
                )

            # Check if this is the follow-up with patient ID
            elif (
                "patient_id" in message_content.lower()
                and "clinical_data" in message_content.lower()
            ):
                # Successfully received all required information
                logger.info(
                    "EMR WRITEBACK: Received all required information, processing EMR update"
                )
                return types.CreateMessageResult(
                    model="emr-writeback-agent",
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text="Success: Clinical data has been written to the EMR system for the specified patient.",
                    ),
                    stopReason="endTurn",
                )

            # Generic response for other cases
            else:
                return types.CreateMessageResult(
                    model="emr-writeback-agent",
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text="Please provide valid clinical_data to write to the EMR system.",
                    ),
                    stopReason="endTurn",
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
