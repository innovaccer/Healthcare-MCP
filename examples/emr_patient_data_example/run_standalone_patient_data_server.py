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
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=8060,
            debug=True,
            instructions="This is a standalone HMCP server for demonstration purposes. It is used to access patient data.",

        )

        # Sample patient database
        patient_db = {
            "John Smith": "PT12345",
            "Jane Doe": "PT67890",
            "Bob Johnson": "PT24680",
        }

        @server.sampling()
        async def handle_patient_data_sampling(
            context: RequestContext[Any, Any], params: types.CreateMessageRequestParams
        ) -> types.CreateMessageResult:
            """Handle patient data access requests."""
            logger.info(
                f"PATIENT DATA: Received sampling request with {len(params.messages)} messages"
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

            logger.info(f"PATIENT DATA: Processing message: {message_content}")

            # Check if a specific patient name is mentioned
            for patient_name, patient_id in patient_db.items():
                if patient_name.lower() in message_content.lower():
                    logger.info(
                        f"PATIENT DATA: Found patient {patient_name} with ID {patient_id}"
                    )
                    return types.CreateMessageResult(
                        model="patient-data-agent",
                        role="assistant",
                        content=types.TextContent(
                            type="text",
                            text=f"Patient identifier for {patient_name}: patient_id={patient_id}",
                        ),
                        stopReason="endTurn",
                    )

            # If no specific patient was found but requesting patient info
            if (
                "patient" in message_content.lower()
                and "identifier" in message_content.lower()
            ):
                logger.info("PATIENT DATA: Generic patient info request")
                available_patients = ", ".join(patient_db.keys())
                return types.CreateMessageResult(
                    model="patient-data-agent",
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text=f"Please specify which patient you need information for. Available patients: {available_patients}.",
                    ),
                    stopReason="endTurn",
                )

            # Generic response for other cases
            return types.CreateMessageResult(
                model="patient-data-agent",
                role="assistant",
                content=types.TextContent(
                    type="text", text="How can I help you access patient information?"
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
