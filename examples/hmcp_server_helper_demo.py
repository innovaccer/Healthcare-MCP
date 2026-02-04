#!/usr/bin/env python3
"""
HMCP Server Helper Demo

This script demonstrates how to use the HMCPServerHelper class to interact with HMCP servers.
It connects to both the EMR Writeback Agent and Patient Data Access Agent from the hmcp_llm_demo.py
example, and shows how to use the helper functions to send messages, call tools, and work with
resources in a simplified way.

Usage:
    python hmcp_server_helper_demo.py

Requirements:
    - Both EMR Writeback Agent and Patient Data Access Agent servers must be running
    - You can start them using:
        python hmcp_llm_demo.py --emr-server
        python hmcp_llm_demo.py --patient-data-server
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

from dotenv import load_dotenv
from hmcp.client.client_connector import HMCPClientConnector
from mcp.shared.exceptions import McpError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Server configuration
HOST = os.getenv("HOST", "localhost")
EMR_PORT = int(os.getenv("WRITEBACK_PORT", "8050"))
PATIENT_PORT = int(os.getenv("PATIENT_DATA_PORT", "8060"))


async def run_emr_demo(host: str, port: int) -> None:
    """Run the EMR writeback demo"""
    logger.info("Running EMR writeback demo...")

    # Create helper for EMR server
    url = f"http://{host}:{port}"
    emr_helper = HMCPClientConnector(url=url, debug=True)

    try:
        # Connect to EMR server
        logger.info("Connecting to EMR Writeback Server")
        emr_info = await emr_helper.connect(transport="streamable-http")
        logger.info(f"Connected to {emr_info['name']} v{emr_info['version']}")

        # List available tools
        logger.info("Listing tools from EMR Server")
        emr_tools = await emr_helper.list_tools()
        logger.info(f"EMR Server has {len(emr_tools)} tools")

        # Send a message to EMR server requesting clinical data processing
        logger.info("Sending clinical data to EMR Server")
        clinical_data = 'clinical_data={"diagnosis": "Hypertension", "blood_pressure": "140/90", "medication": "Lisinopril 10mg"}'

        emr_response = await emr_helper.create_message(clinical_data)
        logger.info(f"EMR Response: {emr_response['content']}")

    finally:
        # Clean up connections
        logger.info("Cleaning up EMR connection")
        await emr_helper.cleanup()


async def run_patient_demo(host: str, port: int) -> None:
    """Run the patient data demo"""
    logger.info("Running patient data demo...")

    # Create helper for patient data server
    url = f"http://{host}:{port}"
    patient_helper = HMCPClientConnector(url=url, debug=True)

    try:
        # Connect to Patient Data server
        logger.info("Connecting to Patient Data Server")
        patient_info = await patient_helper.connect(transport="streamable-http")
        logger.info(
            f"Connected to {patient_info['name']} v{patient_info['version']}")

        # List available tools
        logger.info("Listing tools from Patient Data Server")
        patient_tools = await patient_helper.list_tools()
        logger.info(f"Patient Data Server has {len(patient_tools)} tools")

        # Send a message to Patient Data server
        logger.info("Requesting patient data")
        patient_request = "I need the patient identifier for John Smith"

        patient_response = await patient_helper.create_message(patient_request)
        logger.info(f"Patient Data Response: {patient_response['content']}")

    finally:
        # Clean up connections
        logger.info("Cleaning up Patient Data connection")
        await patient_helper.cleanup()


async def main():
    """
    Main demo function that demonstrates HMCPServerHelper usage with both
    EMR Writeback and Patient Data Access agents.
    """
    logger.info("Starting HMCP Server Helper Demo")

    if "--emr-only" in sys.argv:
        await run_emr_demo(HOST, EMR_PORT)
    elif "--patient-only" in sys.argv:
        await run_patient_demo(HOST, PATIENT_PORT)
    else:
        # Run EMR demo first
        await run_emr_demo(HOST, EMR_PORT)

        # Then run patient data demo
        await run_patient_demo(HOST, PATIENT_PORT)

        logger.info("Full demo completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
