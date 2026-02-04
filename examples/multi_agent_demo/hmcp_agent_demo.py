#!/usr/bin/env python3
"""
HMCP Agent Demo: Multi-Agent Workflow with OpenAI Agents and HMCP Servers

This script demonstrates how to create a multi-agent workflow where OpenAI agents
communicate with HMCP servers using the HMCPAgent class. It sets up a workflow
similar to the one in hmcp_llm_demo.py but using the more structured agent framework.

The workflow involves three agents:
1. Main OpenAI Agent - The primary agent that orchestrates the workflow
2. EMR Writeback Agent - HMCP server agent that handles electronic medical records
3. Patient Data Agent - HMCP server agent that provides patient information

Usage:
    # Ensure both HMCP servers are running first:
    python hmcp_llm_demo.py --emr-server
    python hmcp_llm_demo.py --patient-data-server

    # Then run this demo:
    python hmcp_agent_demo.py

Requirements:
    - OpenAI API key in .env file (OPENAI_API_KEY)
    - Both EMR Writeback Agent and Patient Data Access Agent servers running
    - agents package installed (pip install agents)
"""

from hmcp.client.client_connector import HMCPClientConnector
from .agent import HMCPAgent
import asyncio
import os
import logging
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Import required agents and HMCP modules
# try:
from agents import Agent, Runner
from agents.mcp import MCPServerSse

AGENTS_AVAILABLE = True
# except ImportError:
#     AGENTS_AVAILABLE = False
#     print("Warning: agents package not installed. Install with: pip install agents")

# Import HMCP modules

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Server configuration (same as in other examples)
HOST = os.getenv("HOST", "localhost")
EMR_PORT = int(os.getenv("WRITEBACK_PORT", "8050"))
PATIENT_PORT = int(os.getenv("PATIENT_DATA_PORT", "8060"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class WorkflowContext:
    """Context object for sharing data between agents in the workflow."""

    def __init__(self):
        self.clinical_data = None
        self.patient_id = None
        self.emr_response = None
        self.patient_response = None
        self.final_result = None

    def add_clinical_data(self, data: str):
        self.clinical_data = data

    def add_patient_id(self, patient_id: str):
        self.patient_id = patient_id

    def add_emr_response(self, response: str):
        self.emr_response = response

    def add_patient_response(self, response: str):
        self.patient_response = response

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the workflow as a dictionary."""
        return {
            "clinical_data": self.clinical_data,
            "patient_id": self.patient_id,
            "emr_response": self.emr_response,
            "patient_response": self.patient_response,
            "final_result": self.final_result,
        }


async def create_emr_agent() -> HMCPAgent:
    """Create and initialize the EMR Writeback HMCP Agent."""
    logger.info("Creating EMR Writeback Agent")

    # Create helper for EMR server
    emr_helper = HMCPClientConnector(
        url=f"http://{HOST}:{EMR_PORT}", debug=True)

    # Connect to the EMR server
    await emr_helper.connect(transport="streamable-http")
    logger.info(f"Connected to EMR server: {emr_helper.server_info['name']}")

    # Create the HMCPAgent wrapping the helper
    emr_agent = HMCPAgent(
        name="EMR Writeback Agent",
        hmcp_helper=emr_helper,
        handoff_description="Handles writing clinical data to electronic medical records. "
        "Requires patient_id and clinical_data parameters.",
        instructions="You are the EMR Writeback Agent, a specialized healthcare AI assistant "
        "responsible for securely writing clinical data to Electronic Medical Records.",
        settings={
            "max_tokens": 200,
            "temperature": 0.7,
        },
    )

    return emr_agent


async def create_patient_data_agent() -> HMCPAgent:
    """Create and initialize the Patient Data HMCP Agent."""
    logger.info("Creating Patient Data Agent")

    # Create helper for patient data server
    patient_helper = HMCPClientConnector(
        url=f"http://{HOST}:{PATIENT_PORT}",
        debug=True
    )

    # Connect to the Patient Data server
    await patient_helper.connect(transport="streamable-http")
    logger.info(
        f"Connected to Patient Data server: {patient_helper.server_info['name']}"
    )

    # Create the HMCPAgent wrapping the helper
    patient_agent = HMCPAgent(
        name="Patient Data Agent",
        hmcp_helper=patient_helper,
        handoff_description="Provides access to patient identifier information. "
        "Can look up patient IDs by patient names.",
        instructions="You are the Patient Data Access Agent, a specialized healthcare AI assistant "
        "that helps healthcare providers securely access patient identification information.",
        settings={
            "max_tokens": 200,
            "temperature": 0.7,
        },
    )

    return patient_agent


async def create_main_agent(emr_agent: HMCPAgent, patient_agent: HMCPAgent) -> Agent:
    """Create the main OpenAI agent that orchestrates the workflow."""
    if not AGENTS_AVAILABLE:
        raise ImportError(
            "The agents package is required to create the main agent.")

    # Convert HMCP agents to standard agents for handoffs
    emr_standard_agent = emr_agent.to_agent()
    patient_standard_agent = patient_agent.to_agent()

    # Create the main agent with handoffs to the HMCP agents
    main_agent = Agent(
        name="Healthcare Workflow Orchestrator",
        instructions="""You are the AI Orchestrator Agent, a specialized healthcare AI system 
        that coordinates clinical workflows between different medical systems.
        
        Your responsibilities:
        1. Manage the flow of clinical data between healthcare systems
        2. Ensure all necessary patient identifiers are obtained before writing to EMR
        3. Coordinate communication with specialized agents for different tasks
        4. Maintain proper medical context throughout the workflow
        5. Follow healthcare data handling best practices
        
        You're currently managing a workflow to write clinical data to an Electronic Medical Record
        system. You must obtain both clinical data and a valid patient ID before the data can be
        written to the EMR.
        
        Available specialized agents:
        - EMR Writeback Agent: For writing data to medical records (requires patient_id)
        - Patient Data Agent: For looking up patient identifiers by name
        
        When communicating between systems:
        - Be explicit about required data formats
        - Double check that patient identifiers match expected formats (PT12345)
        - Log each step of the workflow clearly
        - Handle error conditions gracefully
        - Ensure data integrity throughout the process""",
        model="gpt-4o",
        handoffs=[emr_standard_agent, patient_standard_agent],
    )

    return main_agent


async def run_workflow_demo():
    """Run the multi-agent workflow demonstration."""
    logger.info("Starting HMCP Agent multi-agent workflow demo")

    if not AGENTS_AVAILABLE:
        logger.error(
            "Agents package is not available. Please install it first.")
        print("\nERROR: This demo requires the 'agents' package to be installed.")
        print("Install it with: pip install agents")
        return

    if not OPENAI_API_KEY:
        logger.error(
            "OpenAI API key not found. Set OPENAI_API_KEY in .env file.")
        print("\nERROR: OpenAI API key not found. Set OPENAI_API_KEY in .env file.")
        return

    try:
        # Create context object for the workflow
        context = WorkflowContext()

        # Create the HMCP agents
        emr_agent = await create_emr_agent()
        patient_agent = await create_patient_data_agent()

        # Create the main orchestrator agent
        main_agent = await create_main_agent(emr_agent, patient_agent)

        # Start the workflow with an initial prompt
        initial_prompt = """I need to submit the following clinical data to the EMR system:
        clinical_data={"diagnosis": "Hypertension", "blood_pressure": "140/90", "medication": "Lisinopril 10mg"}
        
        The patient's name is John Smith. Please help me complete this workflow.
        """

        logger.info("Starting the workflow with the main agent")

        # Use the agents framework to run the workflow
        result = await Runner.run(
            starting_agent=main_agent, input=initial_prompt, context=context
        )

        # Print the workflow results
        logger.info("Workflow completed!")

        # Log the conversation
        for item in result.new_items:
            if hasattr(item, "role") and hasattr(item, "content"):
                logger.info(f"\n[{item.role}]: {item.content}")

        # Clean up resources when done
        await emr_agent.cleanup()
        await patient_agent.cleanup()

        logger.info("Demo completed successfully")

    except Exception as e:
        logger.error(f"Error in workflow demo: {e}")


async def run_direct_hmcp_agent_demo():
    """
    Run a simpler demo that directly uses the HMCPAgent without the agents framework.

    This is useful for environments where the agents package is not installed.
    """
    logger.info("Starting direct HMCP Agent demo (without agents framework)")

    try:
        # Create the EMR agent
        emr_agent = await create_emr_agent()

        try:
            # Send a message directly to the EMR agent
            logger.info("Sending message to EMR agent")

            # Initial message without patient ID
            response1 = await emr_agent.process_message(
                'clinical_data={"diagnosis": "Hypertension", "blood_pressure": "140/90", "medication": "Lisinopril 10mg"}'
            )

            logger.info(f"EMR agent response: {response1.get('content', '')}")

            # Create the Patient Data agent
            patient_agent = await create_patient_data_agent()

            try:
                # Get patient ID from Patient Data agent
                logger.info("Sending message to Patient Data agent")

                response2 = await patient_agent.process_message(
                    "I need the patient identifier for John Smith"
                )

                logger.info(
                    f"Patient Data agent response: {response2.get('content', '')}"
                )

                # Extract patient ID
                patient_id = "PT12345"  # Default fallback
                patient_response = response2.get("content", "")
                if "patient_id=" in patient_response:
                    patient_id = patient_response.split("patient_id=")[
                        1].split()[0]

                # Complete the EMR request with patient ID
                logger.info("Sending complete message to EMR agent")

                response3 = await emr_agent.process_message(
                    f'clinical_data={{"diagnosis": "Hypertension", "blood_pressure": "140/90", "medication": "Lisinopril 10mg"}} patient_id={patient_id}'
                )

                logger.info(
                    f"Final EMR agent response: {response3.get('content', '')}")

            finally:
                await patient_agent.cleanup()

        finally:
            await emr_agent.cleanup()

        logger.info("Direct demo completed successfully")

    except Exception as e:
        logger.error(f"Error in direct demo: {e}")


async def main():
    """Main entry point for the demo."""
    print("\nHMCP Agent Demo: Multi-Agent Workflow\n")
    print("This script demonstrates using HMCPAgent to build multi-agent workflows")
    print("between OpenAI agents and HMCP servers with sampling capability.\n")

    if AGENTS_AVAILABLE:
        print("Running complete multi-agent workflow demo (using agents framework)...")
        await run_workflow_demo()
    else:
        print("Running simplified direct HMCP agent demo...")
        await run_direct_hmcp_agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
