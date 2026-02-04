#!/usr/bin/env python3
"""
Multi-Handoff Agent Demo: Coordinating Multiple Agents for Complex Healthcare Tasks

This script demonstrates how to use the MultiHandoffAgent to coordinate communication
between multiple specialized healthcare agents to complete a complex objective.

The workflow uses three agents:
1. Main OpenAI Agent - The primary orchestrating agent
2. EMR Writeback Agent - HMCP agent for electronic medical records
3. Patient Data Agent - HMCP agent for patient information

Usage:
    # Ensure both HMCP servers are running first:
    python hmcp_llm_demo.py --emr-server
    python hmcp_llm_demo.py --patient-data-server

    # Then run this demo:
    python multi_handoff_agent_demo.py

Requirements:
    - OpenAI API key in .env file (OPENAI_API_KEY)
    - Both EMR Writeback Agent and Patient Data Access Agent servers running
    - agents package installed (pip install agents)
"""

from multi_agent_handoff import MultiHandoffAgent
from hmcp.client.client_connector import HMCPClientConnector
from agent import HMCPAgent
from agents.model_settings import ModelSettings
from agents.run import RunConfig
from agents.agent_output import AgentOutputSchema
from agents import Agent, Runner
import asyncio
import os
import logging
import sys
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Disable OpenAI tracing
os.environ["OPENAI_TRACING_ENABLED"] = "false"

# Import required agents and HMCP modules

# Import our custom agents and wrappers

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
        self.patient_id = None
        self.clinical_data = {}
        self.emr_status = None


class EMRWritebackResult(BaseModel):
    """Result model for EMR Writeback Agent."""

    status: str
    message: str
    patient_id: Optional[str] = None
    clinical_data: Optional[Dict[str, Any]] = None


class PatientDataResult(BaseModel):
    """Result model for Patient Data Agent."""

    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    message: str


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


async def create_orchestrator_agent(
    emr_agent: HMCPAgent, patient_agent: HMCPAgent
) -> Agent:
    """Create the main OpenAI agent that orchestrates the workflow."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY in .env file.")

    # Convert HMCP agents to standard agents for handoffs
    emr_standard_agent = emr_agent.to_agent()
    patient_standard_agent = patient_agent.to_agent()

    # Create the main agent with handoffs to the HMCP agents
    orchestrator_agent = Agent(
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
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    return orchestrator_agent


async def run_multi_handoff_demo():
    """Run the multi-handoff agent demonstration."""
    logger.info("Starting Multi-Handoff Agent demo")

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

        # Create the orchestrator agent
        orchestrator_agent = await create_orchestrator_agent(emr_agent, patient_agent)

        # Define a complex healthcare objective that requires multiple handoffs
        initial_prompt = """
        I need to complete a comprehensive patient update workflow for John Smith:
        
        1. Get John Smith's patient identifier from the patient database
        2. Update his clinical records with new information:
           - Blood pressure: 130/85
           - Heart rate: 72 bpm
           - Diagnosis: Controlled hypertension
           - New medication: Lisinopril 10mg daily
        3. Verify the information was successfully recorded in the EMR system
        
        Please coordinate with all necessary systems to complete this workflow efficiently.
        """

        # Create the multi-handoff agent
        multi_agent = MultiHandoffAgent(
            base_agent=orchestrator_agent,
            max_iterations=5,  # Limit to 5 iterations for demo purposes
        )

        # Run the multi-handoff agent workflow
        result = await multi_agent.run(
            initial_prompt=initial_prompt,
            context=context,
            run_config=RunConfig(workflow_name="Healthcare Update Workflow"),
        )

        # Print the final result
        print(f"\nFinal result: {result.final_output}")

        # Print the conversation history
        print("\nConversation history:")
        for i, message in enumerate(multi_agent.get_conversation_history()):
            print(
                f"{i+1}. [{message['role']}]: {message['content'][:100]}..."
                if len(message["content"]) > 100
                else f"{i+1}. [{message['role']}]: {message['content']}"
            )

        # Print the handoff history
        print("\nHandoff history:")
        for i, handoff in enumerate(multi_agent.get_handoff_history()):
            print(
                f"{i+1}. [{handoff.agent_name}]: {handoff.response[:100]}..."
                if len(handoff.response) > 100
                else f"{i+1}. [{handoff.agent_name}]: {handoff.response}"
            )

        # Clean up resources
        await emr_agent.cleanup()
        await patient_agent.cleanup()

        logger.info("Multi-handoff demo completed successfully")

    except Exception as e:
        logger.error(f"Error in multi-handoff demo: {e}")
        raise


async def main():
    """Main entry point for the demo."""
    print("\nMulti-Handoff Agent Demo: Healthcare Workflow Orchestration\n")
    print(
        "This demo shows how MultiHandoffAgent coordinates multiple healthcare agents"
    )
    print("to complete a complex patient record update workflow.\n")

    # Check if services are running
    print("Checking if required services are running...")

    try:
        # Run the multi-handoff demo
        await run_multi_handoff_demo()
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        print(f"\nERROR: {str(e)}")
        print("\nMake sure both server components are running:")
        print("1. Start EMR server:    python hmcp_llm_demo.py --emr-server")
        print("2. Start Patient server: python hmcp_llm_demo.py --patient-data-server")


if __name__ == "__main__":
    asyncio.run(main())
