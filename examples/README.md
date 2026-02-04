# HMCP Examples

Healthcare Model Context Protocol (HMCP) examples demonstrating various capabilities and implementations.

## Table of Contents
- [Features](#features)
- [Available Examples](#available-examples)
- [How to Run the Demo](#how-to-run-the-demo)
- [HMCP Demo Workflow](#hmcp-demo-workflow)
- [HMCP LLM Demo (GPT-4o Integration)](#hmcp-llm-demo-gpt-4o-integration)
- [Multi-Agent Handoff Architecture](#multi-agent-handoff-architecture)

## Features
- Bidirectional communication
- Authentication and Authorization

## Available Examples
- **Basic HMCP Demo**: Demonstrates core HMCP communication features
- **HMCP LLM Demo**: Integrates OpenAI's GPT-4o with HMCP
- **Multi-Agent Handoff**: Showcases complex workflows with multiple specialized agents

## How to Run the Demo

```bash
# Temporary steps till the package isn't published:
pip install hatch
hatch build

# Actual steps:
cd examples

# Install required dependencies using below command
pip install -r requirements.txt

# create .env by taking reference from .env.example
# and define below ENV variables 
OPENAI_API_KEY=<your-openai-api-key>  # provide your openai api key here

# START EMR MCP server
# Open a new terminal and execute below commands
python hmcp_demo.py --emr-server

# START PATIENT data MCP server
# Open a new terminal and execute below commands
python hmcp_demo.py --patient-data-server

# Run the DEMO using client server with SSE transport (recommended)
# Open a new terminal and execute below commands
python hmcp_demo.py
```

## HMCP Demo Workflow

### Overview
The demo involves three agents working together:
1. AI Agent (Main Orchestrator)
2. EMR Writeback Agent
3. Patient Data Access Agent

### Detailed Workflow Steps

1. **Initial Setup**
   - AI Agent initializes without authentication (demo mode)
   - Sets up connection to EMR and Patient Data agents
   - Prepares for clinical data workflow

2. **EMR Writeback Agent Connection**
   - AI Agent connects to EMR Writeback Agent
   - Establishes SSE (Server-Sent Events) connection
   - Initializes EMR session

3. **First Clinical Data Submission**
   - AI Agent sends initial clinical data to EMR Writeback Agent
   - Data includes diagnosis, blood pressure, and medication information
   - Format: {"diagnosis": "Hypertension", "blood_pressure": "140/90", "medication": "Lisinopril 10mg"}



5. **Patient Data Access**
   - EMR Writeback Agent requests additional information (patient ID)
   - AI Agent connects to Patient Data Access Agent
   - Requests patient identifier for "John Smith"
   - Receives patient ID (PT12345)

6. **Final EMR Update**
   - AI Agent sends complete data to EMR Writeback Agent
   - Includes both clinical data and patient ID
   - Receives confirmation of successful EMR update

7. **Workflow Completion**
   - All required information is processed
   - EMR system is updated successfully
   - Demo workflow is completed

Note: The demo includes error handling and logging at each step to ensure proper execution and debugging capabilities.

## HMCP LLM Demo (GPT-4o Integration)

The `hmcp_llm_demo.py` file demonstrates an advanced version of the HMCP workflow that integrates OpenAI's GPT-4o to create intelligent agents that can generate dynamic, context-aware responses.

### Overview
Unlike the basic demo that uses predefined responses, the LLM demo employs three LLM-powered agents:

1. **AI Agent (Main Orchestrator)**: Uses GPT-4o to plan and coordinate the entire workflow
2. **EMR Writeback Agent**: Uses GPT-4o to intelligently respond to clinical data submission requests
3. **Patient Data Access Agent**: Uses GPT-4o to process and respond to patient identification requests

Each agent maintains its own conversation history and system prompt that defines its role and behavior.

### LLM Demo Features
- Context-aware responses from all agents
- Dynamic workflow planning by the AI Orchestrator

- Simulated patient database with intelligent lookups
- Conversation history maintenance for coherent multi-turn interactions

### How to Run the LLM Demo

```bash
# Ensure you have the required dependencies
cd examples
pip install -r requirements.txt

# create .env file with your OpenAI API key (REQUIRED for LLM integration)
# You must have access to GPT-4o or specify another model using LLM_MODEL environment variable
OPENAI_API_KEY=<your-openai-api-key>

# Optional environment variables
HOST=0.0.0.0  # Default host
WRITEBACK_PORT=8050  # Default EMR Writeback Agent port
PATIENT_DATA_PORT=8060  # Default Patient Data Access Agent port 
LOG_LEVEL=DEBUG  # Default log level
LLM_MODEL=gpt-4o  # Default LLM model

# START EMR Writeback Agent server (LLM-powered)
# Open a new terminal and execute:
python hmcp_llm_demo.py --emr-server

# START Patient Data Access Agent server (LLM-powered)
# Open a new terminal and execute:
python hmcp_llm_demo.py --patient-data-server

# Run the LLM Demo using the AI Agent orchestrator
# Open a new terminal and execute:
python hmcp_llm_demo.py
```

### LLM Demo Workflow Steps

1. **AI Agent Planning**: The AI Agent uses GPT-4o to generate a comprehensive plan for the workflow.

2. **EMR Connection & Data Submission**: The AI Agent connects to the EMR Writeback Agent and submits clinical data about a hypertension patient.

3. **Intelligent Response**: The EMR Agent (using GPT-4o) determines that it needs a patient identifier and responds accordingly.



5. **Dynamic Decision Making**: The AI Agent analyzes the EMR response and determines it needs to get a patient identifier.

6. **Patient Data Retrieval**: The AI Agent connects to the Patient Data Access Agent and requests information for "John Smith".

7. **Context-Aware Extraction**: The AI Agent intelligently extracts the patient ID from the response.

8. **Complete Data Submission**: The AI Agent sends the complete clinical data with the patient ID to the EMR Writeback Agent.

9. **Workflow Summary**: The AI Agent generates a summary of the completed workflow using GPT-4o.

### Workflow Sequence Diagram

```mermaid
sequenceDiagram
    participant AI as AI Agent (Orchestrator)<br>HMCP Client
    participant EMR as EMR Writeback Agent<br>HMCP Server
    participant Patient as Patient Data Access Agent<br>HMCP Server
    participant GPT4o as GPT-4o API

    Note over AI: Step 1: Workflow Planning
    AI->>GPT4o: Request workflow plan
    GPT4o-->>AI: Return clinical data workflow plan
    
    Note over AI: Step 2: EMR Connection & Data Submission
    AI->>EMR: Connect and initialize session
    EMR-->>AI: Connection established
    AI->>GPT4o: Generate clinical data structure
    GPT4o-->>AI: Return structured clinical data
    AI->>EMR: Submit clinical data (no patient_id)
    
    Note over EMR: Step 3: Intelligent Response
    EMR->>GPT4o: Process request with EMR system prompt
    GPT4o-->>EMR: Determine patient_id is required
    EMR-->>AI: Request patient identifier
    

    
    Note over AI: Step 5: Dynamic Decision Making
    AI->>GPT4o: Analyze EMR response
    GPT4o-->>AI: Determine patient data is needed
    
    Note over AI: Step 6: Patient Data Retrieval
    AI->>Patient: Connect and initialize session
    Patient-->>AI: Connection established
    AI->>Patient: Request patient ID for "John Smith"
    Patient->>GPT4o: Process request with Patient system prompt
    GPT4o-->>Patient: Generate patient identifier response
    Patient-->>AI: Return patient_id=PT12345
    
    Note over AI: Step 7: Context-Aware Extraction
    AI->>GPT4o: Extract patient ID from response
    GPT4o-->>AI: Return extracted patient ID
    
    Note over AI: Step 8: Complete Data Submission
    AI->>EMR: Submit complete data with patient_id
    EMR->>GPT4o: Process complete request
    GPT4o-->>EMR: Generate success response
    EMR-->>AI: Confirm successful EMR update
    
    Note over AI: Step 9: Workflow Summary
    AI->>GPT4o: Generate workflow summary
    GPT4o-->>AI: Return comprehensive summary
    Note over AI: Clinical workflow complete
```

## Multi-Agent Handoff Architecture

For more complex workflows involving multiple specialized agents, see the [Multi-Agent Handoff documentation](multi_agent_handoff.md), which details:

- Architecture for orchestrating complex workflows between specialized agents
- Dynamic routing of messages based on context and needs
- Implementation of secure and compliant healthcare information exchange
- Examples of multi-agent collaboration in healthcare settings

To run the multi-agent handoff demo:

```bash
# Ensure you have the required dependencies
cd examples
pip install -r requirements.txt

# Set up your environment variables (see .env.example)
# Open separate terminals for each server:

# Start the EMR server
python hmcp_llm_demo.py --emr-server

# Start the Patient Data server
python hmcp_llm_demo.py --patient-data-server

# Run the multi-agent handoff demo
python multi_handoff_agent_demo.py
```

## Technical Implementation Details

- Uses `AsyncOpenAI` client for non-blocking LLM calls
- Implements conversation history management for each agent
- Provides specialized system prompts for each agent type
- Uses SSE (Server-Sent Events) for communication
- Implements JWT-based authentication between agents
- Includes comprehensive error handling and logging