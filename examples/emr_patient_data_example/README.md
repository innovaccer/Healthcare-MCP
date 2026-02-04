# EMR and Patient Data Example

This example demonstrates integration between an EMR system and patient data service using multiple specialized agents.

## Prerequisites

- Python 3.11 or above
- Environment variables configured (see `.env.example`)

## Setup Instructions

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Load environment variables:
   ```bash
   export $(cat .env | xargs)  # On Windows use: set /p VARS=<.env
   ```

## Running the Example

1. Start the EMR server in a terminal:
   ```bash
   python examples/emr_patientdata_example/run_standalone_emr_server.py
   ```

2. Start the Patient Data server in another terminal:
   ```bash
   python examples/emr_patientdata_example/run_standalone_patient_data_server.py
   ```

3. Run the client to test the integration:
   ```bash
   python examples/emr_patientdata_example/hmcp_client_standalone_connection.py
   ```

## What to Expect

The example will demonstrate:
- Connection establishment between services
- Patient data retrieval and EMR updates
- Secure information exchange between agents
- Error handling and validation

Check the console output in each terminal to observe the interaction between components.