# HMCP Python SDK

**_Python implementation of the Healthcare Model Context Protocol (HMCP)_**

## Overview

HMCP SDK builds on top of [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), where we implemented the [HMCP specification](../../docs/specification/index.md).

- Authentication, Authorization & Scopes - Implemented
- Patient Context - To-Do

- Agent to Agent communication - Implemented

## Installation

### Adding HMCP to your python project

```bash
# Temporary steps till the package isn't published:
pip install hatch
hatch build
```

## Usage

### Creating an HMCP Server

You can create an HMCP-compliant server using the `HMCPServer` class:

```python
from hmcp.mcpserver.hmcp_server import HMCPServer

from mcp.shared.context import RequestContext
import mcp.types as types

# Initialize the server
server = HMCPServer(
    name="Your Agent Name",
    version="1.0.0",
    host="0.0.0.0",  # Allow connections from any IP
    port=8050,       # Port for your server
    debug=True,      # Enable debug mode for development
    log_level="DEBUG",
    instructions="Description of what your agent does."
)



# Define a sampling endpoint
@server.sampling()
async def handle_sampling(
    context: RequestContext[Any, Any],
    params: types.CreateMessageRequestParams
) -> types.CreateMessageResult:
    """Handle sampling requests."""
    
    # Extract message content
    latest_message = params.messages[-1]
    message_content = ""
    if isinstance(latest_message.content, list):
        message_content = "".join([
            content.text for content in latest_message.content
            if isinstance(content, types.TextContent)
        ])
    elif isinstance(latest_message.content, types.TextContent):
        message_content = latest_message.content.text
    

    
    # Process the message and return a response
    return types.CreateMessageResult(
        model="your-agent-name",
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Your response here"
        ),
        stopReason="endTurn"
    )

# Start the server
server.run(transport="sse")  # Use SSE (Server-Sent Events) as the transport
```

### Creating an HMCP Client

You can create a client to connect to HMCP servers:

```python
import asyncio
from hmcp.client.hmcp_client import HMCPClient
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import SamplingMessage, TextContent

async def connect_to_hmcp_server():
    # Connect to the HMCP server (no authentication required for demo)
    async with sse_client(f"http://localhost:8050/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize HMCP client
            client = HMCPClient(session)
            init_result = await session.initialize()
            print(f"Connected to {init_result.serverInfo.name}")
            
            # Send a message
            message = SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Your message here"
                )
            )
            
            result = await client.create_message(messages=[message])
            
            # Process the response
            if hasattr(result, 'content') and hasattr(result.content, 'text'):
                response = result.content.text
                print(f"Response: {response}")

# Run the client
asyncio.run(connect_to_hmcp_server())
```

### Agent to Agent Communication Example

See the `examples/hmcp_demo.py` file for a complete example of multi-agent communication using HMCP. The demo implements a clinical data workflow with three agents:

1. AI Agent - Central agent that orchestrates the workflow
2. EMR Writeback Agent - Agent that handles writing to electronic medical records
3. Patient Data Access Agent - Agent that provides patient identifier information

The demo shows how agents can communicate with each other, request additional information, and coordinate workflows without authentication requirements.

## Development

### Setting Up Development Environment

1. Clone the repository
2. Install the package in development mode with test dependencies:
```bash
pip install -e ".[dev]"
```

### Running Tests

The project uses pytest for testing. Here are some common test commands:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=hmcp

# Run tests in verbose mode
pytest -v

# Run specific test file
pytest tests/test_specific_file.py

# Run tests matching a specific pattern
pytest -k "test_pattern"
```

### Code Style

The project follows PEP 8 style guidelines. You can check your code style using:

```bash
# Install flake8 if not already installed
pip install flake8
```

### TODO
Need to replace with uv (Python package and project manager)