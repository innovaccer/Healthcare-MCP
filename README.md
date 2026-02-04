![image info](./images/Innovaccer_HMCP_Github_banner.png)

# Healthcare Model Context Protocol (HMCP)

**_An open protocol enabling communication and interoperability between healthcare agentic applications._**

Healthcare is rapidly embracing an AI-driven future. From ambient clinical documentation to decision support, generative AI agents hold immense promise to transform care delivery. However, as the industry swiftly moves to adopt AI-powered solutions, it faces a significant challenge: ensuring AI agents are secure, compliant, and seamlessly interoperable within real-world healthcare environments.

At Innovaccer, we are proud to launch the Healthcare Model Context Protocol (HMCP). HMCP is a specialized extension of the Model Context Protocol (MCP) specifically crafted to integrate healthcare AI agents with data, tools, and workflows, all within a secure, compliant, and standards-based framework.


## Overview & Motivation

### Overview
MCP Model Context Protocol was created by Anthropic to allow host agentic applications (like Claude Desktop App, Cursor) to communicate with other systems (like local files, API servers) to augment the LLM input with additional context 

#### Why Healthcare Needs HMCP

Healthcare demands precision and accountability. AI agents operating within this domain must handle sensitive patient data securely, adhere to rigorous compliance regulations, and maintain consistent interoperability across diverse clinical workflows. Standard, generalized protocols fall short. That is why we developed HMCP.

Built upon the robust foundation of open source MCP (Model Context Protocol), HMCP introduces essential healthcare-specific capabilities by leveraging industry standard controls (OAuth 2.0, OpenID Connect following SMART on FHIR, Data Segregation & Encryption, Audit trails, Rate Limiting & Risk Assessment, etc.), to achieve:
- HIPAA-compliant security and access management
- Comprehensive logging and auditing of agent activities
- Separation and protection of patient identities
- Bidirectional agent-to-agent communication via sampling endpoints
- Support for both SSE and streamable-http transports
- Facilitation of secure, compliant collaboration between multiple AI agents

These enhancements are being designed to ensure that HMCP can meet the unique regulatory, security, and operational needs of healthcare environments.


**_Think of HMCP as the "universal connector" for healthcare AI—a trusted, standardized way to ensure seamless interoperability._**

![image info](./images/HMCP_In_Action.png)

## Quick Start

### Installing HMCP

```bash
# Temporary steps till the package isn't published:
pip install hatch
hatch build
```

### Creating an HMCP Server

```python
from hmcp.server.hmcp_server import HMCPServer
from mcp.shared.context import RequestContext
import mcp.types as types

# Initialize the server
server = HMCPServer(
    name="Your Agent Name",
    version="1.0.0",
    host="0.0.0.0",
    port=8050,
    debug=True,
    instructions="Your agent's description"
)

# Define a sampling endpoint for agent-to-agent communication
@server.sampling()
async def handle_sampling(context, params):
    # Process incoming messages
    latest_message = params.messages[-1]
    message_content = latest_message.content.text if hasattr(latest_message.content, 'text') else str(latest_message.content)
    
    return types.CreateMessageResult(
        model="your-agent-name",
        role="assistant",
        content=types.TextContent(
            type="text",
            text=f"Processed: {message_content}"
        ),
        stopReason="endTurn"
    )

# Start the server (supports both 'sse' and 'streamable-http' transports)
server.run(transport="streamable-http")
```

### Connecting with an HMCP Client

```python
from hmcp.client.client_connector import HMCPClientConnector
import asyncio

async def connect_to_agent():
    # Create client connector (handles auth and connection automatically)
    client = HMCPClientConnector(
        url="http://localhost:8050",
        debug=True
    )
    
    try:
        # Connect to the server (supports 'sse' or 'streamable-http')
        await client.connect(transport="streamable-http")
        
        # Send a message using simplified interface
        response = await client.create_message(
            message="Your message here",
            role="user"
        )
        
        # Process the response
        print(f"Response: {response.get('content')}")
        
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool['name'] for tool in tools]}")
        
    finally:
        # Clean up connection
        await client.cleanup()

# Run the client
asyncio.run(connect_to_agent())
```

### Multi-Agent Workflows

HMCP supports complex multi-agent workflows where specialized agents collaborate to complete healthcare tasks:

```python
from hmcp.client.client_connector import HMCPClientConnector
from agents import Agent
import asyncio

async def multi_agent_workflow():
    # Connect to specialized healthcare agents
    emr_client = HMCPClientConnector(url="http://localhost:8050", debug=True)
    patient_client = HMCPClientConnector(url="http://localhost:8060", debug=True)
    
    await emr_client.connect(transport="streamable-http")
    await patient_client.connect(transport="streamable-http")
    
    try:
        # Step 1: Query patient data agent
        patient_response = await patient_client.create_message(
            message="Get patient ID for John Smith"
        )
        patient_id = patient_response.get('content')
        
        # Step 2: Update EMR with clinical data
        emr_response = await emr_client.create_message(
            message=f'Update clinical data for {patient_id}: BP 130/85, HR 72'
        )
        
        print(f"Workflow complete: {emr_response.get('content')}")
        
    finally:
        await emr_client.cleanup()
        await patient_client.cleanup()

asyncio.run(multi_agent_workflow())
```

For more detailed examples including multi-agent handoffs and OpenAI agent integration, see:
- [EMR & Patient Data Example](./examples/emr_patient_data_example/)
- [Multi-Agent Handoff Demo](./examples/multi_agent_demo/)

For more detailed examples and advanced usage, see the [HMCP SDK documentation](./src/hmcp/README.md) and [examples directory](./examples/).

## Key Features

### Dual Transport Support
HMCP supports both SSE (Server-Sent Events) and streamable-http transports for flexibility in different deployment scenarios:
- **SSE**: Traditional long-polling approach, ideal for real-time updates
- **streamable-http**: Modern HTTP-based streaming, better firewall compatibility

### OAuth 2.0 Authentication
Built-in OAuth 2.0 support following SMART on FHIR specifications:
- Client credentials flow for server-to-server communication
- Authorization code flow with PKCE for user-facing applications
- Patient-scoped access tokens for data segregation
- Token introspection and revocation

See [OAuth Client Documentation](./src/hmcp/shared/auth/oauth_client_README.md) for detailed usage.

### Simplified Client Interface
The `HMCPClientConnector` provides a simplified interface for:
- Automatic connection management
- Built-in authentication handling
- Tool and resource discovery
- Sampling endpoint communication
- Proper cleanup and resource management

## Specification

[Specification](./docs/specification/index.md)

## HMCP SDK

[HMCP SDK](./src/hmcp/README.md)

## Examples

[Examples](./examples/README.md)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this
project.

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for
details.
