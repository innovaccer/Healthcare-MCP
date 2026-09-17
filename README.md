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

Built upon the robust foundation of open source MCP (Model Context Protocol), HMCP is designed around industry standard controls (OAuth 2.0, OpenID Connect following SMART on FHIR, Data Segregation & Encryption, Audit trails, Rate Limiting & Risk Assessment, etc.) to introduce essential healthcare-specific capabilities and achieve:
- HIPAA-compliant security and access management *(partially implemented — opt-in OAuth 2.0 / OIDC authentication only; encryption, audit logging and rate limiting are not yet implemented)*
- Comprehensive logging and auditing of agent activities *(specified; not yet implemented)*
- Separation and protection of patient identities *(specified; not yet implemented — see [Patient Context](docs/specification/context.md))*
- Bidirectional agent-to-agent communication via sampling endpoints *(implemented)*
- Support for both SSE and streamable-http transports *(implemented; `stdio` is also supported and is the default)*
- Facilitation of secure, compliant collaboration between multiple AI agents

These enhancements are being designed to ensure that HMCP can meet the unique regulatory, security, and operational needs of healthcare environments.

#### Implementation status

The list above describes the intended scope of the HMCP protocol and roadmap; individual items are annotated with their current SDK status where applicable. For the authoritative per-feature status, see the [SDK implementation status](src/hmcp/README.md).

**_Think of HMCP as the "universal connector" for healthcare AI—a trusted, standardized way to ensure seamless interoperability._**

![image info](./images/HMCP_In_Action.png)

## Quick Start

### Installing HMCP

Python 3.11 or newer is required.

```bash
# Temporary steps until the package is published:

# Install from a local checkout
pip install .

# ...or build a wheel and install that
# (the wheel filename tracks the project version)
pip install hatch
hatch build
pip install dist/hmcp-0.0.6-py3-none-any.whl
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

# Start the server ('stdio' (default), 'sse' or 'streamable-http')
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
        print(f"Available tools: {[tool.name for tool in tools]}")
        
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

### Transport Support
HMCP supports the following transports for flexibility in different deployment scenarios:
- **stdio**: Standard input/output transport; the default, ideal for local and subprocess use
- **SSE**: Server-to-client messages arrive over a single long-lived SSE (server-push) stream, while client-to-server messages are sent via an HTTP POST endpoint, making the transport bidirectional in effect, ideal for real-time updates
- **streamable-http**: Modern HTTP-based streaming, better firewall compatibility

`stdio` applies to the server (suited to local and subprocess use); `HMCPClientConnector` connects over `sse` or `streamable-http`.

### OAuth 2.0 Authentication
OAuth 2.0 client support following SMART on FHIR specifications (the server enforces authentication only when constructed with an `auth_server_provider`; the default is `None`):
- Client credentials flow for server-to-server communication
- Authorization code flow with PKCE for user-facing applications
- Patient-scoped access tokens for data segregation *(specified; not yet implemented — see [Patient Context](docs/specification/context.md))*
- Token revocation

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
