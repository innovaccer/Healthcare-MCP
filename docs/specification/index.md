# HMCP Specification

Healthcare Model Context Protocol (HMCP) expands on base [MCP](https://modelcontextprotocol.io/specification/2025-03-26) by adding the below enhancements:

- [Authentication \& Scopes](#authentication--scopes)
- [Patient Context](#patient-context)
- [Bi-directional agent to agent communication](#bi-directional-agent-to-agent-communication)
 

## Authentication & Scopes
HMCP implements OAuth 2.0 and OpenID Connect following the SMART on FHIR authorization framework model. This approach provides a standardized, secure method for healthcare applications to authenticate and obtain appropriately scoped access to clinical data.

Two primary authentication mechanisms are supported:

1. **OAuth 2.0 for user-mediated contexts** - When end-user consent and context is needed
2. **mTLS (Mutual TLS)** - For secure service-to-service communication

For detailed implementation examples and flows, see the [Authentication Implementation Guide](./auth.md).

### [Comparison of MCP vs HMCP Authentication](./hmcp_auth_vs_mcp_auth.md)

## Patient Context

HMCP **specifies** patient context sharing based on SMART on FHIR specifications, ensuring healthcare AI agents maintain proper patient context isolation and security. SDK support for it is pending. 

[Patient Context Sharing](./context.md)

## Bi-directional agent to agent communication

Agent to agent bidirectional communication is implemented by adding [sampling](https://modelcontextprotocol.io/specification/2025-03-26/client/sampling) capability to the HMCP server. The base MCP implementation already has sampling implemented on the client. That way HMCP client and server both can communicate using LLM inputs and output.

[Sampling](./sampling.md)

## Architecture Components

### Diagram

```mermaid
---
Title: Health MCP in action
---
graph LR
    subgraph HS[Health System]
        subgraph AIA[AI Agent]
        end
    end

    subgraph HMCP Server
        subgraph GATE[HMCP Gateway]
        end
        subgraph IMA[Agents]
        end
        AIA <--> |HMCP| GATE
        GATE <--> |HMCP| IMA
    end

    subgraph 3PP[3rd party platform]
        subgraph 3PDS[Data store]
        end
        subgraph 3PALGO[Clinical Algorithm]
        end
        GATE <--> |HMCP|3PDS
        GATE <--> |HMCP|3PALGO
    end
```
