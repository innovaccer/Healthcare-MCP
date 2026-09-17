# Comparison of MCP and HMCP Authentication Implementations

Below are the key differences between HMCP authentication and MCP authentication key differences between them:

## 1. Base Standards
- **MCP**: Built on OAuth 2.1 (a newer iteration of OAuth)
- **HMCP**: Adopts SMART on FHIR framework, which builds upon OAuth 2.0 and OpenID Connect standards

## 2. Healthcare Specificity
- **MCP**: General-purpose protocol without healthcare-specific considerations
- **HMCP**: Specifically designed for healthcare applications with healthcare data exchange in mind

## 3. Authentication Flows
- **MCP**: Implements standard OAuth 2.1 flows with emphasis on authorization code grant
- **HMCP**: Clearly defines two specific flows:
  - OAuth 2.0 Authorization Code Flow (user-mediated)
  - Client Credentials Flow (service-to-service)

## 4. Scopes System
- **MCP**: Uses generic OAuth scopes
- **HMCP**: Implements healthcare-specific scopes with a particular format:
  - `[patient/][(read|write)].[resource]`
  - Includes healthcare-specific scopes like `patient/hmcp:read` and `launch/patient`

## 5. Patient Context
- **MCP**: No patient context mechanism defined
- **HMCP**: Specifies extensive patient context support (specified; SDK support pending):
  - Patient-prefixed scopes (e.g., `patient/hmcp:read`)
  - Patient parameter in token responses
  - Integration with FHIR resources

## 6. Token Implementation
- **MCP**: Uses bearer tokens and follows OAuth 2.1 Section 5 requirements
- **HMCP**: Specifically requires JWTs with healthcare-specific claims including:
  - `patient`: Patient identifier
  - `fhirUser`: FHIR practitioner reference
  - `tenant`: Organization identifier

## 7. OpenID Connect Integration
- **MCP**: Mentions OpenID Connect but doesn't detail specific integration points
- **HMCP**: Provides comprehensive OpenID Connect details including:
  - ID Token handling
  - UserInfo endpoint usage
  - Standard claims including healthcare-specific ones

## 8. Security Requirements
- **MCP**: General OAuth 2.1 security best practices
- **HMCP**: Additional healthcare-specific security guidance tailored to protected health information

The HMCP implementation represents a specialized adaptation of authentication standards focused on healthcare use cases, while the base MCP specification provides a more general-purpose authentication framework. HMCP adds significant healthcare-specific functionality, particularly around patient context handling and integration with FHIR resources (patient context is specified but not yet implemented in the SDK).