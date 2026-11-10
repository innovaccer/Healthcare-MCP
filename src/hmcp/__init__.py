"""Healthcare Model Context Protocol (HMCP).

HMCP is a Python SDK layered on the Model Context Protocol (MCP) that adds two
healthcare extensions: a *patient-context* extension, which binds tool calls to
an explicitly opened patient and fails closed when no context is open, and a
*guardrails* extension, which runs best-effort checks on tool inputs and outputs
(for example PHI redaction and pseudonymization). HMCP makes no accuracy,
compliance or patient-safety guarantee; see the Disclaimer in the project README.
The server-side entry point lives in :mod:`hmcp.server` and the client-side
connector in :mod:`hmcp.client`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
