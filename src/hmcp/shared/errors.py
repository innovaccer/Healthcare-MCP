"""HMCP error codes and the MCPError factory.

Codes are outside the JSON-RPC reserved range (-32768..-32000) as the 2026
spec requires. Default messages are short and generic and never contain an
identifier, a patient reference, a tool argument or a token.
"""

from enum import IntEnum

from mcp.shared.exceptions import MCPError


class HMCPErrorCode(IntEnum):
    """HMCP wire error codes, outside the JSON-RPC reserved range.

    Each code is carried as ``error.data["hmcpCode"]`` (the member's name) by
    :func:`hmcp_error`.
    """

    PATIENT_CONTEXT_REQUIRED = -31001  # patient-scoped tool called without a contextId
    PATIENT_CONTEXT_INVALID = -31002  # unknown, expired, closed, or other principal
    PATIENT_CONTEXT_REVOKED = -31003
    PATIENT_CONTEXT_SUSPENDED = -31004  # access under review
    PATIENT_NOT_AUTHORIZED = -31005  # open denied, or token patient claim mismatch
    PATIENT_CONTEXT_UNAVAILABLE = -31006  # store/policy/audit failure -> fail closed
    GUARDRAIL_UNAVAILABLE = -31011  # required guardrail the server cannot apply


# Short, generic, identifier-free default messages.
_DEFAULT_MESSAGES: dict[HMCPErrorCode, str] = {
    HMCPErrorCode.PATIENT_CONTEXT_REQUIRED: "Patient context required for this tool",
    HMCPErrorCode.PATIENT_CONTEXT_INVALID: "Patient context is invalid",
    HMCPErrorCode.PATIENT_CONTEXT_REVOKED: "Patient context has been revoked",
    HMCPErrorCode.PATIENT_CONTEXT_SUSPENDED: "Patient context is suspended",
    HMCPErrorCode.PATIENT_NOT_AUTHORIZED: "Not authorized for this patient",
    HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE: "Patient context is unavailable",
    HMCPErrorCode.GUARDRAIL_UNAVAILABLE: "Required guardrail is unavailable",
}


def hmcp_error(
    code: HMCPErrorCode,
    message: str | None = None,
    data: dict | None = None,
) -> MCPError:
    """Build (do not raise) an MCPError carrying an HMCP error code."""
    return MCPError(
        code=int(code),
        message=message or _DEFAULT_MESSAGES[code],
        data={**(data or {}), "hmcpCode": code.name},
    )
