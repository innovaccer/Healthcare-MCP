"""Tests for hmcp.shared.errors."""

import pytest
from mcp.shared.exceptions import MCPError

from hmcp.shared.errors import HMCPErrorCode, hmcp_error

RESERVED_LOW = -32768
RESERVED_HIGH = -32000


def test_all_codes_outside_jsonrpc_reserved_range():
    for code in HMCPErrorCode:
        assert (
            code.value < RESERVED_LOW or code.value > RESERVED_HIGH
        ), f"{code.name} = {code.value} is inside the JSON-RPC reserved range"


def test_hmcp_error_carries_code_message_and_hmcp_code():
    for code in HMCPErrorCode:
        err = hmcp_error(code)
        assert isinstance(err, MCPError)
        assert err.code == int(code)
        assert isinstance(err.message, str) and err.message.strip()
        assert err.data["hmcpCode"] == code.name


def test_hmcp_error_merges_extra_data():
    err = hmcp_error(
        HMCPErrorCode.PATIENT_CONTEXT_REQUIRED, data={"reason": "synthetic"}
    )
    assert err.data["hmcpCode"] == "PATIENT_CONTEXT_REQUIRED"
    assert err.data["reason"] == "synthetic"


def test_caller_data_cannot_relabel_code():
    code = HMCPErrorCode.PATIENT_CONTEXT_REQUIRED
    err = hmcp_error(code, data={"hmcpCode": "SPOOFED"})
    assert err.data["hmcpCode"] == code.name


def test_custom_message_overrides_default():
    code = HMCPErrorCode.PATIENT_CONTEXT_REQUIRED
    default = hmcp_error(code).message
    custom = hmcp_error(code, message="Custom synthetic message").message
    assert custom == "Custom synthetic message"
    assert custom != default
