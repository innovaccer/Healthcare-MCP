"""Tests for the shared auth exception hierarchy and package exports."""

from __future__ import annotations

from typing import Any, Dict

from hmcp.shared.auth.exceptions import AuthenticationError, InvalidStateError


def test_invalid_state_error_is_authentication_error() -> None:
    assert issubclass(InvalidStateError, AuthenticationError)


def test_exceptions_module_exports_only_kept_classes() -> None:
    import hmcp.shared.auth.exceptions as exc

    assert set(exc.__all__) == {"AuthenticationError", "InvalidStateError"}


def test_star_import_exposes_only_kept_names() -> None:
    namespace: Dict[str, Any] = {}
    exec("from hmcp.shared.auth import *", namespace)
    exported = {name for name in namespace if not name.startswith("__")}
    assert exported == {"OAuthClient", "AuthenticationError", "InvalidStateError"}
