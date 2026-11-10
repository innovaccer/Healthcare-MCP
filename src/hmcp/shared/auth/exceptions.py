"""Authentication exception hierarchy for :mod:`hmcp.shared.auth`.

`AuthenticationError` is the base class for every authentication failure
raised by the OAuth client. `InvalidStateError` is the one subclass that is
raised today: when an authorization redirect's `state` parameter is missing
or does not match the expected value.
"""


class AuthenticationError(Exception):
    """Base class for authentication errors."""


class InvalidStateError(AuthenticationError):
    """Raised when an authorization redirect's state parameter is invalid."""


__all__ = ["AuthenticationError", "InvalidStateError"]
