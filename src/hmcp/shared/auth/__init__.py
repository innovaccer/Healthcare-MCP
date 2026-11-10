"""HMCP shared authentication package.

Provides the OAuth 2.1 client (`OAuthClient`) and the exception hierarchy it
raises: `AuthenticationError` and its `InvalidStateError` subclass.
"""

from .oauth_client import OAuthClient
from .exceptions import AuthenticationError, InvalidStateError

__all__ = [
    "OAuthClient",
    "AuthenticationError",
    "InvalidStateError",
]
