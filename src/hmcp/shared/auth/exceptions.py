class AuthenticationError(Exception):
    """Base class for authentication errors"""

    pass


class InvalidTokenError(AuthenticationError):
    """Error raised when token validation fails"""

    pass


class ClientValidationError(AuthenticationError):
    """Error raised when client validation fails"""

    pass


class ScopeError(AuthenticationError):
    """Error raised when scope validation fails"""

    pass


class InvalidCodeError(AuthenticationError):
    """Error raised when authorization code validation fails"""

    pass


class InvalidPKCEError(AuthenticationError):
    """Error raised when PKCE validation fails"""

    pass


class InvalidRedirectUriError(AuthenticationError):
    """Error raised when redirect URI validation fails"""

    pass


class InvalidStateError(AuthenticationError):
    """Error raised when state parameter validation fails"""

    pass


class TokenExpiredError(AuthenticationError):
    """Error raised when token has expired"""

    pass


class RefreshTokenError(AuthenticationError):
    """Error raised when refresh token operation fails"""

    pass
