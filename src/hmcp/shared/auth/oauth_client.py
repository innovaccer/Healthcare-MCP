from typing import Optional, Dict, List, Tuple
import secrets
import hashlib
import base64
from .exceptions import AuthenticationError
import logging
import aiohttp
from urllib.parse import urlparse, parse_qs
import urllib.parse

logger = logging.getLogger(__name__)


class OAuthClient:
    """OAuth 2.0 client SDK for interacting with the HMCP OAuth server."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        scopes: List[str],
        grant_type: str = "client_credentials",
        server_url: str = "http://localhost:8000",
    ):
        """Initialize the OAuth client.

        Args:
            client_id: The client identifier
            client_secret: The client secret
            server_url: The OAuth server URL (default: http://localhost:8000)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.server_url = server_url.rstrip("/")
        self._session = None
        self.scopes: List[str] = scopes
        self.grant_type: str = grant_type
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.id_token: Optional[str] = None  # For OpenID Connect (future)
        self.token_response: Optional[Dict] = None  # Store full token response
        self.patient_id: Optional[str] = None  # For patient-context operations
        self.code_verifier: Optional[str] = None  # For PKCE

    async def __aenter__(self):
        """Create aiohttp session when entering async context."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting async context."""
        if self._session:
            await self._session.close()
            self._session = None

    async def validate_client(self) -> bool:
        """Validate client credentials by attempting to get a token."""
        try:
            response = await self.get_client_credentials_token()
            return "error" not in response
        except Exception:
            return False

    def generate_pkce_challenge(self) -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        self.code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(self.code_verifier.encode()).digest()
            )
            .decode()
            .rstrip("=")
        )
        return self.code_verifier, code_challenge

    def create_token_request(
        self, scopes: List[str] = None, patient_id: str = None
    ) -> dict:
        """Create token request payload for Client Credentials flow

        Args:
            scopes: List of requested scopes (defaults to instance scopes)
            patient_id: Optional patient identifier for patient-context operations
        """
        requested_scopes = scopes or self.scopes

        # Add patient context scope if patient_id is provided
        if patient_id and not any(s.startswith("patient/") for s in requested_scopes):
            # Convert to list if it's a tuple
            requested_scopes = list(requested_scopes)
            requested_scopes.append("launch/patient")

        request = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": self.grant_type,
            "scope": " ".join(requested_scopes),
        }

        return request

    def create_authorization_request(
        self,
        redirect_uri: str,
        scopes: List[str] = None,
        state: str = None,
        use_pkce: bool = True,
    ) -> Dict[str, str]:
        """Create authorization request parameters for Authorization Code flow"""
        requested_scopes = scopes or self.scopes
        state = state or secrets.token_urlsafe(16)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(requested_scopes),
            "state": state,
        }

        if use_pkce:
            code_verifier, code_challenge = self.generate_pkce_challenge()
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        return params

    def create_code_exchange_request(
        self, code: str, redirect_uri: str, use_pkce: bool = True
    ) -> Dict[str, str]:
        """Create token request for exchanging authorization code"""
        request = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }

        if use_pkce and self.code_verifier:
            request["code_verifier"] = self.code_verifier

        return request

    def create_refresh_token_request(self) -> Dict[str, str]:
        """Create token request for refreshing access token"""
        if not self.refresh_token:
            raise AuthenticationError("No refresh token available")

        return {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
        }

    def set_token(self, token_response: dict):
        """Process and store token response

        Stores the full token response, including access_token, id_token (if OpenID Connect),
        and patient context (if available).
        """
        self.token_response = token_response
        self.access_token = token_response["access_token"]

        # Store refresh token if present
        if "refresh_token" in token_response:
            self.refresh_token = token_response["refresh_token"]

        # Store ID token if present (OpenID Connect)
        if "id_token" in token_response:
            self.id_token = token_response["id_token"]

        # Store patient ID if present
        if "patient" in token_response:
            self.patient_id = token_response["patient"]

    def get_auth_header(self) -> dict:
        """Get authorization header for requests"""
        if not self.access_token:
            raise AuthenticationError("Not authenticated")
        return {"Authorization": f"Bearer {self.access_token}"}

    async def get_client_credentials_token(self, scopes: List[str] = None) -> Dict:
        """Get access token using Client Credentials flow.

        Args:
            scope: Space-separated list of requested scopes

        Returns:
            Token response containing access_token and other details
        """
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")

        url = f"{self.server_url}/oauth/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        scopes = scopes or self.scopes
        data = {
            "grant_type": self.grant_type,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": " ".join(scopes),
        }

        async with self._session.post(url, headers=headers, data=data) as response:
            return await response.json()

    async def set_client_credentials_token(self):
        """Set the access token for the client credentials flow"""
        token_response = await self.get_client_credentials_token()
        self.set_token(token_response)

    async def start_authorization_code_flow(
        self, redirect_uri: str, scope: str = "hmcp:access", state: Optional[str] = None
    ) -> Tuple[str, str]:
        """Start Authorization Code flow with PKCE.

        Args:
            redirect_uri: The redirect URI registered with the client
            scope: Space-separated list of requested scopes
            state: Optional state parameter for CSRF protection

        Returns:
            Tuple of (authorization_url, code_verifier)
        """
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")

        # Generate PKCE challenge
        code_verifier, code_challenge = self.generate_pkce_challenge()

        # Generate state if not provided
        if not state:
            state = secrets.token_urlsafe(16)

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        auth_url = f"{self.server_url}/oauth/authorize?{urllib.parse.urlencode(params)}"
        return auth_url, code_verifier

    async def exchange_code_for_token(
        self, code: str, redirect_uri: str, code_verifier: str
    ) -> Dict:
        """Exchange authorization code for token.

        Args:
            code: The authorization code received from the server
            redirect_uri: The redirect URI used in the authorization request
            code_verifier: The PKCE code verifier

        Returns:
            Token response containing access_token, refresh_token, and other details
        """
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")

        url = f"{self.server_url}/oauth/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }

        async with self._session.post(url, headers=headers, data=data) as response:
            return await response.json()

    async def refresh_access_token(self, refresh_token: str) -> Dict:
        """Refresh access token using refresh token.

        Args:
            refresh_token: The refresh token received from the server

        Returns:
            Token response containing new access_token and other details
        """
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")

        url = f"{self.server_url}/oauth/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
        }

        async with self._session.post(url, headers=headers, data=data) as response:
            return await response.json()

    async def revoke_token(
        self, token: str, token_type_hint: Optional[str] = None
    ) -> Dict:
        """Revoke an access or refresh token.

        Args:
            token: The token to revoke
            token_type_hint: Optional hint about the token type (access_token or refresh_token)

        Returns:
            Response from the revocation endpoint
        """
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")

        url = f"{self.server_url}/oauth/revoke"
        headers = {"Content-Type": "application/json"}
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "token": token,
        }
        if token_type_hint:
            data["token_type_hint"] = token_type_hint

        async with self._session.post(url, headers=headers, json=data) as response:
            return await response.json()

    async def introspect_token(self, token: str) -> Dict:
        """Get information about a token.

        Args:
            token: The token to introspect

        Returns:
            Token information including validity, scopes, and other claims
        """
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")

        url = f"{self.server_url}/oauth/introspect"
        headers = {"Content-Type": "application/json"}
        data = {
            "token": token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with self._session.post(url, headers=headers, json=data) as response:
            return await response.json()

    @staticmethod
    def parse_authorization_response(redirect_url: str) -> Tuple[str, str]:
        """Parse authorization code and state from redirect URL.

        Args:
            redirect_url: The redirect URL received from the server

        Returns:
            Tuple of (code, state)
        """
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        code = query_params.get("code", [None])[0]
        state = query_params.get("state", [None])[0]
        return code, state
