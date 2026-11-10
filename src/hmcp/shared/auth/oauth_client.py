"""OAuth 2.1 client for HMCP servers.

Rewritten for mcp 1.27+ standards:

  - Uses **httpx2** (consistent with mcp 2.x; drops aiohttp).
  - Discovers endpoints via RFC 8414
    `/.well-known/oauth-authorization-server` instead of hard-coding
    `/oauth/token`, `/oauth/register`, etc.
  - Returns **typed `OAuthToken`** from `mcp.shared.auth` instead of raw
    dicts.
  - First-class **PKCE S256** code-verifier/code-challenge handling for
    the authorization-code flow.
  - **RFC 7591 dynamic client registration** (`/register`) replaces the
    legacy ADMIN-API-KEY-gated `/oauth/register` flow.

The previous version targeted the gateway's hand-rolled `/oauth/*`
endpoints. Those are gone — the gateway now mounts mcp's standard routes
(no prefix), so the discovery document points clients straight at
`/authorize`, `/token`, `/register`, `/revoke`.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx2
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthMetadata,
    OAuthToken,
)

from .exceptions import AuthenticationError, InvalidStateError

logger = logging.getLogger(__name__)


_METADATA_PATH = "/.well-known/oauth-authorization-server"

# RFC 6749 error codes are short lowercase tokens; anything else is untrusted.
_ERROR_CODE_RE = re.compile(r"^[a-z_]{1,64}$")


def _valid_oauth_error_code(value: Any) -> Optional[str]:
    """Return `value` when it is a valid RFC 6749 error-code token, else None.

    RFC 6749 error codes are short lowercase tokens; anything else is
    untrusted and is never surfaced in a message.
    """
    if isinstance(value, str) and _ERROR_CODE_RE.fullmatch(value):
        return value
    return None


def _oauth_error_suffix(response: httpx2.Response) -> str:
    """Return ``" (error_code)"`` for a JSON body carrying a valid RFC 6749
    ``error`` string, else an empty string.

    Never includes any other part of the body: an authorization server may
    echo the submitted request (including ``client_secret``), so only the
    validated ``error`` code is ever surfaced in a message.
    """
    try:
        payload = response.json()
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    code = _valid_oauth_error_code(payload.get("error"))
    if code is not None:
        return f" ({code})"
    return ""


class OAuthClient:
    """OAuth 2.1 client for HMCP servers.

    Lifecycle:
      ``async with OAuthClient(...) as oc:``
        - opens an httpx2 client
        - on first use, discovers endpoints from
          `/.well-known/oauth-authorization-server`
        - all subsequent calls hit the discovered URLs

    Three flows supported:
      `get_client_credentials_token()`
          Machine-to-machine. POSTs to the discovered token endpoint with
          `grant_type=client_credentials`. Gateway-side support is an
          extension on top of mcp's stock TokenHandler; spec-compliant.
      `start_authorization_code_flow(redirect_uri, scope)` ➜ tuple
          Builds the `/authorize` URL with PKCE S256 and returns
          `(url, state, code_verifier)`. The caller redirects the user
          (or simulates the consent in a demo) and captures the code.
      `exchange_code_for_token(code, redirect_uri)`
          POSTs to `/token` with the captured code + the stored
          code_verifier. Returns a typed `OAuthToken`.

    Dynamic registration is via `register_client(...)`, which posts an
    `OAuthClientMetadata` to `/register` per RFC 7591.

    The client does not track token expiry (`OAuthToken` carries only the
    relative `expires_in`) and never refreshes automatically.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        server_url: str = "http://localhost:8080",
        *,
        timeout_s: float = 30.0,
        metadata_path: str = _METADATA_PATH,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes: List[str] = list(scopes or [])
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.metadata_path = metadata_path
        self._transport = transport

        # Populated by `_ensure_metadata()` on first call.
        self._metadata: Optional[OAuthMetadata] = None
        # Most-recently-issued token, set by token-acquisition methods.
        self.token: Optional[OAuthToken] = None
        # PKCE state — set during `start_authorization_code_flow`,
        # consumed by `exchange_code_for_token`.
        self.code_verifier: Optional[str] = None

        self._http: Optional[httpx2.AsyncClient] = None

    # --- async context-manager lifecycle ---------------------------------

    async def __aenter__(self) -> "OAuthClient":
        self._http = httpx2.AsyncClient(
            timeout=self.timeout_s, transport=self._transport
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # --- discovery -------------------------------------------------------

    async def discover(self) -> OAuthMetadata:
        """Fetch and cache the RFC 8414 authorization-server metadata.

        Idempotent. First call hits the network; subsequent calls return
        the cached doc.

        Raises `RuntimeError` outside the async context manager and
        `AuthenticationError` if the metadata request fails.
        """
        if self._metadata is not None:
            return self._metadata
        return await self._ensure_metadata()

    async def _ensure_metadata(self) -> OAuthMetadata:
        if self._metadata is not None:
            return self._metadata
        self._require_session()
        assert self._http is not None
        url = f"{self.server_url}{self.metadata_path}"
        auth_error: Optional[AuthenticationError] = None
        try:
            r = await self._http.get(url)
            r.raise_for_status()
        except Exception as e:
            auth_error = AuthenticationError(
                f"OAuth metadata fetch failed for {url}: {e}"
            )
        if auth_error is not None:
            raise auth_error
        document_error: Optional[AuthenticationError] = None
        try:
            self._metadata = OAuthMetadata.model_validate(r.json())
        except ValueError:
            document_error = AuthenticationError(
                "OAuth metadata response was not a valid metadata document"
            )
        if document_error is not None:
            raise document_error
        logger.debug("OAuth metadata: %s", self._metadata.model_dump())
        return self._metadata

    # --- PKCE helpers ----------------------------------------------------

    def generate_pkce_challenge(self) -> Tuple[str, str]:
        """Generate a fresh PKCE verifier + S256 challenge.

        Side effect: stashes the verifier on `self.code_verifier` so
        `exchange_code_for_token` can find it without the caller having
        to thread it through.

        Raises nothing.
        """
        verifier = secrets.token_urlsafe(32)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        self.code_verifier = verifier
        return verifier, challenge

    # --- client registration (RFC 7591) ---------------------------------

    async def register_client(
        self,
        *,
        redirect_uris: Optional[List[str]] = None,
        grant_types: Optional[List[str]] = None,
        scope: Optional[str] = None,
        client_name: Optional[str] = None,
        token_endpoint_auth_method: str = "client_secret_post",
    ) -> OAuthClientInformationFull:
        """Register this client with the authorization server.

        Returns the fully-populated `OAuthClientInformationFull` and
        stores the issued `client_id` / `client_secret` on `self` so
        subsequent token requests can use them.

        `token_endpoint_auth_method` must be `"client_secret_post"`
        (the default: the secret travels in the token-request body) or
        `"none"` (a public client that authenticates with PKCE only).
        Any other value raises `ValueError` before any request is made.
        HTTP Basic client authentication is not implemented.

        Raises `RuntimeError` outside the async context manager and
        `AuthenticationError` if the server advertises no
        `registration_endpoint` or the registration request fails.
        """
        if token_endpoint_auth_method not in ("client_secret_post", "none"):
            raise ValueError(
                "token_endpoint_auth_method must be one of "
                "'client_secret_post', 'none'"
            )
        self._require_session()
        meta = await self._ensure_metadata()
        endpoint = meta.registration_endpoint
        if endpoint is None:
            raise AuthenticationError(
                "Authorization server doesn't advertise a registration_endpoint"
            )

        body: Dict[str, Any] = {
            "redirect_uris": redirect_uris or ["http://localhost/callback"],
            "grant_types": grant_types or ["client_credentials", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": token_endpoint_auth_method,
        }
        if scope is not None:
            body["scope"] = scope
        elif self.scopes:
            body["scope"] = " ".join(self.scopes)
        if client_name:
            body["client_name"] = client_name

        assert self._http is not None
        auth_error: Optional[AuthenticationError] = None
        try:
            r = await self._http.post(str(endpoint), json=body)
            r.raise_for_status()
        except httpx2.HTTPStatusError as e:
            auth_error = AuthenticationError(
                f"OAuth /register HTTP {e.response.status_code}"
                f"{_oauth_error_suffix(e.response)}"
            )
        if auth_error is not None:
            raise auth_error
        document_error: Optional[AuthenticationError] = None
        info: Optional[OAuthClientInformationFull] = None
        try:
            info = OAuthClientInformationFull.model_validate(r.json())
        except ValueError:
            document_error = AuthenticationError(
                "OAuth /register response was not a valid client "
                "registration document"
            )
        if document_error is not None:
            raise document_error
        assert info is not None
        self.client_id = info.client_id
        if info.client_secret:
            self.client_secret = info.client_secret
        return info

    # --- client_credentials grant ---------------------------------------

    async def get_client_credentials_token(
        self, scopes: Optional[List[str]] = None
    ) -> OAuthToken:
        """Machine-to-machine token. Standard grant_type=client_credentials.

        On mcp 1.27+ gateways, the m2m token endpoint sits at `/token/m2m`
        (a server extension because mcp's own `/token` only handles
        authorization_code + refresh_token grants). We use the metadata
        document's `token_endpoint` if it ends with `/token`, and try
        `/token/m2m` as a fallback. Caches the result on `self.token`.

        Raises `AuthenticationError` if `client_id`/`client_secret` are
        missing or the token request fails, and `RuntimeError` outside the
        async context manager.
        """
        if not self.client_id or not self.client_secret:
            raise AuthenticationError(
                "client_id and client_secret are required for client_credentials"
            )
        self._require_session()
        meta = await self._ensure_metadata()
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        scopes = scopes or self.scopes
        if scopes:
            data["scope"] = " ".join(scopes)

        # Build the m2m endpoint URL from the discovered token_endpoint.
        token_url = str(meta.token_endpoint)
        m2m_url = (
            token_url.rstrip("/") + "/m2m"
            if token_url.endswith("/token")
            else f"{self.server_url}/token/m2m"
        )
        token: Optional[OAuthToken] = None
        try:
            token = await self._token_request(m2m_url, data)
        except AuthenticationError:
            pass
        if token is None:
            # Fall back to the canonical endpoint (older gateways that
            # accept client_credentials at /token).
            token = await self._token_request(token_url, data)
        self.token = token
        return token

    async def set_client_credentials_token(
        self, scopes: Optional[List[str]] = None
    ) -> None:
        """Backward-compat wrapper. Mints + caches a token via
        client_credentials. Older callers (e.g. `HMCPClientConnector`)
        invoke this and then read `self.access_token` / `self.token`.

        Raises the same errors as `get_client_credentials_token`.
        """
        await self.get_client_credentials_token(scopes=scopes)

    # --- legacy attribute shims (read-only) -------------------------------

    @property
    def access_token(self) -> Optional[str]:
        """Mirror the v0 attribute. Older code reads `oc.access_token`
        directly; the new API stores tokens under `self.token` (typed).

        Raises nothing.
        """
        return self.token.access_token if self.token else None

    @property
    def refresh_token(self) -> Optional[str]:
        """Most-recently-issued refresh token, or None if there is none.

        Raises nothing.
        """
        return self.token.refresh_token if self.token else None

    # --- authorization_code grant ---------------------------------------

    async def start_authorization_code_flow(
        self,
        redirect_uri: str,
        scope: str = "hmcp:access",
        state: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Build the `/authorize` URL with PKCE.

        Returns `(authorization_url, state, code_verifier)`. The caller
        captures the redirected-back `code`, then calls
        `exchange_code_for_token`. Keep the returned `state` and pass it
        back as `expected_state` to `parse_authorization_response` so the
        redirect is verified against CSRF.

        Raises `AuthenticationError` if no `client_id` is set and
        `RuntimeError` outside the async context manager.
        """
        if not self.client_id:
            raise AuthenticationError(
                "client_id is required (register_client first or pass at init)"
            )
        meta = await self._ensure_metadata()
        verifier, challenge = self.generate_pkce_challenge()
        state = state or secrets.token_urlsafe(16)
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{meta.authorization_endpoint}?{urllib.parse.urlencode(params)}"
        return auth_url, state, verifier

    async def exchange_code_for_token(
        self, code: str, redirect_uri: str, code_verifier: Optional[str] = None
    ) -> OAuthToken:
        """Exchange a freshly-redirected `code` for an `OAuthToken`.

        If `code_verifier` is None, uses the one stashed by
        `start_authorization_code_flow`.

        Raises `RuntimeError` outside the async context manager and
        `AuthenticationError` if no code verifier is available or the
        token request fails.
        """
        self._require_session()
        verifier = code_verifier or self.code_verifier
        if verifier is None:
            raise AuthenticationError(
                "code_verifier required (run start_authorization_code_flow first "
                "or pass verifier explicitly)"
            )
        meta = await self._ensure_metadata()
        data: Dict[str, Any] = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        token = await self._token_request(str(meta.token_endpoint), data)
        self.token = token
        return token

    # --- refresh + revoke -----------------------------------------------

    async def refresh_access_token(
        self, refresh_token: Optional[str] = None
    ) -> OAuthToken:
        """Exchange a refresh token for a fresh access token.

        Uses the cached refresh token when `refresh_token` is not given.
        If the server does not rotate the refresh token, the one just
        used is kept.

        Raises `AuthenticationError` if no refresh token is available or
        the request fails, and `RuntimeError` outside the async context
        manager.
        """
        rt = refresh_token or (self.token.refresh_token if self.token else None)
        if rt is None:
            raise AuthenticationError("No refresh_token available")
        self._require_session()
        meta = await self._ensure_metadata()
        data: Dict[str, Any] = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": rt,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        if self.scopes:
            data["scope"] = " ".join(self.scopes)
        token = await self._token_request(str(meta.token_endpoint), data)
        if token.refresh_token is None:
            token = token.model_copy(update={"refresh_token": rt})
        self.token = token
        return token

    async def revoke_token(
        self, token: Optional[str] = None, token_type_hint: Optional[str] = None
    ) -> None:
        """Revoke a token per RFC 7009.

        Raises `AuthenticationError` if there is no token to revoke, the
        server advertises no `revocation_endpoint`, or the request fails,
        and `RuntimeError` outside the async context manager.
        """
        target = token or (self.token.access_token if self.token else None)
        if target is None:
            raise AuthenticationError("No token to revoke")
        self._require_session()
        meta = await self._ensure_metadata()
        endpoint = meta.revocation_endpoint
        if endpoint is None:
            raise AuthenticationError(
                "Authorization server doesn't advertise a revocation_endpoint"
            )
        data: Dict[str, Any] = {
            "token": target,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        if token_type_hint:
            data["token_type_hint"] = token_type_hint
        assert self._http is not None
        r = await self._http.post(str(endpoint), data=data)
        if r.status_code >= 400:
            raise AuthenticationError(
                f"OAuth /revoke HTTP {r.status_code}{_oauth_error_suffix(r)}"
            )

    # --- header helpers --------------------------------------------------

    def get_auth_header(self) -> Dict[str, str]:
        """Authorization header for resource-server requests.

        Raises `AuthenticationError` if no access token has been acquired.
        """
        if not self.token or not self.token.access_token:
            raise AuthenticationError("Not authenticated — no access_token")
        return {"Authorization": f"Bearer {self.token.access_token}"}

    # --- response parsing ------------------------------------------------

    @staticmethod
    def parse_authorization_response(
        redirect_url: str,
        *,
        expected_state: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract `(code, state)` from an authorization redirect URL.

        Pass the `state` returned by `start_authorization_code_flow` as
        `expected_state` to have it checked (RFC 6749 section 10.12); a
        missing or mismatched `state` raises `InvalidStateError`.

        A redirect carrying an `error` parameter raises
        `AuthenticationError`. The code is named only when it is a valid
        RFC 6749 token; otherwise a fixed message is used. The
        `error_description` and `error_uri` values are never included.

        Without `expected_state` and without an error, the return value
        is unchanged.
        """
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        state = query_params.get("state", [None])[0]
        if expected_state is not None and (
            state is None
            or not secrets.compare_digest(
                state.encode("utf-8"), expected_state.encode("utf-8")
            )
        ):
            raise InvalidStateError("OAuth state mismatch")
        raw_error = query_params.get("error", [None])[0]
        if raw_error is not None:
            error = _valid_oauth_error_code(raw_error)
            if error is not None:
                raise AuthenticationError(f"OAuth authorization error: {error}")
            raise AuthenticationError("OAuth authorization error")
        code = query_params.get("code", [None])[0]
        return code, state

    # --- internals -------------------------------------------------------

    def _require_session(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "OAuthClient must be used as an async context manager: "
                "`async with OAuthClient(...) as oc: ...`"
            )

    async def _token_request(self, url: str, data: Dict[str, Any]) -> OAuthToken:
        assert self._http is not None
        auth_error: Optional[AuthenticationError] = None
        try:
            r = await self._http.post(
                url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            r.raise_for_status()
        except httpx2.HTTPStatusError as e:
            auth_error = AuthenticationError(
                f"OAuth /token HTTP {e.response.status_code}"
                f"{_oauth_error_suffix(e.response)}"
            )
        if auth_error is not None:
            raise auth_error
        document_error: Optional[AuthenticationError] = None
        token: Optional[OAuthToken] = None
        try:
            token = OAuthToken.model_validate(r.json())
        except ValueError:
            document_error = AuthenticationError(
                "OAuth /token response was not a valid token document"
            )
        if document_error is not None:
            raise document_error
        assert token is not None
        return token
