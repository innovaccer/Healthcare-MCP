"""Tests for the httpx2-based OAuthClient.

All HTTP is served by ``httpx2.MockTransport`` handlers that assert the
request path and form body they receive and return 400 for anything
unexpected (a strict fake). No network access.
"""

from __future__ import annotations

import base64
import hashlib
import json
import traceback
from typing import Any, Callable, Dict
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest

from hmcp.shared.auth import AuthenticationError, OAuthClient
from hmcp.shared.auth.exceptions import InvalidStateError

SERVER = "http://auth.test"
METADATA_PATH = "/.well-known/oauth-authorization-server"

Handler = Callable[[httpx2.Request], httpx2.Response]


def _metadata(**overrides: Any) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "issuer": SERVER,
        "authorization_endpoint": f"{SERVER}/authorize",
        "token_endpoint": f"{SERVER}/token",
        "registration_endpoint": f"{SERVER}/register",
        "revocation_endpoint": f"{SERVER}/revoke",
    }
    doc.update(overrides)
    return doc


def _form(request: httpx2.Request) -> Dict[str, list[str]]:
    return parse_qs(request.content.decode())


def _client(handler: Handler, **kwargs: Any) -> OAuthClient:
    return OAuthClient(
        server_url=SERVER, transport=httpx2.MockTransport(handler), **kwargs
    )


def _assert_sanitised_auth_error(
    excinfo: pytest.ExceptionInfo[AuthenticationError], expected_message: str
) -> None:
    err = excinfo.value
    assert str(err) == expected_message
    assert err.__cause__ is None
    assert err.__context__ is None
    assert "SENTINEL-SECRET" not in "".join(traceback.format_exception(err))


@pytest.mark.asyncio
async def test_discover_parses_metadata() -> None:
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        assert request.method == "GET"
        assert request.url.path == METADATA_PATH
        return httpx2.Response(200, json=_metadata())

    async with _client(handler) as oc:
        metadata = await oc.discover()
        # Second call must be served from cache, not the network.
        again = await oc.discover()

    assert calls["n"] == 1
    assert again is metadata
    assert str(metadata.token_endpoint) == f"{SERVER}/token"
    assert str(metadata.authorization_endpoint) == f"{SERVER}/authorize"


@pytest.mark.asyncio
async def test_client_credentials_token_is_cached_and_used_in_header() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token/m2m":
            assert request.method == "POST"
            body = _form(request)
            assert body["grant_type"] == ["client_credentials"]
            assert body["client_id"] == ["test-client"]
            assert body["client_secret"] == ["test-secret"]
            assert body["scope"] == ["read write"]
            return httpx2.Response(
                200, json={"access_token": "tok-test-1", "token_type": "Bearer"}
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        token = await oc.get_client_credentials_token(scopes=["read", "write"])
        assert token.access_token == "tok-test-1"
        assert oc.access_token == "tok-test-1"
        assert oc.get_auth_header() == {"Authorization": "Bearer tok-test-1"}


@pytest.mark.asyncio
async def test_start_authorization_code_flow_returns_pkce_tuple() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == METADATA_PATH
        return httpx2.Response(200, json=_metadata())

    async with _client(handler, client_id="test-client") as oc:
        auth_url, state, verifier = await oc.start_authorization_code_flow(
            "http://localhost/callback"
        )

    query = parse_qs(urlparse(auth_url).query)
    expected_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [expected_challenge]
    assert query["state"] == [state]
    assert query["client_id"] == ["test-client"]
    assert query["response_type"] == ["code"]
    assert oc.code_verifier == verifier


@pytest.mark.asyncio
async def test_exchange_code_sends_stored_verifier() -> None:
    seen: Dict[str, str] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            body = _form(request)
            assert body["grant_type"] == ["authorization_code"]
            assert body["code"] == ["code-test"]
            assert body["code_verifier"] == [seen["verifier"]]
            return httpx2.Response(
                200, json={"access_token": "tok-test-2", "token_type": "Bearer"}
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        _, _, verifier = await oc.start_authorization_code_flow(
            "http://localhost/callback"
        )
        seen["verifier"] = verifier
        token = await oc.exchange_code_for_token(
            "code-test", "http://localhost/callback"
        )

    assert token.access_token == "tok-test-2"


@pytest.mark.asyncio
async def test_refresh_access_token_without_refresh_token_raises() -> None:
    oc = OAuthClient(client_id="test-client", server_url=SERVER)
    with pytest.raises(AuthenticationError):
        await oc.refresh_access_token()


@pytest.mark.asyncio
async def test_token_401_raises_authentication_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            return httpx2.Response(401, json={"error": "invalid_client"})
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        _, _, verifier = await oc.start_authorization_code_flow(
            "http://localhost/callback"
        )
        with pytest.raises(AuthenticationError):
            await oc.exchange_code_for_token(
                "code-test", "http://localhost/callback", code_verifier=verifier
            )


@pytest.mark.asyncio
async def test_revoke_token_posts_to_revocation_endpoint() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/revoke":
            assert request.method == "POST"
            body = _form(request)
            assert body["token"] == ["tok-test-3"]
            assert body["client_id"] == ["test-client"]
            assert body["client_secret"] == ["test-secret"]
            return httpx2.Response(200)
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        await oc.revoke_token("tok-test-3")


@pytest.mark.asyncio
async def test_network_method_outside_context_manager_raises() -> None:
    oc = OAuthClient(
        client_id="test-client", client_secret="test-secret", server_url=SERVER
    )
    with pytest.raises(RuntimeError):
        await oc.discover()


# --- error messages must not leak the response body -----------------------


@pytest.mark.asyncio
async def test_token_401_message_has_error_code_but_not_body() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            return httpx2.Response(
                401,
                json={"error": "invalid_client", "echo": "SENTINEL-SECRET"},
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        _, _, verifier = await oc.start_authorization_code_flow(
            "http://localhost/callback"
        )
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.exchange_code_for_token(
                "code-test", "http://localhost/callback", code_verifier=verifier
            )

    message = str(excinfo.value)
    assert "401" in message
    assert "invalid_client" in message
    assert "SENTINEL-SECRET" not in message


@pytest.mark.asyncio
async def test_token_non_json_error_body_not_in_message() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            return httpx2.Response(500, text="SENTINEL-SECRET <html>")
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        _, _, verifier = await oc.start_authorization_code_flow(
            "http://localhost/callback"
        )
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.exchange_code_for_token(
                "code-test", "http://localhost/callback", code_verifier=verifier
            )

    message = str(excinfo.value)
    assert "500" in message
    assert "SENTINEL-SECRET" not in message


@pytest.mark.asyncio
async def test_token_invalid_error_code_is_omitted() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            return httpx2.Response(400, json={"error": "<script>"})
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        _, _, verifier = await oc.start_authorization_code_flow(
            "http://localhost/callback"
        )
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.exchange_code_for_token(
                "code-test", "http://localhost/callback", code_verifier=verifier
            )

    message = str(excinfo.value)
    assert message == "OAuth /token HTTP 400"
    assert "<script>" not in message


@pytest.mark.asyncio
async def test_register_error_message_has_code_but_not_body() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/register":
            return httpx2.Response(
                400,
                json={"error": "invalid_client_metadata", "echo": "SENTINEL-SECRET"},
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.register_client()

    message = str(excinfo.value)
    assert "400" in message
    assert "invalid_client_metadata" in message
    assert "SENTINEL-SECRET" not in message


@pytest.mark.asyncio
async def test_revoke_error_message_has_code_but_not_body() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/revoke":
            return httpx2.Response(
                400, json={"error": "invalid_token", "echo": "SENTINEL-SECRET"}
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.revoke_token("tok-test-4")

    message = str(excinfo.value)
    assert "400" in message
    assert "invalid_token" in message
    assert "SENTINEL-SECRET" not in message


# --- errors must not chain the raw HTTP exception -------------------------


@pytest.mark.asyncio
async def test_token_error_drops_cause_and_secret() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            return httpx2.Response(
                401,
                json={"error": "invalid_client", "echo": "SENTINEL-SECRET"},
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="SENTINEL-SECRET"
    ) as oc:
        _, _, verifier = await oc.start_authorization_code_flow(
            "http://localhost/callback"
        )
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.exchange_code_for_token(
                "code-test", "http://localhost/callback", code_verifier=verifier
            )

    err = excinfo.value
    assert str(err) == "OAuth /token HTTP 401 (invalid_client)"
    assert err.__cause__ is None
    assert err.__context__ is None
    assert "SENTINEL-SECRET" not in "".join(traceback.format_exception(err))


@pytest.mark.asyncio
async def test_register_error_drops_cause_and_secret() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/register":
            return httpx2.Response(
                400,
                json={
                    "error": "invalid_client_metadata",
                    "echo": "SENTINEL-SECRET",
                },
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="SENTINEL-SECRET"
    ) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.register_client()

    err = excinfo.value
    assert str(err) == "OAuth /register HTTP 400 (invalid_client_metadata)"
    assert err.__cause__ is None
    assert err.__context__ is None
    assert "SENTINEL-SECRET" not in "".join(traceback.format_exception(err))


@pytest.mark.asyncio
async def test_metadata_error_drops_cause_and_secret() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(
                401,
                json={"error": "invalid_client", "echo": "SENTINEL-SECRET"},
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="SENTINEL-SECRET"
    ) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.discover()

    err = excinfo.value
    assert str(err).startswith(
        f"OAuth metadata fetch failed for {SERVER}{METADATA_PATH}"
    )
    assert err.__cause__ is None
    assert err.__context__ is None
    assert "SENTINEL-SECRET" not in "".join(traceback.format_exception(err))


# --- refresh-token retention -----------------------------------------------


@pytest.mark.asyncio
async def test_refresh_rotation_replaces_refresh_token() -> None:
    seen: list[Dict[str, list[str]]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            body = _form(request)
            seen.append(body)
            if body["grant_type"] == ["authorization_code"]:
                return httpx2.Response(
                    200,
                    json={
                        "access_token": "at-1",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": "rt-1",
                    },
                )
            return httpx2.Response(
                200,
                json={
                    "access_token": "at-2",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "rt-2",
                },
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        await oc.start_authorization_code_flow("http://localhost/callback")
        await oc.exchange_code_for_token("code-test", "http://localhost/callback")
        assert oc.refresh_token == "rt-1"
        token = await oc.refresh_access_token()

    assert token.access_token == "at-2"
    assert token.refresh_token == "rt-2"
    assert oc.refresh_token == "rt-2"
    assert seen[1]["grant_type"] == ["refresh_token"]
    assert seen[1]["refresh_token"] == ["rt-1"]


@pytest.mark.asyncio
async def test_refresh_without_rotation_keeps_old_refresh_token() -> None:
    seen: list[Dict[str, list[str]]] = []
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            body = _form(request)
            seen.append(body)
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx2.Response(
                    200,
                    json={
                        "access_token": "at-1",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": "rt-1",
                    },
                )
            # RFC 6749 section 6: the AS may omit refresh_token on refresh.
            return httpx2.Response(
                200,
                json={
                    "access_token": f"at-{calls['n']}",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        await oc.start_authorization_code_flow("http://localhost/callback")
        await oc.exchange_code_for_token("code-test", "http://localhost/callback")
        await oc.refresh_access_token()
        assert oc.refresh_token == "rt-1"
        token = await oc.refresh_access_token()

    assert token.refresh_token == "rt-1"
    assert oc.refresh_token == "rt-1"
    assert seen[1]["refresh_token"] == ["rt-1"]
    assert seen[2]["refresh_token"] == ["rt-1"]


@pytest.mark.asyncio
async def test_exchange_outside_context_manager_raises_runtime_error() -> None:
    oc = OAuthClient(
        client_id="test-client", client_secret="test-secret", server_url=SERVER
    )
    with pytest.raises(RuntimeError):
        await oc.exchange_code_for_token("code-test", "http://localhost/callback")


# --- client registration ---------------------------------------------------


@pytest.mark.asyncio
async def test_register_client_stores_credentials() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/register":
            return httpx2.Response(
                200,
                json={
                    "client_id": "reg-1",
                    "client_secret": "reg-secret",
                    "redirect_uris": ["http://localhost/callback"],
                },
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler) as oc:
        info = await oc.register_client()

    assert info.client_id == "reg-1"
    assert oc.client_id == "reg-1"
    assert oc.client_secret == "reg-secret"


@pytest.mark.asyncio
async def test_register_defaults_to_client_secret_post() -> None:
    seen: Dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/register":
            seen["body"] = json.loads(request.content.decode())
            return httpx2.Response(
                200, json={"client_id": "reg-1", "client_secret": "reg-secret"}
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler) as oc:
        await oc.register_client()

    assert seen["body"]["token_endpoint_auth_method"] == "client_secret_post"


@pytest.mark.asyncio
async def test_register_rejects_unsupported_auth_method_without_request() -> None:
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, json=_metadata())

    async with _client(handler) as oc:
        with pytest.raises(ValueError) as excinfo:
            await oc.register_client(token_endpoint_auth_method="client_secret_basic")

    message = str(excinfo.value)
    assert "client_secret_post" in message
    assert "none" in message
    assert calls == []


@pytest.mark.asyncio
async def test_public_client_sends_no_client_secret_in_token_request() -> None:
    seen: Dict[str, list[str]] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            seen["body"] = _form(request)
            return httpx2.Response(
                200, json={"access_token": "at-1", "token_type": "Bearer"}
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler, client_id="public-client") as oc:
        await oc.start_authorization_code_flow("http://localhost/callback")
        await oc.exchange_code_for_token("code-test", "http://localhost/callback")

    assert "client_secret" not in seen["body"]


# --- authorization redirect parsing ----------------------------------------


def test_parse_authorization_response_returns_code_and_state() -> None:
    code, state = OAuthClient.parse_authorization_response(
        "http://localhost/cb?code=abc&state=xyz"
    )
    assert code == "abc"
    assert state == "xyz"


def test_parse_authorization_response_accepts_matching_state() -> None:
    code, state = OAuthClient.parse_authorization_response(
        "http://localhost/cb?code=abc&state=xyz", expected_state="xyz"
    )
    assert code == "abc"
    assert state == "xyz"


def test_parse_authorization_response_rejects_mismatched_state() -> None:
    with pytest.raises(InvalidStateError):
        OAuthClient.parse_authorization_response(
            "http://localhost/cb?code=abc&state=xyz", expected_state="other"
        )


def test_parse_authorization_response_rejects_missing_state() -> None:
    with pytest.raises(InvalidStateError):
        OAuthClient.parse_authorization_response(
            "http://localhost/cb?code=abc", expected_state="xyz"
        )


def test_parse_authorization_response_error_message_has_code_only() -> None:
    with pytest.raises(AuthenticationError) as excinfo:
        OAuthClient.parse_authorization_response(
            "http://localhost/cb?error=access_denied"
            "&error_description=SENTINEL-DESCRIPTION"
            "&error_uri=http://example.test/SENTINEL-URI"
        )

    message = str(excinfo.value)
    assert "access_denied" in message
    assert "SENTINEL-DESCRIPTION" not in message
    assert "SENTINEL-URI" not in message


def test_parse_authorization_response_invalid_error_value_raises() -> None:
    with pytest.raises(AuthenticationError) as excinfo:
        OAuthClient.parse_authorization_response(
            "http://localhost/cb?error=Access%20Denied%21"
        )

    message = str(excinfo.value)
    assert "Access" not in message
    assert "Denied" not in message
    assert "Access%20Denied" not in message


def test_parse_authorization_response_non_ascii_state_mismatch() -> None:
    with pytest.raises(InvalidStateError):
        OAuthClient.parse_authorization_response(
            "http://localhost/cb?code=abc&state=caf%C3%A9", expected_state="xyz"
        )


def test_parse_authorization_response_non_ascii_state_match() -> None:
    code, state = OAuthClient.parse_authorization_response(
        "http://localhost/cb?code=abc&state=caf%C3%A9", expected_state="café"
    )
    assert code == "abc"
    assert state == "café"


# --- malformed 2xx documents must not leak the body -----------------------


@pytest.mark.asyncio
async def test_metadata_non_json_2xx_body_raises_sanitised_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text="SENTINEL-SECRET not json")

    async with _client(handler) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.discover()

    _assert_sanitised_auth_error(
        excinfo, "OAuth metadata response was not a valid metadata document"
    )


@pytest.mark.asyncio
async def test_metadata_schema_invalid_2xx_body_raises_sanitised_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"client_secret": "SENTINEL-SECRET"})

    async with _client(handler) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.discover()

    _assert_sanitised_auth_error(
        excinfo, "OAuth metadata response was not a valid metadata document"
    )


@pytest.mark.asyncio
async def test_register_non_json_2xx_body_raises_sanitised_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/register":
            return httpx2.Response(200, text="SENTINEL-SECRET not json")
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.register_client()

    _assert_sanitised_auth_error(
        excinfo,
        "OAuth /register response was not a valid client registration document",
    )


@pytest.mark.asyncio
async def test_register_schema_invalid_2xx_body_raises_sanitised_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/register":
            return httpx2.Response(200, json={"client_secret": "SENTINEL-SECRET"})
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.register_client()

    _assert_sanitised_auth_error(
        excinfo,
        "OAuth /register response was not a valid client registration document",
    )


@pytest.mark.asyncio
async def test_token_non_json_2xx_body_raises_sanitised_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            return httpx2.Response(200, text="SENTINEL-SECRET not json")
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler, client_id="c", client_secret="SENTINEL-SECRET") as oc:
        await oc.start_authorization_code_flow("http://localhost/callback")
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.exchange_code_for_token("code-test", "http://localhost/callback")

    _assert_sanitised_auth_error(
        excinfo, "OAuth /token response was not a valid token document"
    )


@pytest.mark.asyncio
async def test_token_schema_invalid_2xx_body_raises_sanitised_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token":
            return httpx2.Response(
                200,
                json={"error": "invalid_grant", "client_secret": "SENTINEL-SECRET"},
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(handler, client_id="c", client_secret="SENTINEL-SECRET") as oc:
        await oc.start_authorization_code_flow("http://localhost/callback")
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.exchange_code_for_token("code-test", "http://localhost/callback")

    _assert_sanitised_auth_error(
        excinfo, "OAuth /token response was not a valid token document"
    )


# --- client_credentials endpoint fallback ---------------------------------


@pytest.mark.asyncio
async def test_client_credentials_fallback_error_has_no_context() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token/m2m":
            return httpx2.Response(404, text="not found")
        if request.url.path == "/token":
            return httpx2.Response(
                401,
                json={"error": "invalid_client", "echo": "SENTINEL-SECRET"},
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="SENTINEL-SECRET"
    ) as oc:
        with pytest.raises(AuthenticationError) as excinfo:
            await oc.get_client_credentials_token()

    err = excinfo.value
    assert str(err) == "OAuth /token HTTP 401 (invalid_client)"
    assert err.__cause__ is None
    assert err.__context__ is None
    assert "SENTINEL-SECRET" not in "".join(traceback.format_exception(err))


@pytest.mark.asyncio
async def test_client_credentials_falls_back_to_token_endpoint() -> None:
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == METADATA_PATH:
            return httpx2.Response(200, json=_metadata())
        if request.url.path == "/token/m2m":
            seen.append("/token/m2m")
            return httpx2.Response(404, text="not found")
        if request.url.path == "/token":
            seen.append("/token")
            return httpx2.Response(
                200, json={"access_token": "at-fallback", "token_type": "Bearer"}
            )
        return httpx2.Response(400, text="unexpected request")

    async with _client(
        handler, client_id="test-client", client_secret="test-secret"
    ) as oc:
        token = await oc.get_client_credentials_token()

    assert token.access_token == "at-fallback"
    assert seen == ["/token/m2m", "/token"]
