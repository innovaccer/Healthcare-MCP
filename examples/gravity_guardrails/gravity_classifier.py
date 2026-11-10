"""EXAMPLE code for HMCP's ``classifier`` guardrail: a Gravity guardrail adapter.

This module is example code. It is NOT part of the ``hmcp`` package: copy it into
your own project and adapt it. It implements HMCP's ``ClassifierProvider``
protocol on top of Gravity guardrail configs, so an HMCP ``classifier`` guardrail
can call Gravity's
``{api_base}/guardrails/{guardrail_id}/configs/{config_id}/invoke`` endpoint.

It never logs, and never raises, the inspected text, a response body or a token:
every error carries a fixed string only.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import httpx2

from hmcp.shared.guardrails import ClassificationRequest, ClassificationResult

# Gravity's per-config check type. HMCP stages and Gravity check types are not
# the same axis, so the mapping is chosen explicitly per guardrail (below).
GravityCheckType = Literal["input", "output"]

_CHECK_TYPES = ("input", "output")

# Guardrail and config ids go into the URL path, so restrict them to a safe set.
# fullmatch (not match) is used below so a trailing newline is rejected.
_PATH_ID = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class GravityGuardrailRef:
    """A Gravity guardrail config plus the checkType to use for each HMCP stage."""

    guardrail_id: str
    config_id: str
    # Gravity checkType per HMCP stage. REQUIRED, with no default: a Gravity
    # config's output side may be set to "allow" (the model is not run), while an
    # HMCP "output" stage inspects a TOOL RESULT, which is input to the host's
    # model. Guessing the mapping would silently disable the guardrail, so the
    # caller must choose it explicitly.
    check_types: Mapping[str, GravityCheckType]


class GravityClassifierError(Exception):
    """A Gravity guardrail call could not produce a verdict.

    Messages are FIXED strings only: never the inspected text, a response body
    or a token.
    """


class GravityClassifier:
    """HMCP ``ClassifierProvider`` backed by Gravity guardrail configs."""

    def __init__(
        self,
        *,
        api_base: str,
        token_provider: Callable[[], Awaitable[str]],
        guardrails: Mapping[str, GravityGuardrailRef],
        tenant_id: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        api_base = api_base.rstrip("/")
        if not api_base.startswith(("http://", "https://")):
            raise ValueError("api_base must be an http(s) URL")
        # Parse the URL so a malformed host/port is rejected here, not by a raw
        # httpx2.InvalidURL (which quotes part of the URL) at call time.
        url_error = False
        parsed_base: httpx2.URL | None = None
        try:
            parsed_base = httpx2.URL(api_base)
        except Exception:  # noqa: BLE001 - fixed message, never echo the value
            url_error = True
        if url_error or not parsed_base.host:
            # Raised OUTSIDE the except block so __context__ stays None.
            raise ValueError("api_base must be an http(s) URL") from None

        # Tenant ids are opaque, so only reject values that would break a header.
        if tenant_id is not None and ("\r" in tenant_id or "\n" in tenant_id):
            raise ValueError("tenant_id must not contain a newline")

        for ref in guardrails.values():
            if not _PATH_ID.fullmatch(ref.guardrail_id):
                raise ValueError("guardrail_id must match ^[A-Za-z0-9_-]+$")
            if not _PATH_ID.fullmatch(ref.config_id):
                raise ValueError("config_id must match ^[A-Za-z0-9_-]+$")
            if not ref.check_types:
                raise ValueError("check_types must not be empty")
            for stage, check_type in ref.check_types.items():
                if stage not in _CHECK_TYPES:
                    raise ValueError("check_types key must be 'input' or 'output'")
                if check_type not in _CHECK_TYPES:
                    raise ValueError("check_types value must be 'input' or 'output'")

        self._api_base = api_base
        self._token_provider = token_provider
        self._guardrails = dict(guardrails)
        self._tenant_id = tenant_id
        self._owns_client = http_client is None
        self._client = http_client or httpx2.AsyncClient(timeout=timeout_s)

    async def classify(self, request: ClassificationRequest) -> ClassificationResult:
        ref = self._guardrails.get(request.classifier)
        if ref is None:
            raise GravityClassifierError("unknown classifier")

        check_type = ref.check_types.get(request.stage)
        if check_type is None:
            raise GravityClassifierError("no Gravity check type for this stage")

        token_failed = False
        token = ""
        try:
            token = await self._token_provider()
        except Exception:  # noqa: BLE001 - fail closed with a fixed message
            # Raised OUTSIDE the except block so __context__ stays None: a
            # provider error could otherwise carry a token or a URL.
            token_failed = True
        if token_failed:
            raise GravityClassifierError("Gravity token unavailable") from None

        url = (
            f"{self._api_base}/guardrails/{ref.guardrail_id}"
            f"/configs/{ref.config_id}/invoke"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if self._tenant_id is not None:
            headers["X-Tenant-ID"] = self._tenant_id

        call_failed = False
        response: httpx2.Response | None = None
        try:
            response = await self._client.post(
                url,
                json={"text": request.text, "checkType": check_type},
                headers=headers,
            )
        except (httpx2.HTTPError, httpx2.InvalidURL):
            # Raised OUTSIDE the except block below so __context__ stays None: a
            # transport error message could carry the URL or the token.
            call_failed = True
        if call_failed or response is None:
            raise GravityClassifierError("Gravity guardrail call failed") from None

        if not 200 <= response.status_code < 300:
            # The status code only, never the body.
            raise GravityClassifierError(
                f"Gravity guardrail call returned HTTP {response.status_code}"
            )

        invalid_body = False
        payload: object = None
        try:
            payload = response.json()
        except ValueError:
            invalid_body = True
        if invalid_body or not isinstance(payload, dict):
            # Raised OUTSIDE the except block so a JSON error cannot carry the
            # response body into __context__.
            raise GravityClassifierError("invalid Gravity guardrail response") from None

        verdict = payload.get("verdict")
        if verdict == "blocked":
            return ClassificationResult(flagged=True)
        if verdict == "allowed":
            return ClassificationResult(flagged=False)
        raise GravityClassifierError("invalid Gravity guardrail response")

    async def aclose(self) -> None:
        """Close the HTTP client, but only if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GravityClassifier":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
