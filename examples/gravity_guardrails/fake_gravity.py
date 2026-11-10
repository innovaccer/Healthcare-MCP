"""Offline, deterministic stand-in for Gravity's guardrail invoke endpoint.

EXAMPLE/test support code: synthetic data only. It mimics the measured behaviour
of Gravity's prompt-injection config — injection text with checkType ``"input"``
is ``"blocked"``, the same text with checkType ``"output"`` is ``"allowed"`` in
0-1 ms because that config's output side is set to allow and the model is not
run — and it never echoes the request body back.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

import httpx2

INJECTION_MARKERS = (
    "ignore all previous instructions",
    "disregard your rules",
    "system override",
)

_DEFAULT_GUARDRAILS: dict[tuple[str, str], str] = {
    ("fake-prompt-injection", "fake-config"): "Prompt Injection Guardrail",
}

# Any api_base prefix is allowed; the tail must be the invoke path.
_INVOKE_PATH = re.compile(r"/guardrails/([^/]+)/configs/([^/]+)/invoke$")


class FakeGravity:
    """A MockTransport-backed fake of Gravity's guardrail invoke endpoint."""

    def __init__(
        self,
        *,
        token: str = "fake-gravity-token",
        tenant_id: str | None = None,
        guardrails: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self._token = token
        self._tenant_id = tenant_id
        self._guardrails = dict(
            _DEFAULT_GUARDRAILS if guardrails is None else guardrails
        )
        # Each successful invoke is recorded here, in order.
        self.requests: list[dict] = []

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(405, json={"detail": "method not allowed"})

        match = _INVOKE_PATH.search(request.url.path)
        if match is None:
            return httpx2.Response(404, json={"detail": "not found"})
        guardrail_id, config_id = match.groups()
        name = self._guardrails.get((guardrail_id, config_id))
        if name is None:
            return httpx2.Response(404, json={"detail": "not found"})

        if request.headers.get("Authorization") != f"Bearer {self._token}":
            return httpx2.Response(401, json={"detail": "unauthorized"})

        if self._tenant_id is not None and (
            request.headers.get("X-Tenant-ID") != self._tenant_id
        ):
            return httpx2.Response(403, json={"detail": "forbidden"})

        parsed = self._parse_body(request)
        if parsed is None:
            # Never echo the body.
            return httpx2.Response(422, json={"detail": "invalid request"})
        text, check_type = parsed

        self.requests.append(
            {"path": request.url.path, "checkType": check_type, "text": text}
        )

        if check_type == "output":
            # This config's output side is "allow": the model is not run.
            blocked = False
        else:
            blocked = any(marker in text.casefold() for marker in INJECTION_MARKERS)

        verdict = "blocked" if blocked else "allowed"
        message = "[Guardrail: Prompt Injection] blocked" if blocked else "allowed"
        return httpx2.Response(
            200,
            json={
                "verdict": verdict,
                "message": message,
                "latencyMs": 1,
                "configId": config_id,
                "guardrailName": name,
            },
        )

    @staticmethod
    def _parse_body(request: httpx2.Request) -> tuple[str, str] | None:
        try:
            body = json.loads(request.content)
        except (ValueError, TypeError):
            return None
        if not isinstance(body, dict):
            return None
        if set(body) != {"text", "checkType"}:
            return None
        text = body["text"]
        check_type = body["checkType"]
        if not isinstance(text, str) or not text:
            return None
        if check_type not in ("input", "output"):
            return None
        return text, check_type
