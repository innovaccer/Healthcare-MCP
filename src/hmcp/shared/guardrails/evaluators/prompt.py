"""LLM-prompt evaluator for HMCP 0.1.0 guardrails.

The enforcing party runs a prompt-kind guardrail on its OWN LLM provider
(MCP sampling is deprecated). Content is chunked, each chunk is sent with a
fresh nonce, and the reply is parsed into findings located in the chunk's
units. Everything fails closed and never logs content.
"""

import json
import secrets
from typing import Protocol, Sequence

from ..definition import PromptEvaluatorSpec
from ..findings import Finding, locate_all
from .base import EvaluationContext, EvaluationError, TextUnit

_CHUNK_SEPARATOR = "\n---\n"


class LLMProvider(Protocol):
    """The LLM a deployer supplies for a prompt-kind guardrail.

    HMCP builds the ``system`` and ``user`` strings itself, framing the
    inspected content between fresh per-call nonces, and calls ``complete`` with
    them. ``complete`` returns the model's reply as a string; the provider must
    not log the content it is given. Any exception raised makes the guardrail
    fail closed.
    """

    async def complete(self, *, system: str, user: str, max_tokens: int) -> str: ...


class PromptEvaluator:
    """Find PHI by sending chunks of the inspected text to an LLMProvider."""

    kind = "prompt"

    def __init__(self, spec: PromptEvaluatorSpec, llm: LLMProvider) -> None:
        self._spec = spec
        self._llm = llm

    async def find(
        self, units: Sequence[TextUnit], ctx: EvaluationContext
    ) -> list[list[Finding]]:
        results: list[list[Finding]] = [[] for _ in units]
        if not units:
            return results

        categories = ", ".join(category.value for category in self._spec.categories)
        allowed = {category.value for category in self._spec.categories}

        offset = 0
        for chunk in _chunk(units, self._spec.max_chars_per_call):
            nonce = secrets.token_hex(8)
            content = _CHUNK_SEPARATOR.join(unit.text for unit in chunk)
            system = _render(self._spec.system, nonce, categories, content)
            user = _render(self._spec.user, nonce, categories, content)

            llm_failed = False
            try:
                reply = await self._llm.complete(
                    system=system, user=user, max_tokens=self._spec.max_tokens
                )
            except Exception:  # noqa: BLE001 - fail closed, but keep the cause clean
                llm_failed = True
            if llm_failed:
                # Raised OUTSIDE the except block so __context__ stays None: a
                # provider message could otherwise carry PHI into the exception.
                raise EvaluationError("LLM call failed") from None

            for item in _parse_verdict(reply):
                text = item["text"]
                category = item["category"]
                if category not in allowed:
                    continue
                for unit_index, unit in enumerate(chunk):
                    for start, end in locate_all(unit.text, text):
                        results[offset + unit_index].append(
                            Finding(
                                category=category,
                                start=start,
                                end=end,
                                source="prompt",
                                value=text,
                            )
                        )
            offset += len(chunk)

        return results


def _chunk(units: Sequence[TextUnit], max_chars: int) -> list[list[TextUnit]]:
    """Pack consecutive units so the joined text stays <= max_chars.

    A single longer unit forms its own chunk and is never truncated.
    """
    chunks: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    current_len = 0
    for unit in units:
        added = (
            len(unit.text)
            if not current
            else current_len + len(_CHUNK_SEPARATOR) + len(unit.text)
        )
        if current and added > max_chars:
            chunks.append(current)
            current = [unit]
            current_len = len(unit.text)
        else:
            current.append(unit)
            current_len = added
    if current:
        chunks.append(current)
    return chunks


def _render(template: str, nonce: str, categories: str, content: str) -> str:
    # Order matters: content is substituted LAST so content containing
    # "{{nonce}}" (or "{{categories}}") is never expanded.
    rendered = template.replace("{{nonce}}", nonce)
    rendered = rendered.replace("{{categories}}", categories)
    return rendered.replace("{{content}}", content)


def _parse_verdict(reply: str) -> list[dict]:
    if not isinstance(reply, str):
        raise EvaluationError("invalid LLM verdict")
    body = reply.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rstrip()
        if body.endswith("```"):
            body = body[:-3]
    body = body.strip()

    parse_failed = False
    try:
        data = json.loads(body)
    except ValueError:
        parse_failed = True
    if parse_failed:
        # Raised OUTSIDE the except block so __context__ stays None: a decoder
        # error can quote the LLM reply, which may echo inspected content.
        raise EvaluationError("invalid LLM verdict") from None

    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        raise EvaluationError("invalid LLM verdict")

    items: list[dict] = []
    for item in data["findings"]:
        if not isinstance(item, dict):
            raise EvaluationError("invalid LLM verdict")
        text = item.get("text")
        category = item.get("category")
        if not isinstance(text, str) or not text:
            raise EvaluationError("invalid LLM verdict")
        if not isinstance(category, str):
            raise EvaluationError("invalid LLM verdict")
        items.append({"text": text, "category": category})
    return items
