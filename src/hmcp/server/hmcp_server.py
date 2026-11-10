"""HMCP server: a thin HMCP-specific layer over the MCP SDK's ``MCPServer``.

HMCP 0.1.0 targets MCP spec 2026-07-28 and builds on
:class:`mcp.server.mcpserver.MCPServer`. :class:`HMCPServer` adds:

* authentication defaults -- when an :class:`~mcp.server.auth.settings.AuthSettings`
  object is supplied without an explicit ``validate_token_resource``, it is
  defaulted to ``True`` so RFC 8707 audience binding is enforced by default; and a
  warning is logged once when the server is constructed with no authentication
  configured at all;
* patient context -- when a :class:`~hmcp.server.patient_context.PatientContextConfig`
  is supplied, an outermost
  :class:`~hmcp.server.patient_context.PatientContextExtension` is installed. Tools
  can then declare a ``patient_scope``; the effective scope is written into each
  tool's ``_meta`` so it is advertised, and the extension enforces the same scope at
  call time. Patient-scoped tools fail closed without an opened context. The
  extension is reachable through the :attr:`HMCPServer.patient_contexts` property;
* guardrails -- when a :class:`~hmcp.server.guardrails.GuardrailsConfig` is supplied,
  a :class:`~hmcp.server.guardrails.GuardrailsExtension` is installed inside the
  patient-context extension, so guardrails always see the resolved patient context.
  It is reachable through the :attr:`HMCPServer.guardrails` property; and
* transport restriction -- only ``stdio`` and ``streamable-http`` are supported.
  HTTP+SSE was deprecated by MCP 2026-07-28 and is rejected by HMCP 0.1.0.

Sampling was removed in HMCP 0.1.0 and is not offered by this server.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, Sequence

from mcp.server.auth.provider import OAuthAuthorizationServerProvider, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.extension import Extension
from mcp.server.mcpserver import MCPServer
from mcp.types import Icon, ToolAnnotations

from hmcp.server.guardrails import GuardrailsConfig, GuardrailsExtension
from hmcp.server.patient_context import (
    PatientContextConfig,
    PatientContextExtension,
)
from hmcp.shared.constants import META_SCOPE, PATIENT_CONTEXT_EXTENSION
from hmcp.shared.types import ToolPatientScope

logger = logging.getLogger(__name__)

# The only patient scopes a tool may declare.
_VALID_PATIENT_SCOPES: tuple[ToolPatientScope, ...] = ("patient", "population", "none")

# Raised whenever an HTTP+SSE transport is requested. It must mention the spec
# revision that deprecated SSE.
_SSE_UNSUPPORTED_MESSAGE = (
    "HTTP+SSE transport is deprecated by MCP 2026-07-28 and is not supported in "
    'HMCP 0.1.0; use "streamable-http".'
)

# Logged once when patient context is configured on a server that has no
# authentication and left allow_unauthenticated at its default False: with no
# principal every patient-context request is denied up front.
_PATIENT_CONTEXT_NO_AUTH_WARNING = (
    "patient_context is configured but the server has no authentication (no "
    "token_verifier and no auth_server_provider) and allow_unauthenticated is "
    "False: every patient-context request (open/get/close and every patient- or "
    "population-scoped tool call) will be denied with PATIENT_NOT_AUTHORIZED. "
    "Configure a token_verifier (with auth) so callers can authenticate, or for "
    "local development set PatientContextConfig(allow_unauthenticated=True) and "
    "supply a policy that permits anonymous callers: the default "
    "SmartLaunchPolicy still denies because there is no token to carry a SMART "
    "patient-launch claim."
)


class HMCPServer(MCPServer):
    """``MCPServer`` with HMCP's auth defaults, patient context and transport rule.

    See the module docstring for exactly what this subclass adds.
    """

    def __init__(
        self,
        name: str,
        *,
        version: str = "",
        instructions: str | None = None,
        token_verifier: TokenVerifier | None = None,
        auth: AuthSettings | None = None,
        auth_server_provider: OAuthAuthorizationServerProvider | None = None,
        extensions: Sequence[Extension] | None = None,
        patient_context: PatientContextConfig | None = None,
        guardrails: GuardrailsConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Tools passed to the MCPServer constructor bypass add_tool, so their
        # patient scope would never be written to _meta: the advertised scope
        # would then differ from the enforced scope. Refuse that combination.
        if patient_context is not None and kwargs.get("tools") is not None:
            raise ValueError(
                "register tools with add_tool()/tool() when patient_context is "
                "configured, not via the MCPServer 'tools=' argument: 'tools=' "
                "bypasses add_tool, so the advertised patient scope would differ "
                "from the enforced scope."
            )

        # Enforce RFC 8707 audience binding by default, without model_copy (which
        # skips AuthSettings validators and could report binding ON while nothing
        # is checked). An explicit False is kept as-is; audience binding is only
        # meaningful with a resource_server_url.
        if auth is not None and auth.validate_token_resource is not False:
            if auth.resource_server_url is None:
                raise ValueError(
                    "HMCP needs resource_server_url to audience-bind tokens per "
                    "RFC 8707; set validate_token_resource=False only if the token "
                    "verifier checks the token's audience itself."
                )
            if auth.validate_token_resource is None:
                auth = AuthSettings.model_validate(
                    {**auth.model_dump(), "validate_token_resource": True}
                )

        # Patient context is set up before super().__init__ so add_tool (which the
        # SDK may call while initializing) can already see the configuration.
        self._patient_context_config = patient_context
        self._patient_context_extension = (
            PatientContextExtension(patient_context, scope_for=self._scope_for)
            if patient_context is not None
            else None
        )
        self._guardrails_config = guardrails
        self._guardrails_extension = (
            GuardrailsExtension(guardrails, scope_for=self._scope_for)
            if guardrails is not None
            else None
        )

        # Extension order: patient context outermost, then guardrails (so it sees
        # the resolved context), then the caller's extensions.
        combined_extensions: list[Extension] = []
        if self._patient_context_extension is not None:
            combined_extensions.append(self._patient_context_extension)
        if self._guardrails_extension is not None:
            combined_extensions.append(self._guardrails_extension)
        if extensions:
            combined_extensions.extend(extensions)

        if token_verifier is None and auth_server_provider is None:
            logger.warning("HMCP server running without authentication")
            if (
                patient_context is not None
                and not patient_context.allow_unauthenticated
            ):
                logger.warning(_PATIENT_CONTEXT_NO_AUTH_WARNING)

        super().__init__(
            name,
            version=version,
            instructions=instructions,
            token_verifier=token_verifier,
            auth=auth,
            auth_server_provider=auth_server_provider,
            extensions=combined_extensions or None,
            **kwargs,
        )

    # -- patient context -------------------------------------------------

    @property
    def patient_contexts(self) -> PatientContextExtension | None:
        """The installed patient-context extension, or ``None`` if not configured."""
        return self._patient_context_extension

    @property
    def guardrails(self) -> GuardrailsExtension | None:
        """The installed guardrails extension, or ``None`` if not configured."""
        return self._guardrails_extension

    def add_tool(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        annotations: ToolAnnotations | None = None,
        icons: list[Icon] | None = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
        *,
        patient_scope: ToolPatientScope | None = None,
    ) -> None:
        """Register a tool, writing its effective patient scope into its ``_meta``."""
        if patient_scope is not None and patient_scope not in _VALID_PATIENT_SCOPES:
            raise ValueError(
                f"patient_scope must be one of {_VALID_PATIENT_SCOPES}; "
                f"got {patient_scope!r}"
            )
        config: PatientContextConfig | None = getattr(
            self, "_patient_context_config", None
        )
        if patient_scope is not None and config is None:
            raise ValueError(
                "patient_scope requires a configured patient_context; construct "
                "HMCPServer(..., patient_context=PatientContextConfig(...))"
            )
        if config is not None:
            scope = patient_scope or config.default_tool_scope
            merged = dict(meta or {})
            nested = dict(merged.get(PATIENT_CONTEXT_EXTENSION) or {})
            nested[META_SCOPE] = scope
            merged[PATIENT_CONTEXT_EXTENSION] = nested
            meta = merged
        super().add_tool(
            fn,
            name=name,
            title=title,
            description=description,
            annotations=annotations,
            icons=icons,
            meta=meta,
            structured_output=structured_output,
        )

    def tool(
        self,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        annotations: ToolAnnotations | None = None,
        icons: list[Icon] | None = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
        *,
        patient_scope: ToolPatientScope | None = None,
    ) -> Callable[[Any], Any]:
        """Decorator form of :meth:`add_tool`; forwards ``patient_scope``."""
        if callable(name):
            raise TypeError(
                "The @tool decorator was used incorrectly. Did you forget to call "
                "it? Use @tool() instead of @tool"
            )

        def decorator(fn: Any) -> Any:
            self.add_tool(
                fn,
                name=name,
                title=title,
                description=description,
                annotations=annotations,
                icons=icons,
                meta=meta,
                structured_output=structured_output,
                patient_scope=patient_scope,
            )
            return fn

        return decorator

    def _scope_for(self, name: str) -> ToolPatientScope:
        """Return the scope registered for ``name`` (advertised == enforced)."""
        config: PatientContextConfig | None = getattr(
            self, "_patient_context_config", None
        )
        default: ToolPatientScope = (
            config.default_tool_scope if config is not None else "patient"
        )
        tool = self._tool_manager.get_tool(name)
        if tool is not None and tool.meta:
            nested = tool.meta.get(PATIENT_CONTEXT_EXTENSION)
            if isinstance(nested, dict) and META_SCOPE in nested:
                return nested[META_SCOPE]
        return default

    # -- transport -------------------------------------------------------

    def run(
        self,
        transport: Literal["stdio", "streamable-http"] = "stdio",
        **kwargs: Any,
    ) -> None:
        """Run the server over a supported transport, rejecting HTTP+SSE."""
        if transport == "sse":
            raise ValueError(_SSE_UNSUPPORTED_MESSAGE)
        if transport not in ("stdio", "streamable-http"):
            raise ValueError(
                f"unsupported transport {transport!r}; supported transports are "
                "'stdio' and 'streamable-http'"
            )
        super().run(transport=transport, **kwargs)

    def sse_app(self, *args: Any, **kwargs: Any) -> Any:
        """HTTP+SSE is not offered by HMCP; always raises ``ValueError``."""
        raise ValueError(_SSE_UNSUPPORTED_MESSAGE)
