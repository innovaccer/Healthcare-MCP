"""Patient-context MCP extension for HMCP 0.1.0.

The extension turns the authenticated principal plus a server-minted opaque
handle into HMCP's session concept. The host (never the model) opens a context
through the extension methods; every patient-scoped tool call then carries the
handle in request ``_meta`` and the interceptor binds it for the duration of the
call. Everything fails closed and every decision is audited.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, get_args

import mcp.types as types
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from pydantic import Field, ValidationError

from hmcp.server.patient_context.audit import (
    AuditAction,
    AuditEvent,
    AuditOutcome,
    AuditSink,
    LoggingAuditSink,
)
from hmcp.server.patient_context.policy import AccessPolicy, SmartLaunchPolicy
from hmcp.server.patient_context.store import ContextStore, InMemoryContextStore
from hmcp.shared.constants import (
    CONTEXT_ID_PREFIX,
    GUARDRAILS_EXTENSION,
    META_CONTEXT_ID,
    META_REPORT,
    METHOD_PATIENT_CONTEXT_CLOSE,
    METHOD_PATIENT_CONTEXT_GET,
    METHOD_PATIENT_CONTEXT_OPEN,
    PATIENT_CONTEXT_EXTENSION,
    PATIENT_CONTEXT_REVISION,
)
from hmcp.shared.errors import HMCPErrorCode, hmcp_error
from hmcp.shared.types import (
    ContextStatus,
    PatientContext,
    PatientRef,
    Principal,
    ToolPatientScope,
    bind_patient_context,
)

logger = logging.getLogger("hmcp.patient_context")

_MIN_TTL_SECONDS = 60
_MAX_TTL_SECONDS = 86_400
_VALID_TOOL_SCOPES: frozenset[str] = frozenset(get_args(ToolPatientScope))

# Message for the no-principal case. It must describe only the request's
# authentication state, never the patient or any other request detail.
_NOT_AUTHENTICATED_MESSAGE = "Request is not authenticated"


@dataclass
class PatientContextConfig:
    """Configuration for :class:`PatientContextExtension`.

    ``allow_unauthenticated`` is a development-only escape hatch: when ``True`` a
    request that carries no access token is treated as one shared anonymous
    principal so it reaches the access policy instead of being rejected up front.
    It does not by itself grant access. The default
    :class:`~hmcp.server.patient_context.policy.SmartLaunchPolicy` still denies
    the anonymous principal, because there is no token to carry a SMART
    patient-launch claim; a deployment must also supply a policy that permits
    anonymous callers to make use of it. Left ``False``, a server with no
    authentication denies every patient-context request.
    """

    store: ContextStore = field(default_factory=InMemoryContextStore)
    policy: AccessPolicy = field(default_factory=SmartLaunchPolicy)
    audit: AuditSink = field(default_factory=LoggingAuditSink)
    context_ttl_seconds: int = 3600
    default_tool_scope: ToolPatientScope = "patient"
    allow_unauthenticated: bool = False

    def __post_init__(self) -> None:
        if not (_MIN_TTL_SECONDS <= self.context_ttl_seconds <= _MAX_TTL_SECONDS):
            raise ValueError(
                f"context_ttl_seconds must be between {_MIN_TTL_SECONDS} "
                f"and {_MAX_TTL_SECONDS}"
            )
        if self.default_tool_scope not in _VALID_TOOL_SCOPES:
            raise ValueError(
                "default_tool_scope must be one of " f"{sorted(_VALID_TOOL_SCOPES)}"
            )
        if self.allow_unauthenticated:
            logger.warning(
                "patient context allow_unauthenticated=True: dev mode only, "
                "every unauthenticated caller is treated as one anonymous principal"
            )


class PatientContextOpenParams(types.RequestParams):
    """Params for ``hmcp/patientContext/open``.

    ``purposeOfUse`` and ``encounter`` enforce the same patterns as
    :class:`~hmcp.shared.types.PatientContext`, so a bad value is rejected by
    params validation before the policy runs.
    """

    patient: PatientRef
    encounter: str | None = Field(
        default=None, pattern=r"^Encounter/[A-Za-z0-9\-\.]{1,64}$"
    )
    purpose_of_use: str = Field(pattern=r"^[A-Z]{3,12}$")
    correlation_id: str | None = None


class PatientContextIdParams(types.RequestParams):
    """Params for ``hmcp/patientContext/get`` and ``.../close``."""

    context_id: str


def _verdict_value(verdict: Any) -> str:
    return str(getattr(verdict, "value", verdict))


def _guardrail_pairs(report: Any) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for entry in report or ():
        if isinstance(entry, dict):
            gid, verdict = entry.get("id"), entry.get("verdict")
        else:
            gid, verdict = getattr(entry, "id", None), getattr(entry, "verdict", None)
        if gid is not None and verdict is not None:
            pairs.append((str(gid), _verdict_value(verdict)))
    return tuple(pairs)


class PatientContextExtension(Extension):
    """HMCP patient-context extension: methods, interceptor and lifecycle.

    A context is bound to the :class:`~hmcp.shared.types.Principal` derived from
    the access token (client id, ``iss`` and subject), not to the raw token, so a
    token refreshed mid-context keeps the same handle. Binding is therefore only
    as strong as the verifier's identity claims: two tokens of the same client
    that carry neither a subject nor an ``iss`` claim resolve to the same
    principal, and a handle opened with one can be used with the other. Verifiers
    must populate ``AccessToken.subject`` (and the ``iss`` claim) for per-user
    binding.

    With no authentication configured (no token verifier) and
    ``PatientContextConfig.allow_unauthenticated`` left ``False``, every request
    has no principal: ``open``/``get``/``close`` and every patient- or
    population-scoped tool call are denied with ``PATIENT_NOT_AUTHORIZED`` and a
    message saying the request is not authenticated.
    """

    identifier = PATIENT_CONTEXT_EXTENSION

    def __init__(
        self,
        config: PatientContextConfig,
        scope_for: Callable[[str], ToolPatientScope],
    ) -> None:
        super().__init__()
        self._config = config
        self._scope_for = scope_for

    # -- advertisement ---------------------------------------------------

    def settings(self) -> dict[str, Any]:
        return {
            "revision": PATIENT_CONTEXT_REVISION,
            "contextTtlSeconds": self._config.context_ttl_seconds,
            "defaultToolScope": self._config.default_tool_scope,
        }

    def methods(self) -> Sequence[MethodBinding]:
        return [
            MethodBinding(
                method=METHOD_PATIENT_CONTEXT_OPEN,
                params_type=PatientContextOpenParams,
                handler=self._handle_open,
            ),
            MethodBinding(
                method=METHOD_PATIENT_CONTEXT_GET,
                params_type=PatientContextIdParams,
                handler=self._handle_get,
            ),
            MethodBinding(
                method=METHOD_PATIENT_CONTEXT_CLOSE,
                params_type=PatientContextIdParams,
                handler=self._handle_close,
            ),
        ]

    # -- audit helpers ---------------------------------------------------

    def _log_exception(self, exc: BaseException) -> None:
        # Never log the exception message: it may carry PHI.
        logger.warning("patient-context extension error: %s", type(exc).__name__)

    async def _emit(
        self,
        *,
        action: AuditAction,
        outcome: AuditOutcome,
        principal: Principal | None = None,
        context_id: str | None = None,
        patient_reference: str | None = None,
        purpose_of_use: str | None = None,
        tool: str | None = None,
        error_code: HMCPErrorCode | None = None,
        reason: str | None = None,
        guardrails: tuple[tuple[str, str], ...] = (),
        correlation_id: str | None = None,
    ) -> None:
        await self._config.audit.emit(
            AuditEvent(
                timestamp=datetime.now(timezone.utc),
                action=action,
                outcome=outcome,
                principal=principal,
                context_id=context_id,
                patient_reference=patient_reference,
                purpose_of_use=purpose_of_use,
                tool=tool,
                error_code=int(error_code) if error_code is not None else None,
                reason=reason,
                guardrails=guardrails,
                correlation_id=correlation_id,
            )
        )

    async def _emit_best_effort(self, **kwargs: Any) -> None:
        try:
            await self._emit(**kwargs)
        except Exception as exc:  # best effort: log the type only
            self._log_exception(exc)

    # -- principal resolution --------------------------------------------

    def _resolve_principal(self) -> Principal | None:
        token = get_access_token()
        if token is not None:
            return Principal.from_access_token(token)
        if self._config.allow_unauthenticated:
            return Principal.anonymous()
        return None

    # -- extension methods -----------------------------------------------

    async def _handle_open(
        self, ctx: Any, params: PatientContextOpenParams
    ) -> dict[str, Any]:
        principal = self._resolve_principal()
        if principal is None:
            await self._emit_best_effort(
                action="patient_context.open",
                outcome="denied",
                error_code=HMCPErrorCode.PATIENT_NOT_AUTHORIZED,
                reason="unauthenticated",
                patient_reference=params.patient.reference,
                purpose_of_use=params.purpose_of_use,
            )
            raise hmcp_error(
                HMCPErrorCode.PATIENT_NOT_AUTHORIZED, _NOT_AUTHENTICATED_MESSAGE
            )

        token = get_access_token()
        unavailable: MCPError | None = None
        try:
            decision = await self._config.policy.authorize_open(
                principal=principal,
                token=token,
                patient=params.patient,
                purpose_of_use=params.purpose_of_use,
            )
        except Exception as exc:
            self._log_exception(exc)
            await self._emit_best_effort(
                action="patient_context.open",
                outcome="error",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                patient_reference=params.patient.reference,
                purpose_of_use=params.purpose_of_use,
            )
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        if not decision.allowed:
            await self._emit_best_effort(
                action="patient_context.open",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_NOT_AUTHORIZED,
                reason=decision.reason,
                patient_reference=params.patient.reference,
                purpose_of_use=params.purpose_of_use,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)

        now = datetime.now(timezone.utc)
        unavailable = None
        try:
            context = PatientContext(
                context_id=CONTEXT_ID_PREFIX + secrets.token_urlsafe(32),
                patient=PatientRef(
                    reference=params.patient.reference,
                    identifiers=decision.verified_identifiers,
                ),
                encounter=params.encounter,
                purpose_of_use=params.purpose_of_use,
                principal=principal,
                status=ContextStatus.ACTIVE,
                created_at=now,
                expires_at=now + timedelta(seconds=self._config.context_ttl_seconds),
                correlation_id=params.correlation_id,
            )
        except ValidationError as exc:
            self._log_exception(exc)
            await self._emit_best_effort(
                action="patient_context.open",
                outcome="error",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                patient_reference=params.patient.reference,
                purpose_of_use=params.purpose_of_use,
            )
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        # The attempt emit before store.put is fail-closed: a sink failure must
        # not leave a stored context whose id the host never received.
        unavailable = None
        try:
            await self._emit(
                action="patient_context.open",
                outcome="attempt",
                principal=principal,
                patient_reference=context.patient.reference,
                purpose_of_use=context.purpose_of_use,
                correlation_id=context.correlation_id,
            )
        except Exception as exc:
            self._log_exception(exc)
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        unavailable = None
        try:
            await self._config.store.put(context)
        except Exception as exc:
            self._log_exception(exc)
            await self._emit_best_effort(
                action="patient_context.open",
                outcome="error",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                patient_reference=context.patient.reference,
                purpose_of_use=context.purpose_of_use,
            )
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        await self._emit_best_effort(
            action="patient_context.open",
            outcome="success",
            principal=principal,
            context_id=context.context_id,
            patient_reference=context.patient.reference,
            purpose_of_use=context.purpose_of_use,
            correlation_id=context.correlation_id,
        )
        return {
            "contextId": context.context_id,
            "status": context.status.value,
            "expiresAt": context.expires_at.isoformat(),
        }

    async def _load_for_principal(
        self, principal: Principal, context_id: str
    ) -> PatientContext:
        unavailable = None
        try:
            context = await self._config.store.get(context_id)
        except Exception as exc:
            self._log_exception(exc)
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable
        if context is None or context.principal != principal:
            raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_INVALID)
        return context

    async def _handle_get(
        self, ctx: Any, params: PatientContextIdParams
    ) -> dict[str, Any]:
        principal = self._resolve_principal()
        if principal is None:
            await self._emit_best_effort(
                action="patient_context.get",
                outcome="denied",
                error_code=HMCPErrorCode.PATIENT_NOT_AUTHORIZED,
                reason="unauthenticated",
                context_id=params.context_id,
            )
            raise hmcp_error(
                HMCPErrorCode.PATIENT_NOT_AUTHORIZED, _NOT_AUTHENTICATED_MESSAGE
            )

        unavailable = None
        try:
            context = await self._load_for_principal(principal, params.context_id)
        except MCPError as exc:
            await self._emit_best_effort(
                action="patient_context.get",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode(exc.code),
                reason="invalid_context",
                context_id=params.context_id,
            )
            raise
        except Exception as exc:
            self._log_exception(exc)
            await self._emit_best_effort(
                action="patient_context.get",
                outcome="error",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                context_id=params.context_id,
            )
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        await self._emit_best_effort(
            action="patient_context.get",
            outcome="success",
            principal=principal,
            context_id=context.context_id,
            patient_reference=context.patient.reference,
            purpose_of_use=context.purpose_of_use,
        )
        return {
            "contextId": context.context_id,
            "status": context.status.value,
            "expiresAt": context.expires_at.isoformat(),
            "patient": context.patient.model_dump(by_alias=True, mode="json"),
            "encounter": context.encounter,
            "purposeOfUse": context.purpose_of_use,
        }

    async def _handle_close(
        self, ctx: Any, params: PatientContextIdParams
    ) -> dict[str, Any]:
        principal = self._resolve_principal()
        if principal is None:
            await self._emit_best_effort(
                action="patient_context.close",
                outcome="denied",
                error_code=HMCPErrorCode.PATIENT_NOT_AUTHORIZED,
                reason="unauthenticated",
                context_id=params.context_id,
            )
            raise hmcp_error(
                HMCPErrorCode.PATIENT_NOT_AUTHORIZED, _NOT_AUTHENTICATED_MESSAGE
            )

        unavailable = None
        try:
            context = await self._load_for_principal(principal, params.context_id)
        except MCPError as exc:
            await self._emit_best_effort(
                action="patient_context.close",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode(exc.code),
                reason="invalid_context",
                context_id=params.context_id,
            )
            raise
        except Exception as exc:
            self._log_exception(exc)
            await self._emit_best_effort(
                action="patient_context.close",
                outcome="error",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                context_id=params.context_id,
            )
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        if context.status in (ContextStatus.ACTIVE, ContextStatus.SUSPENDED):
            unavailable = None
            try:
                updated = await self._config.store.set_status(
                    context.context_id, ContextStatus.CLOSED
                )
            except Exception as exc:
                self._log_exception(exc)
                await self._emit_best_effort(
                    action="patient_context.close",
                    outcome="error",
                    principal=principal,
                    error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                    context_id=params.context_id,
                )
                unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
            if unavailable is not None:
                raise unavailable
            final_status = updated.status if updated is not None else context.status
        else:
            # Terminal state: no change, report the current status.
            final_status = context.status

        await self._emit_best_effort(
            action="patient_context.close",
            outcome="success",
            principal=principal,
            context_id=context.context_id,
            patient_reference=context.patient.reference,
            purpose_of_use=context.purpose_of_use,
        )
        return {"contextId": context.context_id, "status": final_status.value}

    # -- tool-call interceptor -------------------------------------------

    async def intercept_tool_call(
        self, params: Any, ctx: Any, call_next: Callable[[Any], Any]
    ) -> Any:
        scope = self._scope_for(params.name)
        if scope == "none":
            return await call_next(ctx)

        principal = self._resolve_principal()
        if principal is None:
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                error_code=HMCPErrorCode.PATIENT_NOT_AUTHORIZED,
                reason="unauthenticated",
                tool=params.name,
            )
            raise hmcp_error(
                HMCPErrorCode.PATIENT_NOT_AUTHORIZED, _NOT_AUTHENTICATED_MESSAGE
            )

        token = get_access_token()

        if scope == "population":
            return await self._intercept_population(
                params, ctx, call_next, principal, token
            )

        return await self._intercept_patient(params, ctx, call_next, principal, token)

    async def _intercept_population(
        self,
        params: Any,
        ctx: Any,
        call_next: Callable[[Any], Any],
        principal: Principal,
        token: Any,
    ) -> Any:
        unavailable: MCPError | None = None
        try:
            decision = await self._config.policy.authorize_population(
                principal=principal, token=token, tool_name=params.name
            )
        except Exception as exc:
            self._log_exception(exc)
            await self._emit_best_effort(
                action="tool.call",
                outcome="error",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                tool=params.name,
            )
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        if not decision.allowed:
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_NOT_AUTHORIZED,
                reason=decision.reason,
                tool=params.name,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)

        unavailable = None
        try:
            await self._emit(
                action="tool.call",
                outcome="attempt",
                principal=principal,
                tool=params.name,
            )
        except Exception as exc:
            self._log_exception(exc)
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        try:
            result = await call_next(ctx)
        except Exception as exc:
            await self._emit_best_effort(
                action="tool.call",
                outcome="error",
                principal=principal,
                tool=params.name,
            )
            raise
        await self._emit_tool_outcome(principal, params.name, result)
        return result

    async def _intercept_patient(
        self,
        params: Any,
        ctx: Any,
        call_next: Callable[[Any], Any],
        principal: Principal,
        token: Any,
    ) -> Any:
        container = (params.meta or {}).get(PATIENT_CONTEXT_EXTENSION)
        if not isinstance(container, dict):
            container = {}
        context_id = container.get(META_CONTEXT_ID)
        if not (isinstance(context_id, str) and context_id):
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_REQUIRED,
                reason="missing_context_id",
                tool=params.name,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_REQUIRED)

        unavailable: MCPError | None = None
        try:
            context = await self._config.store.get(context_id)
        except Exception as exc:
            self._log_exception(exc)
            await self._emit_best_effort(
                action="tool.call",
                outcome="error",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE,
                tool=params.name,
            )
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        if context is None or context.principal != principal:
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_INVALID,
                reason="invalid_context",
                tool=params.name,
                context_id=context_id,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_INVALID)

        # Expiry applies to ACTIVE and SUSPENDED contexts (terminal states are
        # sticky and are handled below); an elapsed TTL wins over the status.
        now = datetime.now(timezone.utc)
        if (
            context.status in (ContextStatus.ACTIVE, ContextStatus.SUSPENDED)
            and now >= context.expires_at
        ):
            try:
                await self._config.store.set_status(context_id, ContextStatus.EXPIRED)
            except Exception as exc:
                self._log_exception(exc)
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_INVALID,
                reason="expired",
                tool=params.name,
                context_id=context_id,
                patient_reference=context.patient.reference,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_INVALID)

        if context.status is ContextStatus.SUSPENDED:
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_SUSPENDED,
                reason="suspended",
                tool=params.name,
                context_id=context_id,
                patient_reference=context.patient.reference,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_SUSPENDED)

        if context.status is ContextStatus.REVOKED:
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_REVOKED,
                reason="revoked",
                tool=params.name,
                context_id=context_id,
                patient_reference=context.patient.reference,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_REVOKED)

        if context.status in (ContextStatus.CLOSED, ContextStatus.EXPIRED):
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_CONTEXT_INVALID,
                reason="not_active",
                tool=params.name,
                context_id=context_id,
                patient_reference=context.patient.reference,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_INVALID)

        claim = (token.claims or {}).get("patient") if token else None
        if isinstance(claim, str) and context.patient.reference != f"Patient/{claim}":
            await self._emit_best_effort(
                action="tool.call",
                outcome="denied",
                principal=principal,
                error_code=HMCPErrorCode.PATIENT_NOT_AUTHORIZED,
                reason="token_patient_mismatch",
                tool=params.name,
                context_id=context_id,
                patient_reference=context.patient.reference,
            )
            raise hmcp_error(HMCPErrorCode.PATIENT_NOT_AUTHORIZED)

        unavailable = None
        try:
            await self._emit(
                action="tool.call",
                outcome="attempt",
                principal=principal,
                tool=params.name,
                context_id=context_id,
                patient_reference=context.patient.reference,
            )
        except Exception as exc:
            self._log_exception(exc)
            unavailable = hmcp_error(HMCPErrorCode.PATIENT_CONTEXT_UNAVAILABLE)
        if unavailable is not None:
            raise unavailable

        try:
            with bind_patient_context(context):
                result = await call_next(ctx)
        except Exception as exc:
            await self._emit_best_effort(
                action="tool.call",
                outcome="error",
                principal=principal,
                tool=params.name,
                context_id=context_id,
                patient_reference=context.patient.reference,
            )
            raise
        await self._emit_tool_outcome(principal, params.name, result, context)
        return result

    async def _emit_tool_outcome(
        self,
        principal: Principal,
        tool_name: str,
        result: Any,
        context: PatientContext | None = None,
    ) -> None:
        is_error = isinstance(result, types.CallToolResult) and bool(result.is_error)
        guardrails: tuple[tuple[str, str], ...] = ()
        if isinstance(result, types.CallToolResult):
            meta = result.meta or {}
            container = meta.get(GUARDRAILS_EXTENSION)
            report = (
                container.get(META_REPORT, []) if isinstance(container, dict) else []
            )
            if isinstance(report, list):
                guardrails = _guardrail_pairs(report)
        await self._emit_best_effort(
            action="tool.call",
            outcome="error" if is_error else "success",
            principal=principal,
            tool=tool_name,
            context_id=context.context_id if context else None,
            patient_reference=context.patient.reference if context else None,
            guardrails=guardrails,
        )

    # -- lifecycle API ---------------------------------------------------

    async def _transition(
        self,
        *,
        action: AuditAction,
        target: ContextStatus,
        allowed_from: frozenset[ContextStatus],
        context_id: str | None,
        patient_reference: str | None,
        subject: str | None,
    ) -> int:
        if context_id is not None:
            single = await self._config.store.get(context_id)
            contexts = [single] if single is not None else []
        else:
            contexts = await self._config.store.find(
                patient_reference=patient_reference, subject=subject
            )
        count = 0
        for context in contexts:
            if context.status not in allowed_from:
                continue
            updated = await self._config.store.set_status(context.context_id, target)
            if updated is None:
                continue
            count += 1
            await self._emit_best_effort(
                action=action,
                outcome="success",
                principal=updated.principal,
                context_id=updated.context_id,
                patient_reference=updated.patient.reference,
                purpose_of_use=updated.purpose_of_use,
            )
        return count

    async def revoke(
        self,
        *,
        context_id: str | None = None,
        patient_reference: str | None = None,
        subject: str | None = None,
    ) -> int:
        return await self._transition(
            action="patient_context.revoke",
            target=ContextStatus.REVOKED,
            allowed_from=frozenset({ContextStatus.ACTIVE, ContextStatus.SUSPENDED}),
            context_id=context_id,
            patient_reference=patient_reference,
            subject=subject,
        )

    async def suspend(
        self,
        *,
        context_id: str | None = None,
        patient_reference: str | None = None,
        subject: str | None = None,
    ) -> int:
        return await self._transition(
            action="patient_context.suspend",
            target=ContextStatus.SUSPENDED,
            allowed_from=frozenset({ContextStatus.ACTIVE}),
            context_id=context_id,
            patient_reference=patient_reference,
            subject=subject,
        )

    async def resume(
        self,
        *,
        context_id: str | None = None,
        patient_reference: str | None = None,
        subject: str | None = None,
    ) -> int:
        return await self._transition(
            action="patient_context.resume",
            target=ContextStatus.ACTIVE,
            allowed_from=frozenset({ContextStatus.SUSPENDED}),
            context_id=context_id,
            patient_reference=patient_reference,
            subject=subject,
        )
