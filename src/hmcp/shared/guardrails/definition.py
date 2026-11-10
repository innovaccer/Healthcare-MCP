"""Declarative guardrail definitions for HMCP 0.1.0.

A guardrail is a portable, declarative definition (JSON), separate from the
evaluator implementation that runs it. All models are frozen, forbid extra
fields, and use camelCase aliases on the wire.
"""

import hashlib
import json
import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from hmcp.shared.types import GuardrailStage, ToolPatientScope


class Model(BaseModel):
    """Base model: camelCase wire aliases, frozen, extra fields forbidden."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        frozen=True,
        extra="forbid",
    )


class PhiCategory(str, Enum):
    """Vendor-neutral PHI category labels named after the HIPAA Safe Harbor identifier list. They are labels only; using them is not a Safe Harbor de-identification."""

    NAME = "NAME"
    DATE = "DATE"
    PHONE = "PHONE"
    FAX = "FAX"
    EMAIL = "EMAIL"
    SSN = "SSN"
    MRN = "MRN"
    HEALTH_PLAN_ID = "HEALTH_PLAN_ID"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    LICENSE_NUMBER = "LICENSE_NUMBER"
    VEHICLE_ID = "VEHICLE_ID"
    DEVICE_ID = "DEVICE_ID"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"
    BIOMETRIC_ID = "BIOMETRIC_ID"
    PHOTO = "PHOTO"
    LOCATION = "LOCATION"
    OTHER_ID = "OTHER_ID"


class IdentifierType(Model):
    """A PHI identifier type: text regexes and/or JSON keys, plus its category."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    category: PhiCategory
    # each must compile and have EXACTLY 1 capture group; used with re.IGNORECASE
    text_patterns: tuple[str, ...] = ()
    # matched case-insensitively against the dict key a string sits under
    json_keys: tuple[str, ...] = ()
    # EMPI/MRN-like keys that identify a patient for access control
    is_access_key: bool = False

    @field_validator("text_patterns")
    @classmethod
    def _validate_text_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            compile_error: str | None = None
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                compile_error = str(exc)
            if compile_error is not None:
                # Raised OUTSIDE the except block so __context__ stays None.
                raise ValueError(f"invalid regex: {compile_error}") from None
            if compiled.groups != 1:
                raise ValueError("text pattern must have exactly one capture group")
        return patterns

    @model_validator(mode="after")
    def _require_pattern_or_key(self) -> "IdentifierType":
        if not self.text_patterns and not self.json_keys:
            raise ValueError("identifier type requires a text pattern or a json key")
        return self


class PatternEvaluatorSpec(Model):
    """Spec for the deterministic regex / JSON-key evaluator."""

    kind: Literal["pattern"]
    identifier_types: tuple[IdentifierType, ...] = Field(min_length=1)


class PromptEvaluatorSpec(Model):
    """Spec for the LLM-prompt evaluator."""

    kind: Literal["prompt"]
    system: str  # must contain "{{nonce}}"
    user: str  # must contain "{{content}}" and "{{nonce}}"
    categories: tuple[PhiCategory, ...] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    max_chars_per_call: int = Field(default=12000, ge=500, le=200000)

    @field_validator("system")
    @classmethod
    def _system_requires_nonce(cls, system: str) -> str:
        if "{{nonce}}" not in system:
            raise ValueError("system must contain '{{nonce}}'")
        return system

    @field_validator("user")
    @classmethod
    def _user_requires_content_and_nonce(cls, user: str) -> str:
        if "{{content}}" not in user:
            raise ValueError("user must contain '{{content}}'")
        if "{{nonce}}" not in user:
            raise ValueError("user must contain '{{nonce}}'")
        return user


class DeidEvaluatorSpec(Model):
    """Spec for the de-identification evaluator."""

    kind: Literal["deid"]
    # () = keep every category the provider reports
    categories: tuple[PhiCategory, ...] = ()


class ClassifierEvaluatorSpec(Model):
    """Spec for a text-classifier evaluator, for example a small language model (SLM)."""

    kind: Literal["classifier"]
    # Deployer-scoped name of the classifier; the ClassifierProvider maps it to a model or service.
    classifier: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=128)
    max_chars_per_call: int = Field(default=12000, ge=500, le=200000)


class BuiltinEvaluatorSpec(Model):
    """Spec for an HMCP builtin evaluator (context consistency or placeholder)."""

    kind: Literal["builtin"]
    name: Literal["patient_context_consistency", "redaction_placeholder"]
    identifier_types: tuple[IdentifierType, ...] = ()


CascadeStep = Annotated[
    PatternEvaluatorSpec | PromptEvaluatorSpec | DeidEvaluatorSpec,
    Field(discriminator="kind"),
]


class CascadeEvaluatorSpec(Model):
    """Spec that runs its steps in order and unions their findings."""

    kind: Literal["cascade"]
    steps: tuple[CascadeStep, ...] = Field(min_length=2)


EvaluatorSpec = Annotated[
    PatternEvaluatorSpec
    | PromptEvaluatorSpec
    | DeidEvaluatorSpec
    | ClassifierEvaluatorSpec
    | BuiltinEvaluatorSpec
    | CascadeEvaluatorSpec,
    Field(discriminator="kind"),
]


class AppliesTo(Model):
    """Which patient scopes and tools a guardrail applies to (None = all)."""

    # None = every scope / every tool
    patient_scopes: tuple[ToolPatientScope, ...] | None = None
    tools: tuple[str, ...] | None = None


class ActionSpec(Model):
    """What to do on a finding, and the placeholder used when redacting."""

    on_finding: Literal["block", "redact", "pseudonymize", "annotate"]
    placeholder: str = "[REDACTED:{category}]"

    @field_validator("placeholder")
    @classmethod
    def _validate_placeholder(cls, placeholder: str) -> str:
        if "{category}" not in placeholder:
            raise ValueError("placeholder must contain '{category}'")
        remainder = placeholder.replace("{category}", "")
        if "{" in remainder or "}" in remainder:
            raise ValueError("placeholder may only contain the {category} field")
        return placeholder


class GuardrailDefinition(Model):
    """A portable, declarative guardrail definition."""

    id: str = Field(pattern=r"^[a-z0-9]+(\.[a-z0-9-]+)+/[a-z0-9][a-z0-9-]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str
    description: str = ""
    stages: tuple[GuardrailStage, ...] = Field(min_length=1)
    applies_to: AppliesTo = AppliesTo()
    evaluator: EvaluatorSpec
    action: ActionSpec
    failure_mode: Literal["closed", "open"] = "closed"
    timeout_ms: int = Field(default=5000, ge=100, le=60000)
    data_handling: dict[str, str] = Field(default_factory=dict)

    @field_validator("stages")
    @classmethod
    def _no_duplicate_stages(
        cls, stages: tuple[GuardrailStage, ...]
    ) -> tuple[GuardrailStage, ...]:
        if len(set(stages)) != len(stages):
            raise ValueError("duplicate stages")
        return stages

    @model_validator(mode="after")
    def _cross_validate(self) -> "GuardrailDefinition":
        if (
            isinstance(self.evaluator, BuiltinEvaluatorSpec)
            and self.action.on_finding != "block"
        ):
            raise ValueError("builtin evaluators require action.onFinding 'block'")
        if isinstance(self.evaluator, ClassifierEvaluatorSpec) and (
            self.action.on_finding not in ("block", "annotate")
        ):
            raise ValueError(
                "classifier evaluators require action.onFinding 'block' or 'annotate'"
            )
        if self.action.on_finding == "pseudonymize" and "output" not in self.stages:
            raise ValueError("pseudonymize requires the 'output' stage")
        return self

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(by_alias=True, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def major(self) -> int:
        return int(self.version.split(".")[0])

    @property
    def short_name(self) -> str:
        return self.id.split("/", 1)[1]

    def applies(self, scope: ToolPatientScope, tool_name: str) -> bool:
        if (
            self.applies_to.patient_scopes is not None
            and scope not in self.applies_to.patient_scopes
        ):
            return False
        if self.applies_to.tools is not None and tool_name not in self.applies_to.tools:
            return False
        return True

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)

    @classmethod
    def from_json(cls, text: str) -> "GuardrailDefinition":
        return cls.model_validate_json(text)
