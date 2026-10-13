"""Strict immutable values, explicit knowledge, and deterministic content identity."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
T = TypeVar("T")


class SourceKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    MODEL = "MODEL"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    DOCUMENTATION = "DOCUMENTATION"
    USER_REQUIREMENT = "USER_REQUIREMENT"
    SYNTHETIC = "SYNTHETIC"
    MOCK = "MOCK"
    REAL_EXECUTION = "REAL_EXECUTION"
    EMULATOR = "EMULATOR"


class FactStatus(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    NOT_CHECKED = "NOT_CHECKED"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


IDENTITY_EXCLUSIONS = frozenset(
    {
        "id",
        "locator",
        "host_path",
        "hostpath",
        "run_id",
        "timestamp",
        "created_at",
        "updated_at",
        "recorded_at",
        "observed_at",
        "collected_at",
        "started_at",
        "ended_at",
        "model_confidence",
    }
)


def canonical_json(value: Any) -> str:
    """JSON with stable ordering, no NaN, and no implementation-specific whitespace."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _scientific_content(value: Any, exclude_keys: frozenset[str]) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            key: _scientific_content(item, exclude_keys)
            for key, item in value.items()
            if key not in exclude_keys
        }
    if isinstance(value, (tuple, list)):
        return [_scientific_content(item, exclude_keys) for item in value]
    return value


def stable_id(
    prefix: str, payload: Any, *, exclude_keys: frozenset[str] = IDENTITY_EXCLUSIONS
) -> str:
    """Scientific content ID; run identity and nested locators are deliberately separate."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", prefix):
        raise ValueError("identity prefix must be a short lowercase identifier")
    encoded = canonical_json(_scientific_content(payload, exclude_keys)).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _require_immutable(value: Any) -> None:
    if isinstance(value, (dict, list, set, bytearray)):
        raise ValueError("mutable containers are prohibited; use tuples and frozen models")
    if isinstance(value, tuple):
        for item in value:
            _require_immutable(item)
    elif isinstance(value, BaseModel):
        if not value.model_config.get("frozen"):
            raise ValueError("nested models must also be frozen")
        for field in type(value).model_fields:
            _require_immutable(getattr(value, field))


class StrictFrozenModel(BaseModel):
    """Use model_validate_json for wire JSON; Python construction remains strict."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )
    schema_version: Literal["1.0"] = "1.0"

    @model_validator(mode="after")
    def _immutable_fields(self) -> StrictFrozenModel:
        for field in type(self).model_fields:
            _require_immutable(getattr(self, field))
        return self

    def to_json(self) -> str:
        return canonical_json(self)


class IdentityModel(StrictFrozenModel):
    id: str = ""
    identity_prefix: ClassVar[str] = "artifact"

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"id"})

    @model_validator(mode="after")
    def _validate_content_id(self) -> IdentityModel:
        expected = stable_id(self.identity_prefix, self.identity_payload())
        if self.id and self.id != expected:
            raise ValueError("provided id conflicts with scientific content")
        object.__setattr__(self, "id", expected)
        return self


class Fact(StrictFrozenModel, Generic[T]):
    """Unknown is explicit: a missing fact never becomes False, zero, or a capability."""

    value: T | None = None
    status: FactStatus = FactStatus.UNKNOWN
    source: SourceKind = SourceKind.UNKNOWN
    provenance: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _consistent_knowledge(self) -> Fact[T]:
        if self.status in (FactStatus.UNKNOWN, FactStatus.NOT_CHECKED, FactStatus.UNSUPPORTED):
            if self.value is not None:
                raise ValueError("unknown, unchecked, or unsupported facts cannot carry a value")
        elif self.value is None or self.source is SourceKind.UNKNOWN or self.provenance is None:
            raise ValueError("known facts require a value, source, and provenance")
        if self.status is FactStatus.OBSERVED and self.source is SourceKind.MODEL:
            raise ValueError("model output is an inference, not an objective observation")
        return self
