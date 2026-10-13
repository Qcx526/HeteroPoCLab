"""Independent platform facts and conservative capability negotiation."""

from enum import Enum
from typing import Annotated, ClassVar

from pydantic import Field

from .core import Fact, IdentityModel, StrictFrozenModel

Identifier = Annotated[str, Field(min_length=1, max_length=256, strict=True, pattern=r"\S")]
BitWidth = Annotated[int, Field(ge=1, le=1024, strict=True)]


class ExecutionMode(str, Enum):
    REAL_HARDWARE = "REAL_HARDWARE"
    SIMULATOR = "SIMULATOR"
    MOCK = "MOCK"


class CompatibilityStatus(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class PlatformCapability(IdentityModel):
    """Every dimension is independent; unknown facts carry no guessed value.

    Interface identifiers are application configuration. No debugger backend,
    bus name, instruction profile, or device-specific ABI is hardcoded here.
    """

    identity_prefix: ClassVar[str] = "platform"
    label: Identifier
    category: Fact[Identifier]
    isa: Fact[Identifier]
    isa_profile: Fact[Identifier]
    os: Fact[Identifier]
    binary_format: Fact[Identifier]
    endian: Fact[Identifier]
    bit_width: Fact[BitWidth]
    abi: Fact[Identifier]
    toolchain: Fact[Identifier]
    runtime: Fact[Identifier]
    execution_mode: Fact[ExecutionMode]
    debug_capabilities: Fact[tuple[Identifier, ...]]
    observation_capabilities: Fact[tuple[Identifier, ...]]
    input_interfaces: Fact[tuple[Identifier, ...]]
    evidence_capabilities: Fact[tuple[Identifier, ...]]


class PlatformRequirement(StrictFrozenModel):
    """None means unconstrained; a requested UNKNOWN fact never means wildcard."""

    platform_id: Identifier | None = None
    category: Fact[Identifier] | None = None
    isa: Fact[Identifier] | None = None
    isa_profile: Fact[Identifier] | None = None
    os: Fact[Identifier] | None = None
    binary_format: Fact[Identifier] | None = None
    endian: Fact[Identifier] | None = None
    bit_width: Fact[BitWidth] | None = None
    abi: Fact[Identifier] | None = None
    toolchain: Fact[Identifier] | None = None
    runtime: Fact[Identifier] | None = None
    execution_mode: Fact[ExecutionMode] | None = None
    debug_capabilities: Fact[tuple[Identifier, ...]] | None = None
    observation_capabilities: Fact[tuple[Identifier, ...]] | None = None
    input_interfaces: Fact[tuple[Identifier, ...]] | None = None
    evidence_capabilities: Fact[tuple[Identifier, ...]] | None = None


class CompatibilityIssue(StrictFrozenModel):
    dimension: Identifier
    status: CompatibilityStatus
    reason: str


class CompatibilityDecision(StrictFrozenModel):
    platform_id: Identifier
    status: CompatibilityStatus
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def can_execute(self) -> bool:
        return self.status is CompatibilityStatus.COMPATIBLE and not self.issues


_SET_DIMENSIONS = frozenset(
    {"debug_capabilities", "observation_capabilities", "input_interfaces", "evidence_capabilities"}
)
_DIMENSIONS = (
    "category",
    "isa",
    "isa_profile",
    "os",
    "binary_format",
    "endian",
    "bit_width",
    "abi",
    "toolchain",
    "runtime",
    "execution_mode",
    "debug_capabilities",
    "observation_capabilities",
    "input_interfaces",
    "evidence_capabilities",
)


def _state(fact: Fact[object]) -> str:
    status = fact.status
    return str(getattr(status, "value", status)).upper()


def negotiate(
    capability: PlatformCapability, requirement: PlatformRequirement
) -> CompatibilityDecision:
    """Match exact scalar dimensions and subset interface requirements.

    A definite mismatch is UNSUPPORTED even if another dimension is unknown.
    This function never translates an ISA, ABI, runtime, or interface identifier.
    """
    capability = PlatformCapability.model_validate(capability.model_dump())
    requirement = PlatformRequirement.model_validate(requirement.model_dump())
    issues: list[CompatibilityIssue] = []
    if requirement.platform_id is not None and requirement.platform_id != capability.id:
        issues.append(
            CompatibilityIssue(
                dimension="platform_id",
                status=CompatibilityStatus.UNSUPPORTED,
                reason="Requested platform identity differs from this adapter.",
            )
        )
    for dimension in _DIMENSIONS:
        requested = getattr(requirement, dimension)
        if requested is None:
            continue
        available = getattr(capability, dimension)
        requested_state = _state(requested)
        available_state = _state(available)
        issue_status: CompatibilityStatus | None = None
        reason = ""
        if requested_state == "UNSUPPORTED":
            issue_status = CompatibilityStatus.UNSUPPORTED
            reason = "The requested fact explicitly states UNSUPPORTED."
        elif requested_state not in {"OBSERVED", "INFERRED"} or requested.value is None:
            issue_status = CompatibilityStatus.UNKNOWN
            reason = "The requested fact is unknown or not checked; it is not a wildcard."
        elif dimension in _SET_DIMENSIONS and not requested.value:
            continue
        elif available_state == "UNSUPPORTED":
            issue_status = CompatibilityStatus.UNSUPPORTED
            reason = "This capability is explicitly unsupported."
        elif available_state not in {"OBSERVED", "INFERRED"} or available.value is None:
            issue_status = CompatibilityStatus.UNKNOWN
            reason = "This required capability has not been established."
        elif dimension in _SET_DIMENSIONS:
            if not set(requested.value).issubset(available.value):
                issue_status = CompatibilityStatus.UNSUPPORTED
                reason = "One or more requested identifiers are unavailable."
        elif requested.value != available.value:
            issue_status = CompatibilityStatus.UNSUPPORTED
            reason = "Exact dimension values differ; no implicit conversion is allowed."
        if issue_status is not None:
            issues.append(
                CompatibilityIssue(dimension=dimension, status=issue_status, reason=reason)
            )
    status = CompatibilityStatus.COMPATIBLE
    if any(issue.status is CompatibilityStatus.UNSUPPORTED for issue in issues):
        status = CompatibilityStatus.UNSUPPORTED
    elif issues:
        status = CompatibilityStatus.UNKNOWN
    return CompatibilityDecision(platform_id=capability.id, status=status, issues=tuple(issues))
