"""Scientific artifacts and conservative, evidence-linked verification contracts.

Provenance is a declaration, not an attestation. This module executes nothing;
M0-A producers emit only synthetic/Mock data. A future trusted executor must
supply real artifacts before the structural VERIFIED rules can be meaningful.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from .core import (
    SHA256,
    Fact,
    FactStatus,
    IdentityModel,
    NonEmptyStr,
    SourceKind,
    StrictFrozenModel,
)
from .platforms import PlatformRequirement


class SourceTrust(str, Enum):
    UNKNOWN = "UNKNOWN"
    UNREVIEWED = "UNREVIEWED"
    REVIEWED = "REVIEWED"


class ExecutionIntent(str, Enum):
    PREPARE = "PREPARE"
    RUN = "RUN"
    PREPARE_AND_RUN = "PREPARE_AND_RUN"


class VerificationRequirement(str, Enum):
    NONE = "NONE"
    CONDITION_LINKED_RUNTIME = "CONDITION_LINKED_RUNTIME"


class ExecutionScope(str, Enum):
    COMPILE_ONLY = "COMPILE_ONLY"
    TARGET_RUNTIME = "TARGET_RUNTIME"
    EMULATOR_RUNTIME = "EMULATOR_RUNTIME"
    MOCK = "MOCK"


class TerminationStatus(str, Enum):
    NORMAL = "NORMAL"
    CRASH = "CRASH"
    TIMEOUT = "TIMEOUT"
    NOT_EXECUTED = "NOT_EXECUTED"


class EvidenceKind(str, Enum):
    BUILD_LOG = "BUILD_LOG"
    RUNTIME_LOG = "RUNTIME_LOG"
    SANITIZER_LOG = "SANITIZER_LOG"
    TRACE = "TRACE"
    STDOUT = "STDOUT"
    STDERR = "STDERR"
    INPUT = "INPUT"
    MODEL_RESPONSE = "MODEL_RESPONSE"
    SYNTHETIC = "SYNTHETIC"


class ObservationKind(str, Enum):
    COMPILE = "COMPILE"
    RUN = "RUN"
    CONDITION = "CONDITION"
    SANITIZER_ERROR = "SANITIZER_ERROR"
    CRASH = "CRASH"
    MODEL = "MODEL"
    DIAGNOSTIC = "DIAGNOSTIC"


class VerificationVerdict(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    NOT_CHECKED = "NOT_CHECKED"
    UNKNOWN = "UNKNOWN"


class Provenance(StrictFrozenModel):
    source: SourceKind
    method: NonEmptyStr
    # Opaque scientific platform/profile ID; a driver change is a different scope.
    scope: NonEmptyStr
    source_sha256: SHA256 | None = None
    run_id: NonEmptyStr | None = None
    locator: NonEmptyStr | None = None
    host_path: NonEmptyStr | None = None
    recorded_at: NonEmptyStr | None = None
    model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class VulnerabilitySpec(IdentityModel):
    identity_prefix: ClassVar[str] = "vulnerability"
    name: NonEmptyStr
    affected_component: NonEmptyStr
    source_trust: SourceTrust = SourceTrust.UNKNOWN
    platform_id: NonEmptyStr
    description: NonEmptyStr
    provenance: Provenance
    source_sha256: SHA256 | None = None

    @model_validator(mode="after")
    def _vulnerability_source(self) -> VulnerabilitySpec:
        if self.provenance.scope != self.platform_id:
            raise ValueError("vulnerability provenance must describe its platform profile")
        if self.source_sha256 is not None and self.provenance.source_sha256 not in (
            None,
            self.source_sha256,
        ):
            raise ValueError("vulnerability source hashes conflict")
        return self


class TriggerConstraint(IdentityModel):
    identity_prefix: ClassVar[str] = "condition"
    name: NonEmptyStr
    expression: NonEmptyStr
    knowledge: Fact[bool] = Field(default_factory=Fact[bool])


class TriggerSpec(IdentityModel):
    identity_prefix: ClassVar[str] = "trigger"
    vulnerability_id: NonEmptyStr
    platform_id: NonEmptyStr
    constraints: tuple[TriggerConstraint, ...]

    @model_validator(mode="after")
    def _unique_constraints(self) -> TriggerSpec:
        ids = tuple(item.id for item in self.constraints)
        if len(ids) != len(set(ids)):
            raise ValueError("trigger constraints must have unique content identities")
        return self


class PoCCandidate(IdentityModel):
    identity_prefix: ClassVar[str] = "candidate"
    vulnerability_id: NonEmptyStr
    trigger_id: NonEmptyStr
    platform_id: NonEmptyStr
    language: NonEmptyStr
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    platform_requirement: PlatformRequirement = Field(default_factory=PlatformRequirement)
    payload: NonEmptyStr
    provenance: Provenance
    source_sha256: SHA256 = ""

    @model_validator(mode="before")
    @classmethod
    def _payload_hash(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and isinstance(raw.get("payload"), str):
            values = dict(raw)
            expected = hashlib.sha256(values["payload"].encode("utf-8")).hexdigest()
            if "source_sha256" in values and values["source_sha256"] != expected:
                raise ValueError("source_sha256 conflicts with the actual candidate payload")
            values["source_sha256"] = expected
            return values
        return raw

    @model_validator(mode="after")
    def _provenance_hash(self) -> PoCCandidate:
        if self.provenance.scope != self.platform_id:
            raise ValueError("candidate provenance must describe its platform profile")
        if self.provenance.source_sha256 not in (None, self.source_sha256):
            raise ValueError("candidate provenance source hash conflicts with payload")
        return self


class ExecutionBudget(StrictFrozenModel):
    timeout_seconds: float = Field(gt=0.0)
    max_memory_bytes: int = Field(gt=0)
    max_output_bytes: int = Field(gt=0)
    max_input_bytes: int = Field(ge=0)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _strict_timeout(cls, value: Any) -> Any:
        if type(value) is not float:
            raise ValueError("timeout_seconds requires a float without implicit conversion")
        return value


class ExecutionRequest(IdentityModel):
    identity_prefix: ClassVar[str] = "request"
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    platform_id: NonEmptyStr
    budget: ExecutionBudget
    intent: ExecutionIntent
    verification_requirement: VerificationRequirement


class InputEvent(StrictFrozenModel):
    run_id: NonEmptyStr
    name: NonEmptyStr
    data: bytes


class EvidenceReference(IdentityModel):
    identity_prefix: ClassVar[str] = "evidence"
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    platform_id: NonEmptyStr
    kind: EvidenceKind
    sha256: SHA256
    byte_length: int = Field(ge=0)
    provenance: Provenance
    locator: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _reference_scope(self) -> EvidenceReference:
        if self.provenance.scope != self.platform_id or self.provenance.run_id != self.run_id:
            raise ValueError("evidence provenance must match its platform and run")
        return self


class RawEvidence(StrictFrozenModel):
    reference: EvidenceReference
    content: bytes

    @model_validator(mode="after")
    def _actual_content(self) -> RawEvidence:
        if len(self.content) != self.reference.byte_length:
            raise ValueError("evidence byte_length does not match actual bytes")
        if hashlib.sha256(self.content).hexdigest() != self.reference.sha256:
            raise ValueError("evidence sha256 does not match actual bytes")
        return self


class Observation(IdentityModel):
    identity_prefix: ClassVar[str] = "observation"
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    platform_id: NonEmptyStr
    kind: ObservationKind
    status: FactStatus
    provenance: Provenance
    satisfied: bool | None = None
    condition_id: NonEmptyStr | None = None
    evidence_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _observation_scope(self) -> Observation:
        if self.provenance.scope != self.platform_id or self.provenance.run_id != self.run_id:
            raise ValueError("observation provenance must match its platform and run")
        if self.kind in (ObservationKind.CONDITION, ObservationKind.SANITIZER_ERROR):
            if self.condition_id is None:
                raise ValueError("condition and sanitizer observations require a target condition")
        if self.status in (FactStatus.UNKNOWN, FactStatus.NOT_CHECKED, FactStatus.UNSUPPORTED):
            if self.satisfied is not None:
                raise ValueError("unknown observations cannot assert a condition result")
        if self.status is FactStatus.OBSERVED:
            if self.provenance.source in (SourceKind.MODEL, SourceKind.UNKNOWN):
                raise ValueError("objective observations require an explicit non-model source")
            if self.kind in (ObservationKind.CONDITION, ObservationKind.SANITIZER_ERROR):
                if self.satisfied is None:
                    raise ValueError("observed target conditions require an explicit result")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("observation evidence references must be unique")
        return self


class ExecutionRecord(StrictFrozenModel):
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    platform_id: NonEmptyStr
    source: SourceKind
    scope: ExecutionScope
    completed: bool
    termination: TerminationStatus
    exit_code: int | None = None

    @model_validator(mode="after")
    def _execution_semantics(self) -> ExecutionRecord:
        sources = {
            ExecutionScope.TARGET_RUNTIME: SourceKind.REAL_EXECUTION,
            ExecutionScope.EMULATOR_RUNTIME: SourceKind.EMULATOR,
            ExecutionScope.MOCK: SourceKind.MOCK,
        }
        if self.scope in sources and self.source is not sources[self.scope]:
            raise ValueError("execution scope conflicts with its declared source")
        if self.termination is TerminationStatus.NOT_EXECUTED:
            if self.completed or self.exit_code is not None:
                raise ValueError("not-executed records cannot claim completion or an exit code")
        if self.termination is TerminationStatus.NORMAL:
            if not self.completed or self.exit_code is None:
                raise ValueError("normal completion requires an actual recorded exit code")
        return self


class VerificationResult(IdentityModel):
    identity_prefix: ClassVar[str] = "verification"
    candidate: PoCCandidate
    trigger: TriggerSpec
    execution: ExecutionRecord
    evidence: tuple[RawEvidence, ...]
    observations: tuple[Observation, ...]
    verdict: VerificationVerdict
    reason: NonEmptyStr
    scope: ExecutionScope
    unmet_conditions: tuple[NonEmptyStr, ...] = ()
    limitations: tuple[NonEmptyStr, ...] = ()

    def identity_payload(self) -> dict[str, Any]:
        # Derived explanations do not change the underlying scientific conclusion ID.
        payload = super().identity_payload()
        payload.pop("unmet_conditions", None)
        payload.pop("limitations", None)
        return payload

    @model_validator(mode="after")
    def _supported_conclusion(self) -> VerificationResult:
        _validate_relationships(
            self.candidate, self.trigger, self.execution, self.evidence, self.observations
        )
        if self.scope is not self.execution.scope:
            raise ValueError("verification cannot extend beyond the recorded execution scope")
        condition_ids = {item.id for item in self.trigger.constraints}
        if not set(self.unmet_conditions).issubset(condition_ids):
            raise ValueError("unmet conditions must belong to the current trigger")
        if len(self.unmet_conditions) != len(set(self.unmet_conditions)):
            raise ValueError("unmet condition references must be unique")
        if self.verdict is VerificationVerdict.VERIFIED and self.unmet_conditions:
            raise ValueError("VERIFIED cannot claim unmet target conditions")
        if self.verdict is VerificationVerdict.NOT_CHECKED:
            return self
        expected, _ = _assess(self.trigger, self.execution, self.evidence, self.observations)
        if expected is VerificationVerdict.UNKNOWN and self.verdict is not expected:
            raise ValueError("unknown conditions or evidence must propagate an UNKNOWN conclusion")
        if self.verdict is VerificationVerdict.VERIFIED and expected is not self.verdict:
            raise ValueError("VERIFIED requires objective condition-linked runtime evidence")
        return self


def _validate_relationships(
    candidate: PoCCandidate,
    trigger: TriggerSpec,
    execution: ExecutionRecord,
    evidence: tuple[RawEvidence, ...],
    observations: tuple[Observation, ...],
) -> None:
    if candidate.trigger_id != trigger.id or candidate.vulnerability_id != trigger.vulnerability_id:
        raise ValueError("candidate and trigger identities do not match")
    if (
        candidate.platform_id != trigger.platform_id
        or candidate.platform_id != execution.platform_id
    ):
        raise ValueError("platform profiles differ; no cross-platform inference is allowed")
    if candidate.id != execution.candidate_id:
        raise ValueError("execution belongs to a different candidate")
    references = {item.reference.id: item.reference for item in evidence}
    if len(references) != len(evidence):
        raise ValueError("raw evidence identities must be unique")
    condition_ids = {item.id for item in trigger.constraints}
    artifacts: tuple[EvidenceReference | Observation, ...] = (
        *tuple(raw.reference for raw in evidence),
        *observations,
    )
    for item in artifacts:
        if (item.run_id, item.candidate_id, item.platform_id) != (
            execution.run_id,
            candidate.id,
            candidate.platform_id,
        ):
            raise ValueError("evidence or observation belongs to a different run or target")
        if item.provenance.source_sha256 not in (None, candidate.source_sha256):
            raise ValueError("evidence or observation source hash conflicts with the candidate")
    for raw in evidence:
        if raw.reference.provenance.source is not execution.source:
            raise ValueError("raw evidence source conflicts with the recorded executor")
    for observation in observations:
        if observation.condition_id is not None and observation.condition_id not in condition_ids:
            raise ValueError("observation references an unrelated trigger condition")
        if not set(observation.evidence_ids).issubset(references):
            raise ValueError("observation references unavailable raw evidence")


def _assess(
    trigger: TriggerSpec,
    execution: ExecutionRecord,
    evidence: tuple[RawEvidence, ...],
    observations: tuple[Observation, ...],
) -> tuple[VerificationVerdict, str]:
    unknown = (FactStatus.UNKNOWN, FactStatus.NOT_CHECKED, FactStatus.UNSUPPORTED)
    if not trigger.constraints or any(
        item.knowledge.status in unknown for item in trigger.constraints
    ):
        return VerificationVerdict.UNKNOWN, "Target trigger conditions are missing or unknown."
    if any(item.status in unknown for item in observations if item.condition_id is not None):
        return VerificationVerdict.UNKNOWN, "A target condition is unknown or unchecked."
    if execution.source is SourceKind.UNKNOWN:
        return VerificationVerdict.UNKNOWN, "Executor provenance is unknown."
    if execution.source in (
        SourceKind.MOCK,
        SourceKind.MODEL,
        SourceKind.SYNTHETIC,
        SourceKind.STATIC_ANALYSIS,
        SourceKind.DOCUMENTATION,
        SourceKind.USER_REQUIREMENT,
    ):
        return (
            VerificationVerdict.NOT_VERIFIED,
            "Source cannot establish objective runtime execution.",
        )
    if execution.scope is ExecutionScope.COMPILE_ONLY or not execution.completed:
        return (
            VerificationVerdict.NOT_VERIFIED,
            "A build or incomplete run does not prove triggering.",
        )
    if execution.termination is not TerminationStatus.NORMAL:
        return VerificationVerdict.NOT_VERIFIED, "Crash or timeout does not prove a target trigger."
    if not evidence or any(raw.reference.provenance.source_sha256 is None for raw in evidence):
        return VerificationVerdict.UNKNOWN, "Raw evidence or candidate provenance is missing."
    references = {raw.reference.id: raw.reference for raw in evidence}
    runtime_kinds = (
        EvidenceKind.RUNTIME_LOG,
        EvidenceKind.SANITIZER_LOG,
        EvidenceKind.TRACE,
        EvidenceKind.STDOUT,
        EvidenceKind.STDERR,
    )
    for condition in trigger.constraints:
        if condition.knowledge.value is not True:
            return VerificationVerdict.NOT_VERIFIED, "A required condition is not established."
        related = [item for item in observations if item.condition_id == condition.id]
        valid = [
            item
            for item in related
            if item.kind in (ObservationKind.CONDITION, ObservationKind.SANITIZER_ERROR)
            and item.status is FactStatus.OBSERVED
            and item.satisfied is True
            and item.provenance.source is execution.source
            and item.provenance.source_sha256 is not None
            and item.evidence_ids
            and all(references[key].kind in runtime_kinds for key in item.evidence_ids)
        ]
        if not valid:
            return (
                VerificationVerdict.NOT_VERIFIED,
                "A condition lacks objective linked runtime proof.",
            )
        if any(item.status is FactStatus.OBSERVED and item.satisfied is False for item in related):
            return VerificationVerdict.UNKNOWN, "Objective condition observations conflict."
    return (
        VerificationVerdict.VERIFIED,
        "All conditions have objective proof within the recorded scope.",
    )


def _unmet_conditions(
    trigger: TriggerSpec,
    execution: ExecutionRecord,
    evidence: tuple[RawEvidence, ...],
    observations: tuple[Observation, ...],
) -> tuple[str, ...]:
    """Explain lack of objective support, without changing the verdict evaluator."""
    if (
        execution.source not in (SourceKind.REAL_EXECUTION, SourceKind.EMULATOR)
        or execution.scope is ExecutionScope.COMPILE_ONLY
        or not execution.completed
        or execution.termination is not TerminationStatus.NORMAL
        or not evidence
        or any(raw.reference.provenance.source_sha256 is None for raw in evidence)
    ):
        return tuple(condition.id for condition in trigger.constraints)
    references = {raw.reference.id: raw.reference for raw in evidence}
    runtime_kinds = (
        EvidenceKind.RUNTIME_LOG,
        EvidenceKind.SANITIZER_LOG,
        EvidenceKind.TRACE,
        EvidenceKind.STDOUT,
        EvidenceKind.STDERR,
    )
    unmet = []
    for condition in trigger.constraints:
        related = tuple(item for item in observations if item.condition_id == condition.id)
        known = condition.knowledge.status in (FactStatus.OBSERVED, FactStatus.INFERRED)
        contradicted = any(
            item.status in (FactStatus.UNKNOWN, FactStatus.NOT_CHECKED, FactStatus.UNSUPPORTED)
            or (item.status is FactStatus.OBSERVED and item.satisfied is False)
            for item in related
        )
        supported = any(
            item.kind in (ObservationKind.CONDITION, ObservationKind.SANITIZER_ERROR)
            and item.status is FactStatus.OBSERVED
            and item.satisfied is True
            and item.provenance.source is execution.source
            and item.provenance.source_sha256 is not None
            and item.evidence_ids
            and all(references[key].kind in runtime_kinds for key in item.evidence_ids)
            for item in related
        )
        if not known or condition.knowledge.value is not True or contradicted or not supported:
            unmet.append(condition.id)
    return tuple(unmet)


def _limitations(execution: ExecutionRecord) -> tuple[str, ...]:
    items = [
        "Structural checks validate declared provenance; they do not attest real execution.",
        f"Support is limited to {execution.scope.value} on profile {execution.platform_id}.",
    ]
    if execution.source is SourceKind.MOCK:
        items.append(
            "Mock output does not prove physical hardware or real vulnerability triggering."
        )
    elif execution.scope is ExecutionScope.EMULATOR_RUNTIME:
        items.append("Emulator evidence supports its modeled environment, not physical hardware.")
    if execution.scope is ExecutionScope.COMPILE_ONLY:
        items.append("Compilation does not establish runtime triggering.")
    if execution.termination in (TerminationStatus.CRASH, TerminationStatus.TIMEOUT):
        items.append(
            "A crash or timeout alone does not prove that target conditions were satisfied."
        )
    return tuple(items)


def verify(
    candidate: PoCCandidate,
    trigger: TriggerSpec,
    execution: ExecutionRecord,
    evidence: tuple[RawEvidence, ...],
    observations: tuple[Observation, ...],
) -> VerificationResult:
    """Conservative contract evaluator; never runs code or upgrades Mock/model evidence."""
    _validate_relationships(candidate, trigger, execution, evidence, observations)
    verdict, reason = _assess(trigger, execution, evidence, observations)
    return VerificationResult(
        candidate=candidate,
        trigger=trigger,
        execution=execution,
        evidence=evidence,
        observations=observations,
        verdict=verdict,
        reason=reason,
        scope=execution.scope,
        unmet_conditions=_unmet_conditions(trigger, execution, evidence, observations),
        limitations=_limitations(execution),
    )
