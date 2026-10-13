"""Small adapter contract; preparation and run instances have separate identities."""

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import model_validator

from .core import FactStatus, StrictFrozenModel
from .domain import (
    EvidenceKind,
    ExecutionBudget,
    ExecutionRecord,
    ExecutionRequest,
    InputEvent,
    PoCCandidate,
    RawEvidence,
    TriggerSpec,
    VerificationResult,
)
from .platforms import CompatibilityDecision, Identifier, PlatformCapability, PlatformRequirement


class AdapterContractError(ValueError):
    """An invalid identity, ordering, budget, or scope was explicitly rejected."""


class PreparationStatus(str, Enum):
    READY = "READY"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class PreparedRun(StrictFrozenModel):
    preparation_id: Identifier
    content_id: Identifier
    candidate_id: Identifier
    platform_id: Identifier
    requirement: PlatformRequirement
    budget: ExecutionBudget


class PreparationResult(StrictFrozenModel):
    status: PreparationStatus
    decision: CompatibilityDecision
    prepared: PreparedRun | None = None
    reason: str

    @model_validator(mode="after")
    def ready_has_compatible_preparation(self) -> "PreparationResult":
        if self.status is PreparationStatus.READY:
            if self.prepared is None or not self.decision.can_execute:
                raise ValueError("READY requires a compatible prepared artifact.")
            if self.decision.platform_id != self.prepared.platform_id:
                raise ValueError("READY decision and preparation must bind the same platform.")
        elif self.prepared is not None:
            raise ValueError(
                "A rejected, failed, or unknown preparation has no executable artifact."
            )
        return self


class InputReceipt(StrictFrozenModel):
    status: FactStatus
    accepted: bool
    evidence: RawEvidence | None = None
    reason: str

    @model_validator(mode="after")
    def accepted_input_has_evidence(self) -> "InputReceipt":
        if self.accepted:
            if self.status is not FactStatus.OBSERVED or self.evidence is None:
                raise ValueError("Accepted input requires observed receipt bytes.")
            if self.evidence.reference.kind is not EvidenceKind.INPUT:
                raise ValueError("Accepted input receipt requires INPUT evidence.")
        elif self.evidence is not None:
            raise ValueError("Rejected input cannot claim accepted-input evidence.")
        return self


@runtime_checkable
class PlatformAdapter(Protocol):
    """No method implies hardware, model access, or a debugger backend.

    Input is injected before execution and is bound to its ExecutionRequest.
    Implementations must reject cross-candidate/platform/run/preparation reuse.
    """

    def query_capability(self) -> PlatformCapability: ...

    def negotiate(self, requirement: PlatformRequirement) -> CompatibilityDecision: ...

    def prepare(
        self, candidate: PoCCandidate, requirement: PlatformRequirement, budget: ExecutionBudget
    ) -> PreparationResult: ...

    def inject_input(
        self, prepared: PreparedRun, request: ExecutionRequest, event: InputEvent
    ) -> InputReceipt: ...

    def execute(self, prepared: PreparedRun, request: ExecutionRequest) -> ExecutionRecord: ...

    def collect_raw_evidence(self, execution: ExecutionRecord) -> tuple[RawEvidence, ...]: ...

    def verify(
        self, candidate: PoCCandidate, trigger: TriggerSpec, execution: ExecutionRecord
    ) -> VerificationResult: ...
