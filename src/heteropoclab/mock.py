"""Deterministic in-memory adapter; it never evaluates candidate program text.

Budgets bound accounted candidate/input/evidence bytes and a fixed logical step,
not the Python interpreter's resident memory. max_output_bytes bounds simulated
execution output; input receipts are separately bounded by max_input_bytes.
No subprocess, network, filesystem,
or hardware operation is used by this adapter.
"""

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from uuid import uuid4

from .core import Fact, FactStatus, SourceKind, stable_id
from .domain import (
    EvidenceKind,
    EvidenceReference,
    ExecutionBudget,
    ExecutionIntent,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionScope,
    InputEvent,
    Observation,
    ObservationKind,
    PoCCandidate,
    Provenance,
    RawEvidence,
    TerminationStatus,
    TriggerSpec,
    VerificationRequirement,
    VerificationResult,
    VerificationVerdict,
)
from .domain import (
    verify as verify_result,
)
from .platforms import (
    CompatibilityDecision,
    CompatibilityIssue,
    CompatibilityStatus,
    ExecutionMode,
    PlatformCapability,
    PlatformRequirement,
    negotiate,
)
from .protocol import (
    AdapterContractError,
    InputReceipt,
    PreparationResult,
    PreparationStatus,
    PreparedRun,
)


class MockScenario(str, Enum):
    NORMAL = "NORMAL"
    PREPARATION_FAILED = "PREPARATION_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    UNKNOWN = "UNKNOWN"


def default_mock_capability() -> PlatformCapability:
    """These are declared mock settings, not observations of a physical target."""
    return PlatformCapability(
        label="mock.default.v1",
        category=Fact[str](
            value="MOCK",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        isa=Fact[str](
            value="mock.isa",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        isa_profile=Fact[str](
            value="mock.profile",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        os=Fact[str](
            value="mock.os",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        binary_format=Fact[str](
            value="mock.text",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        endian=Fact[str](
            value="little",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        bit_width=Fact[int](
            value=32,
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        abi=Fact[str](
            value="mock.abi",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        toolchain=Fact[str](
            value="mock.no-compiler.v1",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        runtime=Fact[str](
            value="mock.memory.v1",
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        execution_mode=Fact[ExecutionMode](
            value=ExecutionMode.MOCK,
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        debug_capabilities=Fact[tuple[str, ...]](
            value=(),
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        observation_capabilities=Fact[tuple[str, ...]](
            value=("mock.observation",),
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        input_interfaces=Fact[tuple[str, ...]](
            value=("mock.input",),
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
        evidence_capabilities=Fact[tuple[str, ...]](
            value=("mock.raw",),
            status=FactStatus.OBSERVED,
            source=SourceKind.MOCK,
            provenance="declared mock configuration",
        ),
    )


@dataclass
class _Preparation:
    artifact: PreparedRun
    candidate: PoCCandidate


@dataclass
class _Run:
    preparation: _Preparation
    request: ExecutionRequest
    inputs: list[RawEvidence] = field(default_factory=list)
    execution: ExecutionRecord | None = None
    evidence: tuple[RawEvidence, ...] = ()
    observations: tuple[Observation, ...] = ()


_PREPARATION_DIMENSIONS = (
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
    "observation_capabilities",
    "evidence_capabilities",
)


class MockAdapter:
    def __init__(
        self,
        *,
        capability: PlatformCapability | None = None,
        scenario: MockScenario = MockScenario.NORMAL,
    ) -> None:
        if not isinstance(scenario, MockScenario):
            raise AdapterContractError("scenario must be an explicit MockScenario.")
        self._capability = (
            PlatformCapability.model_validate(capability.model_dump())
            if capability is not None
            else default_mock_capability()
        )
        mode = self._capability.execution_mode
        if mode.value is not ExecutionMode.MOCK or mode.status is not FactStatus.OBSERVED:
            raise AdapterContractError("An in-memory adapter must declare observed MOCK execution.")
        self._scenario = scenario
        self._preparations: dict[str, _Preparation] = {}
        self._runs: dict[str, _Run] = {}

    def query_capability(self) -> PlatformCapability:
        return self._capability

    def negotiate(self, requirement: PlatformRequirement) -> CompatibilityDecision:
        return negotiate(self._capability, requirement)

    def _effective_requirement(self, requirement: PlatformRequirement) -> PlatformRequirement:
        values = {name: getattr(requirement, name) for name in PlatformRequirement.model_fields}
        if values["platform_id"] is None:
            values["platform_id"] = self._capability.id
        for name in _PREPARATION_DIMENSIONS:
            if values[name] is None:
                values[name] = getattr(self._capability, name)
        return PlatformRequirement(**values)

    def _output(self, candidate: PoCCandidate) -> bytes:
        return (
            f"MOCK scenario={self._scenario.value}; candidate_source_sha256="
            f"{candidate.source_sha256}; no candidate program was executed.\n"
        ).encode("ascii")

    def prepare(
        self, candidate: PoCCandidate, requirement: PlatformRequirement, budget: ExecutionBudget
    ) -> PreparationResult:
        candidate = PoCCandidate.model_validate(candidate.model_dump())
        budget = ExecutionBudget.model_validate(budget.model_dump())
        requirement = PlatformRequirement.model_validate(requirement.model_dump())
        effective = self._effective_requirement(requirement)
        decision = self.negotiate(effective)
        candidate_decision = self.negotiate(candidate.platform_requirement)
        if not candidate_decision.can_execute:
            issues = (*decision.issues, *candidate_decision.issues)
            decision_status = (
                CompatibilityStatus.UNSUPPORTED
                if any(item.status is CompatibilityStatus.UNSUPPORTED for item in issues)
                else CompatibilityStatus.UNKNOWN
            )
            decision = CompatibilityDecision(
                platform_id=self._capability.id,
                status=decision_status,
                issues=issues,
            )
        if candidate.platform_id != self._capability.id:
            decision = CompatibilityDecision(
                platform_id=self._capability.id,
                status=CompatibilityStatus.UNSUPPORTED,
                issues=(
                    CompatibilityIssue(
                        dimension="platform_id",
                        status=CompatibilityStatus.UNSUPPORTED,
                        reason="Candidate platform identity differs from this adapter.",
                    ),
                ),
            )
        if not decision.can_execute:
            status = (
                PreparationStatus.UNKNOWN
                if decision.status is CompatibilityStatus.UNKNOWN
                else PreparationStatus.REJECTED
            )
            return PreparationResult(
                status=status,
                decision=decision,
                reason="Required platform facts are incompatible or not established.",
            )
        output_bytes = len(self._output(candidate))
        memory_bytes = len(candidate.payload.encode("utf-8")) + output_bytes
        if (
            budget.timeout_seconds < 0.001
            or budget.max_memory_bytes < memory_bytes
            or budget.max_output_bytes < output_bytes
        ):
            return PreparationResult(
                status=PreparationStatus.FAILED,
                decision=decision,
                reason="Budget is insufficient for this bounded in-memory simulation.",
            )
        if not self._capability.evidence_capabilities.value:
            return PreparationResult(
                status=PreparationStatus.FAILED,
                decision=decision,
                reason="No raw-evidence capability is declared.",
            )
        if self._scenario is MockScenario.PREPARATION_FAILED:
            return PreparationResult(
                status=PreparationStatus.FAILED,
                decision=decision,
                reason="Explicitly configured mock preparation failure; no compilation occurred.",
            )
        content_id = stable_id(
            "prepared",
            {
                "candidate_id": candidate.id,
                "capability": self._capability.model_dump(mode="json"),
                "requirement": effective.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            },
        )
        prepared = PreparedRun(
            preparation_id="mock-preparation-" + uuid4().hex,
            content_id=content_id,
            candidate_id=candidate.id,
            platform_id=candidate.platform_id,
            requirement=effective,
            budget=budget,
        )
        self._preparations[prepared.preparation_id] = _Preparation(prepared, candidate)
        return PreparationResult(
            status=PreparationStatus.READY,
            decision=decision,
            prepared=prepared,
            reason="Prepared a mock artifact in memory; no compiler or target was invoked.",
        )

    def _bind(self, prepared: PreparedRun, request: ExecutionRequest) -> _Run:
        request = ExecutionRequest.model_validate(request.model_dump())
        if request.intent is ExecutionIntent.PREPARE:
            raise AdapterContractError("A PREPARE-only request cannot inject or execute a run.")
        registration = self._preparations.get(prepared.preparation_id)
        if registration is None or prepared != registration.artifact:
            raise AdapterContractError("Unknown, foreign, or modified prepared artifact.")
        if (
            request.candidate_id != prepared.candidate_id
            or request.platform_id != prepared.platform_id
        ):
            raise AdapterContractError(
                "Execution request candidate/platform differs from preparation."
            )
        if request.budget != prepared.budget:
            raise AdapterContractError("Execution request substituted the prepared budget.")
        if not self.negotiate(prepared.requirement).can_execute:
            raise AdapterContractError("Prepared platform requirements are no longer compatible.")
        run = self._runs.get(request.run_id)
        if run is None:
            run = _Run(registration, request)
            self._runs[request.run_id] = run
        elif run.request != request or run.preparation.artifact != prepared:
            raise AdapterContractError(
                "Run identity was reused with a different request or preparation."
            )
        return run

    def _raw(self, run: _Run, content: bytes, kind: EvidenceKind, method: str) -> RawEvidence:
        candidate = run.preparation.candidate
        provenance = Provenance(
            source=SourceKind.MOCK,
            method=method,
            scope=candidate.platform_id,
            source_sha256=candidate.source_sha256,
            run_id=run.request.run_id,
        )
        reference = EvidenceReference(
            run_id=run.request.run_id,
            candidate_id=candidate.id,
            platform_id=candidate.platform_id,
            kind=kind,
            sha256=sha256(content).hexdigest(),
            byte_length=len(content),
            provenance=provenance,
        )
        return RawEvidence(reference=reference, content=content)

    def inject_input(
        self, prepared: PreparedRun, request: ExecutionRequest, event: InputEvent
    ) -> InputReceipt:
        event = InputEvent.model_validate(event.model_dump())
        if not event.data:
            raise AdapterContractError(
                "Empty input events are unsupported by this bounded Mock adapter."
            )
        run = self._bind(prepared, request)
        if run.execution is not None:
            raise AdapterContractError("Input cannot be injected after this run has executed.")
        if event.run_id != request.run_id:
            raise AdapterContractError("Input event belongs to a different run instance.")
        interfaces = self._capability.input_interfaces
        if interfaces.status in {FactStatus.UNKNOWN, FactStatus.NOT_CHECKED}:
            return InputReceipt(
                status=FactStatus.UNKNOWN,
                accepted=False,
                reason="Input-interface capability has not been established.",
            )
        if interfaces.status is FactStatus.UNSUPPORTED or event.name not in (
            interfaces.value or ()
        ):
            return InputReceipt(
                status=FactStatus.UNSUPPORTED,
                accepted=False,
                reason="Exact input-interface identifier is unavailable; no translation occurred.",
            )
        input_bytes = sum(len(item.content) for item in run.inputs) + len(event.data)
        candidate = run.preparation.candidate
        if input_bytes > request.budget.max_input_bytes:
            raise AdapterContractError("Input exceeds the run's cumulative input-byte budget.")
        if (
            len(candidate.payload.encode("utf-8")) + input_bytes + len(self._output(candidate))
            > request.budget.max_memory_bytes
        ):
            raise AdapterContractError("Input exceeds the accounted mock memory-byte budget.")
        evidence = self._raw(
            run,
            event.data,
            EvidenceKind.INPUT,
            f"mock input receipt {len(run.inputs)} interface={event.name}",
        )
        run.inputs.append(evidence)
        return InputReceipt(
            status=FactStatus.OBSERVED,
            accepted=True,
            evidence=evidence,
            reason="Bytes were stored only in this in-memory mock run.",
        )

    def execute(self, prepared: PreparedRun, request: ExecutionRequest) -> ExecutionRecord:
        run = self._bind(prepared, request)
        if run.execution is not None:
            raise AdapterContractError("A run instance cannot be executed twice.")
        unknown = self._scenario is MockScenario.UNKNOWN
        failed = self._scenario is MockScenario.EXECUTION_FAILED
        execution = ExecutionRecord(
            run_id=request.run_id,
            candidate_id=request.candidate_id,
            platform_id=request.platform_id,
            source=SourceKind.MOCK,
            scope=ExecutionScope.MOCK,
            completed=not unknown,
            exit_code=None if unknown else (1 if failed else 0),
            termination=(
                TerminationStatus.NOT_EXECUTED
                if unknown
                else TerminationStatus.ERROR_EXIT
                if failed
                else TerminationStatus.NORMAL
            ),
        )
        raw = self._raw(
            run,
            self._output(run.preparation.candidate),
            EvidenceKind.RUNTIME_LOG,
            "deterministic in-memory mock outcome",
        )
        observation = Observation(
            run_id=request.run_id,
            candidate_id=request.candidate_id,
            platform_id=request.platform_id,
            kind=ObservationKind.RUN,
            status=FactStatus.UNKNOWN if unknown else FactStatus.OBSERVED,
            satisfied=None if unknown else not failed,
            evidence_ids=(raw.reference.id,),
            provenance=raw.reference.provenance,
        )
        run.execution = execution
        run.evidence = (*run.inputs, raw)
        run.observations = (observation,)
        return execution

    def _completed_run(self, execution: ExecutionRecord) -> _Run:
        run = self._runs.get(execution.run_id)
        if run is None or run.execution is None or run.execution != execution:
            raise AdapterContractError(
                "Execution record is unknown or differs from this run instance."
            )
        return run

    def collect_raw_evidence(self, execution: ExecutionRecord) -> tuple[RawEvidence, ...]:
        run = self._completed_run(execution)
        candidate = run.preparation.candidate
        for raw in run.evidence:
            RawEvidence.model_validate(raw.model_dump())
            ref = raw.reference
            if (
                ref.run_id != execution.run_id
                or ref.candidate_id != candidate.id
                or ref.platform_id != candidate.platform_id
                or ref.provenance.scope != candidate.platform_id
                or ref.provenance.source is not SourceKind.MOCK
                or ref.provenance.source_sha256 != candidate.source_sha256
            ):
                raise AdapterContractError(
                    "Raw-evidence ownership or source hash differs from the run."
                )
        return run.evidence

    def verify(
        self, candidate: PoCCandidate, trigger: TriggerSpec, execution: ExecutionRecord
    ) -> VerificationResult:
        run = self._completed_run(execution)
        if candidate != run.preparation.candidate:
            raise AdapterContractError(
                "Verification candidate differs from the prepared candidate."
            )
        if run.request.verification_requirement is VerificationRequirement.NONE:
            return VerificationResult(
                candidate=candidate,
                trigger=trigger,
                execution=execution,
                evidence=self.collect_raw_evidence(execution),
                observations=run.observations,
                verdict=VerificationVerdict.NOT_CHECKED,
                reason=(
                    "Verification was not requested by the bound execution request; "
                    "target conditions were not evaluated."
                ),
                scope=execution.scope,
            )
        observations = run.observations
        if self._scenario is MockScenario.UNKNOWN:
            provenance = run.evidence[-1].reference.provenance
            observations = (
                *observations,
                *(
                    Observation(
                        run_id=execution.run_id,
                        candidate_id=candidate.id,
                        platform_id=candidate.platform_id,
                        kind=ObservationKind.CONDITION,
                        status=FactStatus.UNKNOWN,
                        condition_id=condition.id,
                        evidence_ids=(),
                        provenance=provenance,
                    )
                    for condition in trigger.constraints
                ),
            )
        return verify_result(
            candidate, trigger, execution, self.collect_raw_evidence(execution), observations
        )
