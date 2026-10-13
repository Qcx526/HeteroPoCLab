"""The mock exercises only the adapter contract; no target code is evaluated."""

from dataclasses import dataclass
from hashlib import sha256

import pytest
from conftest import known
from pydantic import ValidationError

from heteropoclab.core import Fact, FactStatus, SourceKind
from heteropoclab.domain import (
    EvidenceKind,
    EvidenceReference,
    ExecutionBudget,
    ExecutionIntent,
    ExecutionRequest,
    ExecutionScope,
    InputEvent,
    PoCCandidate,
    Provenance,
    RawEvidence,
    TerminationStatus,
    TriggerConstraint,
    TriggerSpec,
    VerificationRequirement,
    VerificationVerdict,
)
from heteropoclab.mock import MockAdapter, MockScenario, default_mock_capability
from heteropoclab.platforms import (
    CompatibilityStatus,
    ExecutionMode,
    PlatformCapability,
    PlatformRequirement,
)
from heteropoclab.protocol import (
    AdapterContractError,
    InputReceipt,
    PlatformAdapter,
    PreparationResult,
    PreparationStatus,
    PreparedRun,
)


@dataclass(frozen=True)
class MockContract:
    candidate: PoCCandidate
    trigger: TriggerSpec
    budget: ExecutionBudget


def mock_contract(capability: PlatformCapability) -> MockContract:
    condition = TriggerConstraint(
        name="mock target", expression="mock stimulus received", knowledge=known(True)
    )
    trigger = TriggerSpec(
        vulnerability_id="fixture-vulnerability",
        platform_id=capability.id,
        constraints=(condition,),
    )
    candidate = PoCCandidate(
        vulnerability_id=trigger.vulnerability_id,
        trigger_id=trigger.id,
        platform_id=capability.id,
        language="fixture-text",
        payload="DO NOT EXECUTE: synthetic interface fixture",
        description="mock contract fixture",
        expected_behavior="mock receipt only",
        provenance=Provenance(
            source=SourceKind.SYNTHETIC,
            method="CPU-only mock contract fixture",
            scope=capability.id,
        ),
    )
    budget = ExecutionBudget(
        timeout_seconds=1.0, max_memory_bytes=16384, max_output_bytes=4096, max_input_bytes=512
    )
    return MockContract(candidate, trigger, budget)


def ready(adapter: MockAdapter, contract: MockContract) -> PreparedRun:
    preparation = adapter.prepare(contract.candidate, PlatformRequirement(), contract.budget)
    assert preparation.status is PreparationStatus.READY
    assert preparation.decision.can_execute
    assert preparation.prepared is not None
    return preparation.prepared


def request_for(prepared: PreparedRun, run_id: str = "mock-run-A") -> ExecutionRequest:
    return ExecutionRequest(
        run_id=run_id,
        candidate_id=prepared.candidate_id,
        platform_id=prepared.platform_id,
        budget=prepared.budget,
        intent=ExecutionIntent.RUN,
        verification_requirement=VerificationRequirement.CONDITION_LINKED_RUNTIME,
    )


def test_complete_mock_protocol_preserves_bytes_and_cannot_verify_vulnerability() -> None:
    adapter = MockAdapter()
    assert isinstance(adapter, PlatformAdapter)
    capability = adapter.query_capability()
    assert capability.execution_mode.value is ExecutionMode.MOCK
    assert capability.execution_mode.source is SourceKind.MOCK
    assert adapter.negotiate(PlatformRequirement(platform_id=capability.id)).can_execute
    contract = mock_contract(capability)
    prepared = ready(adapter, contract)
    assert PreparedRun.model_validate_json(prepared.model_dump_json()) == prepared
    request = request_for(prepared)
    content = bytes(range(256))
    receipt = adapter.inject_input(
        prepared, request, InputEvent(run_id=request.run_id, name="mock.input", data=content)
    )
    assert receipt.accepted and receipt.status is FactStatus.OBSERVED
    assert receipt.evidence is not None
    assert receipt.evidence.content == content
    assert receipt.evidence.reference.sha256 == sha256(content).hexdigest()
    assert receipt.evidence.reference.provenance.source is SourceKind.MOCK
    execution = adapter.execute(prepared, request)
    assert execution.completed and execution.exit_code == 0
    assert execution.source is SourceKind.MOCK and execution.scope is ExecutionScope.MOCK
    evidence = adapter.collect_raw_evidence(execution)
    assert evidence[0] == receipt.evidence
    assert b"no candidate program was executed" in evidence[-1].content
    for raw in evidence:
        assert raw.reference.run_id == request.run_id
        assert raw.reference.candidate_id == contract.candidate.id
        assert raw.reference.platform_id == capability.id
        assert raw.reference.provenance.source_sha256 == contract.candidate.source_sha256
    verified = adapter.verify(contract.candidate, contract.trigger, execution)
    assert verified.verdict is VerificationVerdict.NOT_VERIFIED
    assert verified.scope is ExecutionScope.MOCK
    assert type(verified).model_validate_json(verified.model_dump_json()) == verified


def test_mock_output_content_is_deterministic_while_run_and_preparation_are_separate() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    first = ready(adapter, contract)
    second = ready(adapter, contract)
    assert first.content_id == second.content_id
    assert first.preparation_id != second.preparation_id
    run_a = adapter.execute(first, request_for(first, "mock-run-A"))
    run_b = adapter.execute(second, request_for(second, "mock-run-B"))
    raw_a = adapter.collect_raw_evidence(run_a)[-1]
    raw_b = adapter.collect_raw_evidence(run_b)[-1]
    assert raw_a.content == raw_b.content
    assert raw_a.reference.sha256 == raw_b.reference.sha256
    assert raw_a.reference.id == raw_b.reference.id
    assert raw_a.reference.run_id != raw_b.reference.run_id


@pytest.mark.parametrize(
    "scenario,completed,exit_code,verdict",
    [
        (MockScenario.EXECUTION_FAILED, True, 1, VerificationVerdict.NOT_VERIFIED),
        (MockScenario.UNKNOWN, False, None, VerificationVerdict.UNKNOWN),
    ],
)
def test_explicit_mock_execution_failures_and_unknowns(
    scenario: MockScenario,
    completed: bool,
    exit_code: int | None,
    verdict: VerificationVerdict,
) -> None:
    adapter = MockAdapter(scenario=scenario)
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    execution = adapter.execute(prepared, request_for(prepared))
    assert execution.completed is completed
    assert execution.exit_code == exit_code
    if not completed:
        assert execution.termination is TerminationStatus.NOT_EXECUTED
    assert adapter.verify(contract.candidate, contract.trigger, execution).verdict is verdict


def test_preparation_failure_has_no_executable_artifact() -> None:
    adapter = MockAdapter(scenario=MockScenario.PREPARATION_FAILED)
    contract = mock_contract(adapter.query_capability())
    preparation = adapter.prepare(contract.candidate, PlatformRequirement(), contract.budget)
    assert preparation.status is PreparationStatus.FAILED
    assert preparation.prepared is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", 0.0001),
        ("max_memory_bytes", 1),
        ("max_output_bytes", 1),
    ],
)
def test_insufficient_prepare_budget_is_explicit_failure(field: str, value: int | float) -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    data = contract.budget.model_dump()
    data[field] = value
    budget = ExecutionBudget.model_validate(data)
    preparation = adapter.prepare(contract.candidate, PlatformRequirement(), budget)
    assert preparation.status is PreparationStatus.FAILED
    assert preparation.prepared is None


def test_input_budget_is_cumulative_and_preserves_accepted_receipts() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    request = request_for(prepared)
    first = adapter.inject_input(
        prepared, request, InputEvent(run_id=request.run_id, name="mock.input", data=b"a" * 512)
    )
    assert first.accepted
    with pytest.raises(AdapterContractError):
        adapter.inject_input(
            prepared, request, InputEvent(run_id=request.run_id, name="mock.input", data=b"b")
        )
    execution = adapter.execute(prepared, request)
    assert adapter.collect_raw_evidence(execution)[0] == first.evidence


def test_input_memory_budget_is_accounted_separately() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    # Enough for preparation, insufficient after a 512-byte input receipt.
    budget = ExecutionBudget(
        timeout_seconds=1.0, max_memory_bytes=250, max_output_bytes=4096, max_input_bytes=512
    )
    prepared_result = adapter.prepare(contract.candidate, PlatformRequirement(), budget)
    assert prepared_result.status is PreparationStatus.READY
    assert prepared_result.prepared is not None
    prepared = prepared_result.prepared
    with pytest.raises(AdapterContractError):
        adapter.inject_input(
            prepared,
            request_for(prepared),
            InputEvent(run_id="mock-run-A", name="mock.input", data=b"x" * 512),
        )


@pytest.mark.parametrize("interface", ["UART", "MOCK.INPUT", "mock.input.extra"])
def test_unsupported_input_returns_explicit_status(interface: str) -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    receipt = adapter.inject_input(
        prepared, request_for(prepared), InputEvent(run_id="mock-run-A", name=interface, data=b"x")
    )
    assert not receipt.accepted
    assert receipt.status is FactStatus.UNSUPPORTED
    assert receipt.evidence is None


def test_unknown_input_interface_is_not_assumed_supported() -> None:
    capability = default_mock_capability()
    data = capability.model_dump(exclude={"id"})
    data["input_interfaces"] = Fact[tuple[str, ...]](status=FactStatus.NOT_CHECKED)
    adapter = MockAdapter(capability=PlatformCapability.model_validate(data))
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    receipt = adapter.inject_input(
        prepared,
        request_for(prepared),
        InputEvent(run_id="mock-run-A", name="mock.input", data=b"x"),
    )
    assert not receipt.accepted and receipt.status is FactStatus.UNKNOWN


def test_c8000_declared_mock_capability_cannot_prepare_unknown_abi_profile_or_endian(
    platform_fixtures: dict[str, PlatformCapability],
) -> None:
    adapter = MockAdapter(capability=platform_fixtures["c8000"])
    contract = mock_contract(adapter.query_capability())
    preparation = adapter.prepare(contract.candidate, PlatformRequirement(), contract.budget)
    assert preparation.status is PreparationStatus.UNKNOWN
    assert preparation.prepared is None
    assert preparation.decision.status is CompatibilityStatus.UNKNOWN
    assert {issue.dimension for issue in preparation.decision.issues} >= {
        "isa_profile",
        "endian",
        "bit_width",
        "abi",
    }


@pytest.mark.parametrize("platform", ["gpu", "arm", "riscv_baremetal", "riscv_rtos"])
def test_mock_cannot_claim_real_hardware_or_simulator(
    platform_fixtures: dict[str, PlatformCapability],
    platform: str,
) -> None:
    with pytest.raises(AdapterContractError):
        MockAdapter(capability=platform_fixtures[platform])


def test_preparation_rejects_platform_scope_or_requirement_mismatch() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    rejection = adapter.prepare(
        contract.candidate, PlatformRequirement(runtime=known("driver-B")), contract.budget
    )
    assert rejection.status is PreparationStatus.REJECTED
    assert rejection.prepared is None
    candidate_data = contract.candidate.model_dump(exclude={"id"})
    candidate_data["platform_id"] = "different-platform"
    provenance = contract.candidate.provenance.model_dump()
    provenance["scope"] = "different-platform"
    candidate_data["provenance"] = provenance
    candidate = PoCCandidate.model_validate(candidate_data)
    rejection = adapter.prepare(candidate, PlatformRequirement(), contract.budget)
    assert rejection.status is PreparationStatus.REJECTED


@pytest.mark.parametrize(
    "field,value", [("candidate_id", "different-candidate"), ("platform_id", "different-platform")]
)
def test_execution_request_identity_reuse_is_rejected(field: str, value: str) -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    data = request_for(prepared).model_dump(exclude={"id"})
    data[field] = value
    with pytest.raises(AdapterContractError):
        adapter.execute(prepared, ExecutionRequest.model_validate(data))


def test_prepared_requirement_budget_and_foreign_adapter_tokens_are_not_replaceable() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    modified_data = prepared.model_dump()
    modified_data["requirement"] = PlatformRequirement()
    modified = PreparedRun.model_validate(modified_data)
    with pytest.raises(AdapterContractError):
        adapter.execute(modified, request_for(prepared))
    request_data = request_for(prepared).model_dump(exclude={"id"})
    budget_data = contract.budget.model_dump()
    budget_data["max_memory_bytes"] += 1
    request_data["budget"] = ExecutionBudget.model_validate(budget_data)
    with pytest.raises(AdapterContractError):
        adapter.execute(prepared, ExecutionRequest.model_validate(request_data))
    with pytest.raises(AdapterContractError):
        MockAdapter().execute(prepared, request_for(prepared))


def test_run_reuse_duplicate_execution_and_post_execution_input_are_rejected() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    first = ready(adapter, contract)
    second = ready(adapter, contract)
    request = request_for(first)
    adapter.inject_input(
        first, request, InputEvent(run_id=request.run_id, name="mock.input", data=b"x")
    )
    with pytest.raises(AdapterContractError):
        adapter.execute(second, request_for(second, request.run_id))
    execution = adapter.execute(first, request)
    with pytest.raises(AdapterContractError):
        adapter.execute(first, request)
    with pytest.raises(AdapterContractError):
        adapter.inject_input(
            first, request, InputEvent(run_id=request.run_id, name="mock.input", data=b"late")
        )
    execution_data = execution.model_dump()
    execution_data["run_id"] = "other-run"
    with pytest.raises(AdapterContractError):
        adapter.collect_raw_evidence(type(execution).model_validate(execution_data))


def test_input_run_binding_and_prepare_only_intent_are_checked() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    request = request_for(prepared)
    with pytest.raises(AdapterContractError):
        adapter.inject_input(
            prepared, request, InputEvent(run_id="other-run", name="mock.input", data=b"x")
        )
    data = request.model_dump(exclude={"id"})
    data["intent"] = ExecutionIntent.PREPARE
    with pytest.raises(AdapterContractError):
        adapter.execute(prepared, ExecutionRequest.model_validate(data))


def test_mock_requires_explicit_enum_and_input_bytes_are_strict() -> None:
    with pytest.raises(AdapterContractError):
        MockAdapter(scenario="NORMAL")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        InputEvent.model_validate(
            {"run_id": "mock-run", "name": "mock.input", "data": bytearray(b"x")}
        )


@pytest.mark.parametrize("max_input_bytes", [0, 512])
def test_empty_mock_events_are_rejected_without_polluting_later_evidence(
    max_input_bytes: int,
) -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    budget_data = contract.budget.model_dump()
    budget_data["max_input_bytes"] = max_input_bytes
    preparation = adapter.prepare(
        contract.candidate, PlatformRequirement(), ExecutionBudget.model_validate(budget_data)
    )
    assert preparation.status is PreparationStatus.READY
    assert preparation.prepared is not None
    prepared = preparation.prepared
    request = request_for(prepared)
    for _ in range(3):
        with pytest.raises(AdapterContractError, match="Empty input events"):
            adapter.inject_input(
                prepared, request, InputEvent(run_id=request.run_id, name="mock.input", data=b"")
            )
    accepted = None
    if max_input_bytes > 0:
        accepted = adapter.inject_input(
            prepared, request, InputEvent(run_id=request.run_id, name="mock.input", data=b"valid")
        )
        assert accepted.accepted
    execution = adapter.execute(prepared, request)
    evidence = adapter.collect_raw_evidence(execution)
    if accepted is not None:
        assert len(evidence) == 2
        assert evidence[0] == accepted.evidence
    else:
        assert len(evidence) == 1
    assert all(raw.content for raw in evidence)


def test_mock_verification_none_is_not_checked_and_request_cannot_be_substituted() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    data = request_for(prepared).model_dump(exclude={"id"})
    data["verification_requirement"] = VerificationRequirement.NONE
    request = ExecutionRequest.model_validate(data)
    execution = adapter.execute(prepared, request)
    unchecked = adapter.verify(contract.candidate, contract.trigger, execution)
    assert unchecked.verdict is VerificationVerdict.NOT_CHECKED
    assert unchecked.scope is ExecutionScope.MOCK
    assert unchecked.evidence == adapter.collect_raw_evidence(execution)
    assert "not requested" in unchecked.reason.lower()
    with pytest.raises(AdapterContractError):
        adapter.execute(prepared, request_for(prepared, request.run_id))
    trigger_data = contract.trigger.model_dump(exclude={"id"})
    trigger_data["vulnerability_id"] = "unrelated-vulnerability"
    with pytest.raises(ValueError):
        adapter.verify(contract.candidate, TriggerSpec.model_validate(trigger_data), execution)


def test_ready_preparation_decision_and_artifact_must_share_platform_identity() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    preparation = adapter.prepare(contract.candidate, PlatformRequirement(), contract.budget)
    assert PreparationResult.model_validate_json(preparation.model_dump_json()) == preparation
    assert preparation.prepared is not None
    decision_data = preparation.decision.model_dump()
    decision_data["platform_id"] = "different-platform"
    decision = type(preparation.decision).model_validate(decision_data)
    with pytest.raises(ValidationError):
        PreparationResult(
            status=PreparationStatus.READY,
            decision=decision,
            prepared=preparation.prepared,
            reason="invalid cross-platform READY fixture",
        )


def test_accepted_input_receipt_cannot_substitute_a_build_log() -> None:
    adapter = MockAdapter()
    contract = mock_contract(adapter.query_capability())
    prepared = ready(adapter, contract)
    request = request_for(prepared)
    receipt = adapter.inject_input(
        prepared, request, InputEvent(run_id=request.run_id, name="mock.input", data=b"fixture")
    )
    assert receipt.evidence is not None
    data = receipt.evidence.reference.model_dump(exclude={"id"})
    data["kind"] = EvidenceKind.BUILD_LOG
    false_input = RawEvidence(
        reference=EvidenceReference.model_validate(data), content=receipt.evidence.content
    )
    with pytest.raises(ValidationError):
        InputReceipt(
            status=FactStatus.OBSERVED,
            accepted=True,
            evidence=false_input,
            reason="invalid build-log receipt fixture",
        )
