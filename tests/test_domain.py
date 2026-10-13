"""Offline scientific-contract tests; no declared runtime fixture is an attestation."""

from hashlib import sha256
from typing import TypeVar

import pytest
from conftest import RuntimeContract, known
from pydantic import BaseModel, ValidationError

from heteropoclab.core import Fact, FactStatus, SourceKind, stable_id
from heteropoclab.domain import (
    EvidenceKind,
    EvidenceReference,
    ExecutionBudget,
    ExecutionIntent,
    ExecutionRequest,
    ExecutionScope,
    InputEvent,
    ObservationKind,
    PoCCandidate,
    RawEvidence,
    TerminationStatus,
    TriggerConstraint,
    TriggerSpec,
    VerificationRequirement,
    VerificationResult,
    VerificationVerdict,
    verify,
)

M = TypeVar("M", bound=BaseModel)


def clone(model: M, **updates: object) -> M:
    data = model.model_dump()
    data.pop("id", None)
    data.update(updates)
    return type(model).model_validate(data)


def result(contract: RuntimeContract, **updates: object) -> VerificationResult:
    values: dict[str, object] = {
        "candidate": contract.candidate,
        "trigger": contract.trigger,
        "execution": contract.execution,
        "evidence": (contract.evidence,),
        "observations": (contract.observation,),
    }
    values.update(updates)
    # Public API is exercised with values that vary by rejection scenario.
    return verify(**values)  # type: ignore[arg-type]


def test_fictional_runtime_fixture_exercises_only_structural_verified_contract(
    runtime_contract: RuntimeContract,
) -> None:
    verified = result(runtime_contract)
    assert verified.verdict is VerificationVerdict.VERIFIED
    assert verified.scope is ExecutionScope.TARGET_RUNTIME
    assert "fictional" in runtime_contract.observation.provenance.method
    assert "synthetic" in runtime_contract.evidence.content.decode("utf-8", errors="replace")


def test_all_domain_models_and_binary_input_roundtrip(runtime_contract: RuntimeContract) -> None:
    budget = ExecutionBudget(
        timeout_seconds=1.0, max_memory_bytes=4096, max_output_bytes=4096, max_input_bytes=256
    )
    request = ExecutionRequest(
        run_id="test-run",
        candidate_id=runtime_contract.candidate.id,
        platform_id=runtime_contract.candidate.platform_id,
        budget=budget,
        intent=ExecutionIntent.RUN,
        verification_requirement=VerificationRequirement.CONDITION_LINKED_RUNTIME,
    )
    event = InputEvent(run_id=request.run_id, name="UART", data=bytes(range(256)))
    models = (
        runtime_contract.vulnerability,
        runtime_contract.trigger.constraints[0],
        runtime_contract.trigger,
        runtime_contract.candidate,
        budget,
        request,
        event,
        runtime_contract.evidence.reference,
        runtime_contract.evidence,
        runtime_contract.observation,
        runtime_contract.execution,
        result(runtime_contract),
    )
    for model in models:
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model
        assert restored.schema_version == "1.0"
    assert InputEvent.model_validate_json(event.model_dump_json()).data == bytes(range(256))


def test_content_id_excludes_incidental_provenance_but_keeps_scientific_content(
    runtime_contract: RuntimeContract,
) -> None:
    candidate = runtime_contract.candidate
    incidental = clone(
        candidate.provenance,
        locator="fixture://host-B/source",
        host_path="fixture://host-B/path",
        recorded_at="fictional-time-B",
        model_confidence=0.25,
        run_id="unrelated-generation-run",
    )
    assert clone(candidate, provenance=incidental).id == candidate.id
    changed_data = candidate.model_dump(exclude={"id", "source_sha256"})
    changed_data["payload"] = candidate.payload + " changed scientific payload"
    changed = PoCCandidate.model_validate(changed_data)
    assert changed.id != candidate.id
    assert changed.source_sha256 == sha256(changed.payload.encode()).hexdigest()
    assert stable_id("test", {"x": 1, "y": 2}) == stable_id("test", {"y": 2, "x": 1})


def test_run_identity_is_independent_from_evidence_content_identity(
    runtime_contract: RuntimeContract,
) -> None:
    reference = runtime_contract.evidence.reference
    other = clone(
        reference,
        run_id="fictional-run-B",
        provenance=clone(reference.provenance, run_id="fictional-run-B"),
    )
    assert other.id == reference.id
    assert other.run_id != reference.run_id
    with pytest.raises(ValueError):
        result(
            runtime_contract,
            evidence=(RawEvidence(reference=other, content=runtime_contract.evidence.content),),
        )


@pytest.mark.parametrize("model_name", ["vulnerability", "trigger", "candidate", "observation"])
def test_supplied_content_id_conflict_is_rejected(
    runtime_contract: RuntimeContract, model_name: str
) -> None:
    model = getattr(runtime_contract, model_name)
    data = model.model_dump()
    data["id"] = "candidate:" + "0" * 64
    with pytest.raises(ValidationError):
        type(model).model_validate(data)


def test_evidence_identity_and_candidate_source_hash_conflicts(
    runtime_contract: RuntimeContract,
) -> None:
    evidence_data = runtime_contract.evidence.reference.model_dump()
    evidence_data["id"] = "evidence:" + "0" * 64
    with pytest.raises(ValidationError):
        EvidenceReference.model_validate(evidence_data)
    data = runtime_contract.candidate.model_dump(exclude={"id"})
    data["source_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        PoCCandidate.model_validate(data)
    with pytest.raises(ValidationError):
        clone(
            runtime_contract.candidate,
            provenance=clone(runtime_contract.candidate.provenance, source_sha256="0" * 64),
        )


@pytest.mark.parametrize("mutation", ["bytes", "length", "hash"])
def test_raw_evidence_validates_actual_bytes(
    runtime_contract: RuntimeContract, mutation: str
) -> None:
    raw = runtime_contract.evidence
    reference = raw.reference
    content = raw.content
    if mutation == "bytes":
        content = b"changed"
    elif mutation == "length":
        reference = clone(reference, byte_length=reference.byte_length + 1)
    else:
        reference = clone(reference, sha256="0" * 64)
    with pytest.raises(ValidationError):
        RawEvidence(reference=reference, content=content)


@pytest.mark.parametrize("bad", [True, 1, "1.0", 0.0, -1.0, float("nan"), float("inf")])
def test_timeout_budget_is_strict_positive_finite_float(bad: object) -> None:
    with pytest.raises(ValidationError):
        ExecutionBudget.model_validate(
            {
                "timeout_seconds": bad,
                "max_memory_bytes": 1,
                "max_output_bytes": 1,
                "max_input_bytes": 0,
            }
        )


@pytest.mark.parametrize("field", ["max_memory_bytes", "max_output_bytes", "max_input_bytes"])
@pytest.mark.parametrize("bad", [True, "1", 1.0, -1])
def test_integer_budgets_have_no_silent_coercion(field: str, bad: object) -> None:
    data = {
        "timeout_seconds": 1.0,
        "max_memory_bytes": 1,
        "max_output_bytes": 1,
        "max_input_bytes": 0,
    }
    data[field] = bad
    with pytest.raises(ValidationError):
        ExecutionBudget.model_validate(data)


@pytest.mark.parametrize(
    "status", [FactStatus.UNKNOWN, FactStatus.NOT_CHECKED, FactStatus.UNSUPPORTED]
)
def test_unknown_facts_cannot_assert_values(status: FactStatus) -> None:
    assert Fact[bool](status=status).value is None
    with pytest.raises(ValidationError):
        Fact[bool](value=False, status=status)


@pytest.mark.parametrize("bad", ["true", 1, [], {}])
def test_condition_boolean_has_no_coercion(bad: object) -> None:
    with pytest.raises(ValidationError):
        Fact[bool].model_validate(
            {
                "value": bad,
                "status": FactStatus.INFERRED,
                "source": SourceKind.SYNTHETIC,
                "provenance": "fixture",
            }
        )


def test_known_fact_and_observed_model_claim_are_invalid_without_evidence() -> None:
    with pytest.raises(ValidationError):
        Fact[bool](value=True, status=FactStatus.OBSERVED)
    with pytest.raises(ValidationError):
        Fact[bool](
            value=True,
            status=FactStatus.OBSERVED,
            source=SourceKind.MODEL,
            provenance="model response",
        )
    with pytest.raises(ValidationError):
        InputEvent.model_validate({"run_id": "run", "name": "UART", "data": "text"})
    with pytest.raises(ValidationError):
        TriggerConstraint.model_validate(
            {"name": "fixture", "expression": "x", "schema_version": "2.0"}
        )
    with pytest.raises(ValidationError):
        TriggerConstraint.model_validate({"name": "fixture", "expression": "x", "confidence": 1.0})


@pytest.mark.parametrize(
    "status", [FactStatus.UNKNOWN, FactStatus.NOT_CHECKED, FactStatus.UNSUPPORTED]
)
def test_unknown_trigger_knowledge_propagates(
    status: FactStatus, runtime_contract: RuntimeContract
) -> None:
    constraint = clone(runtime_contract.trigger.constraints[0], knowledge=Fact[bool](status=status))
    trigger = clone(runtime_contract.trigger, constraints=(constraint,))
    candidate = clone(runtime_contract.candidate, trigger_id=trigger.id)
    execution = clone(runtime_contract.execution, candidate_id=candidate.id)
    raw = runtime_contract.evidence
    evidence = RawEvidence(
        reference=clone(raw.reference, candidate_id=candidate.id), content=raw.content
    )
    observation = clone(
        runtime_contract.observation,
        candidate_id=candidate.id,
        condition_id=constraint.id,
        evidence_ids=(evidence.reference.id,),
    )
    assessed = verify(candidate, trigger, execution, (evidence,), (observation,))
    assert assessed.verdict is VerificationVerdict.UNKNOWN
    with pytest.raises(ValidationError):
        clone(assessed, verdict=VerificationVerdict.VERIFIED)


@pytest.mark.parametrize("status", [FactStatus.UNKNOWN, FactStatus.NOT_CHECKED])
def test_unknown_target_observation_propagates(
    status: FactStatus, runtime_contract: RuntimeContract
) -> None:
    observation = clone(runtime_contract.observation, status=status, satisfied=None)
    assert (
        result(runtime_contract, observations=(observation,)).verdict is VerificationVerdict.UNKNOWN
    )


def test_missing_evidence_and_missing_source_provenance_cannot_verify(
    runtime_contract: RuntimeContract,
) -> None:
    assert (
        result(runtime_contract, evidence=(), observations=()).verdict
        is VerificationVerdict.UNKNOWN
    )
    raw = runtime_contract.evidence
    reference = clone(raw.reference, provenance=clone(raw.reference.provenance, source_sha256=None))
    evidence = RawEvidence(reference=reference, content=raw.content)
    observation = clone(runtime_contract.observation, evidence_ids=(reference.id,))
    assert (
        result(runtime_contract, evidence=(evidence,), observations=(observation,)).verdict
        is VerificationVerdict.UNKNOWN
    )


@pytest.mark.parametrize(
    "kind",
    [ObservationKind.COMPILE, ObservationKind.RUN, ObservationKind.CRASH, ObservationKind.MODEL],
)
def test_unrelated_outcome_is_not_target_condition_proof(
    kind: ObservationKind, runtime_contract: RuntimeContract
) -> None:
    observation = clone(runtime_contract.observation, kind=kind)
    assessed = result(runtime_contract, observations=(observation,))
    assert assessed.verdict is VerificationVerdict.NOT_VERIFIED
    with pytest.raises(ValidationError):
        clone(assessed, verdict=VerificationVerdict.VERIFIED)


@pytest.mark.parametrize("termination", [TerminationStatus.CRASH, TerminationStatus.TIMEOUT])
def test_crash_or_timeout_alone_is_not_verified(
    termination: TerminationStatus, runtime_contract: RuntimeContract
) -> None:
    execution = clone(runtime_contract.execution, termination=termination, exit_code=1)
    assert result(runtime_contract, execution=execution).verdict is VerificationVerdict.NOT_VERIFIED


def test_compile_and_build_logs_cannot_establish_runtime_trigger(
    runtime_contract: RuntimeContract,
) -> None:
    compile_record = clone(runtime_contract.execution, scope=ExecutionScope.COMPILE_ONLY)
    assert (
        result(runtime_contract, execution=compile_record).verdict
        is VerificationVerdict.NOT_VERIFIED
    )
    raw = runtime_contract.evidence
    build = RawEvidence(
        reference=clone(raw.reference, kind=EvidenceKind.BUILD_LOG), content=raw.content
    )
    observation = clone(runtime_contract.observation, evidence_ids=(build.reference.id,))
    assert (
        result(runtime_contract, evidence=(build,), observations=(observation,)).verdict
        is VerificationVerdict.NOT_VERIFIED
    )


def test_model_inference_and_mock_bytes_cannot_be_upgraded(
    runtime_contract: RuntimeContract,
) -> None:
    model_observation = clone(
        runtime_contract.observation,
        status=FactStatus.INFERRED,
        provenance=clone(runtime_contract.observation.provenance, source=SourceKind.MODEL),
    )
    assert (
        result(runtime_contract, observations=(model_observation,)).verdict
        is VerificationVerdict.NOT_VERIFIED
    )
    raw = runtime_contract.evidence
    mock_reference = clone(
        raw.reference, provenance=clone(raw.reference.provenance, source=SourceKind.MOCK)
    )
    mock_evidence = RawEvidence(reference=mock_reference, content=raw.content)
    mock_observation = clone(
        runtime_contract.observation,
        evidence_ids=(mock_reference.id,),
        provenance=clone(runtime_contract.observation.provenance, source=SourceKind.MOCK),
    )
    execution = clone(runtime_contract.execution, source=SourceKind.MOCK, scope=ExecutionScope.MOCK)
    assessed = result(
        runtime_contract,
        execution=execution,
        evidence=(mock_evidence,),
        observations=(mock_observation,),
    )
    assert assessed.verdict is VerificationVerdict.NOT_VERIFIED
    with pytest.raises(ValidationError):
        clone(assessed, verdict=VerificationVerdict.VERIFIED)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "another-run"),
        ("candidate_id", "another-candidate"),
        ("platform_id", "fictional.contract.platform.driver-B"),
    ],
)
def test_cross_run_candidate_and_runtime_scope_are_rejected(
    field: str,
    value: str,
    runtime_contract: RuntimeContract,
) -> None:
    execution = clone(runtime_contract.execution, **{field: value})
    with pytest.raises(ValueError):
        result(runtime_contract, execution=execution)


def test_evidence_and_observation_provenance_relationships_are_enforced(
    runtime_contract: RuntimeContract,
) -> None:
    with pytest.raises(ValidationError):
        clone(
            runtime_contract.evidence.reference,
            provenance=clone(runtime_contract.evidence.reference.provenance, run_id="other"),
        )
    with pytest.raises(ValidationError):
        clone(
            runtime_contract.observation,
            provenance=clone(runtime_contract.observation.provenance, scope="other"),
        )
    raw = runtime_contract.evidence
    for field, changed in (("source", SourceKind.EMULATOR), ("source_sha256", "0" * 64)):
        provenance = clone(raw.reference.provenance, **{field: changed})
        evidence = RawEvidence(
            reference=clone(raw.reference, provenance=provenance), content=raw.content
        )
        with pytest.raises(ValueError):
            result(runtime_contract, evidence=(evidence,), observations=())


def test_sanitizer_errors_must_link_the_target_constraint_and_available_bytes(
    runtime_contract: RuntimeContract,
) -> None:
    sanitizer = clone(runtime_contract.observation, kind=ObservationKind.SANITIZER_ERROR)
    assert (
        result(runtime_contract, observations=(sanitizer,)).verdict is VerificationVerdict.VERIFIED
    )
    with pytest.raises(ValidationError):
        clone(sanitizer, condition_id=None)
    with pytest.raises(ValueError):
        result(
            runtime_contract, observations=(clone(sanitizer, condition_id="unrelated-condition"),)
        )
    with pytest.raises(ValueError):
        result(
            runtime_contract, observations=(clone(sanitizer, evidence_ids=("missing-evidence",)),)
        )
    assert (
        result(runtime_contract, observations=(clone(sanitizer, evidence_ids=()),)).verdict
        is VerificationVerdict.NOT_VERIFIED
    )


def test_conflicting_observations_and_uncovered_constraints_are_conservative(
    runtime_contract: RuntimeContract,
) -> None:
    false = clone(runtime_contract.observation, satisfied=False)
    assert (
        result(runtime_contract, observations=(runtime_contract.observation, false)).verdict
        is VerificationVerdict.UNKNOWN
    )
    additional = TriggerConstraint(
        name="uncovered", expression="second fixture condition", knowledge=known(True)
    )
    trigger = clone(
        runtime_contract.trigger, constraints=(*runtime_contract.trigger.constraints, additional)
    )
    candidate = clone(runtime_contract.candidate, trigger_id=trigger.id)
    execution = clone(runtime_contract.execution, candidate_id=candidate.id)
    raw = runtime_contract.evidence
    evidence = RawEvidence(
        reference=clone(raw.reference, candidate_id=candidate.id), content=raw.content
    )
    observation = clone(
        runtime_contract.observation,
        candidate_id=candidate.id,
        evidence_ids=(evidence.reference.id,),
    )
    assert (
        verify(candidate, trigger, execution, (evidence,), (observation,)).verdict
        is VerificationVerdict.NOT_VERIFIED
    )


def test_duplicates_and_scope_escalation_are_rejected(runtime_contract: RuntimeContract) -> None:
    with pytest.raises(ValidationError):
        TriggerSpec(
            vulnerability_id=runtime_contract.vulnerability.id,
            platform_id=runtime_contract.candidate.platform_id,
            constraints=(runtime_contract.trigger.constraints[0],) * 2,
        )
    with pytest.raises(ValueError):
        result(runtime_contract, evidence=(runtime_contract.evidence,) * 2)
    with pytest.raises(ValidationError):
        clone(result(runtime_contract), scope=ExecutionScope.EMULATOR_RUNTIME)


def test_execution_request_content_id_excludes_run_identity_but_includes_intent_and_budget(
    runtime_contract: RuntimeContract,
) -> None:
    budget = ExecutionBudget(
        timeout_seconds=1.0, max_memory_bytes=4096, max_output_bytes=4096, max_input_bytes=0
    )
    request = ExecutionRequest(
        run_id="request-run-A",
        candidate_id=runtime_contract.candidate.id,
        platform_id=runtime_contract.candidate.platform_id,
        budget=budget,
        intent=ExecutionIntent.RUN,
        verification_requirement=VerificationRequirement.CONDITION_LINKED_RUNTIME,
    )
    assert clone(request, run_id="request-run-B").id == request.id
    assert clone(request, intent=ExecutionIntent.PREPARE).id != request.id
    assert clone(request, budget=clone(budget, max_memory_bytes=8192)).id != request.id
    with pytest.raises(ValidationError):
        ExecutionRequest.model_validate({**request.model_dump(), "intent": ExecutionIntent.PREPARE})


def test_emulator_contract_can_only_describe_emulator_scope(
    runtime_contract: RuntimeContract,
) -> None:
    execution = clone(
        runtime_contract.execution,
        source=SourceKind.EMULATOR,
        scope=ExecutionScope.EMULATOR_RUNTIME,
    )
    raw = runtime_contract.evidence
    reference = clone(
        raw.reference, provenance=clone(raw.reference.provenance, source=SourceKind.EMULATOR)
    )
    evidence = RawEvidence(reference=reference, content=raw.content)
    observation = clone(
        runtime_contract.observation,
        evidence_ids=(reference.id,),
        provenance=clone(runtime_contract.observation.provenance, source=SourceKind.EMULATOR),
    )
    assessed = result(
        runtime_contract, execution=execution, evidence=(evidence,), observations=(observation,)
    )
    assert assessed.verdict is VerificationVerdict.VERIFIED
    assert assessed.scope is ExecutionScope.EMULATOR_RUNTIME
    with pytest.raises(ValidationError):
        clone(assessed, scope=ExecutionScope.TARGET_RUNTIME)
    with pytest.raises(ValidationError):
        clone(execution, source=SourceKind.REAL_EXECUTION)


def test_unchecked_and_empty_trigger_contracts_do_not_gain_verification(
    runtime_contract: RuntimeContract,
) -> None:
    checked = result(runtime_contract)
    unchecked = clone(
        checked,
        verdict=VerificationVerdict.NOT_CHECKED,
        reason="this declared result has not been evaluated by a producer",
    )
    assert unchecked.verdict is VerificationVerdict.NOT_CHECKED
    empty = clone(runtime_contract.trigger, constraints=())
    candidate = clone(runtime_contract.candidate, trigger_id=empty.id)
    execution = clone(runtime_contract.execution, candidate_id=candidate.id)
    assert verify(candidate, empty, execution, (), ()).verdict is VerificationVerdict.UNKNOWN


def test_unsafe_model_copy_is_revalidated_at_verification_boundary(
    runtime_contract: RuntimeContract,
) -> None:
    tampered = runtime_contract.evidence.model_copy(update={"content": b"tampered"})
    with pytest.raises(ValidationError):
        result(runtime_contract, evidence=(tampered,))
    tampered_candidate = runtime_contract.candidate.model_copy(
        update={"payload": "tampered candidate payload"}
    )
    with pytest.raises(ValidationError):
        result(runtime_contract, candidate=tampered_candidate)


def test_verification_explanations_bind_unmet_conditions_and_record_scope_limits(
    runtime_contract: RuntimeContract,
) -> None:
    checked = result(runtime_contract)
    assert checked.unmet_conditions == ()
    assert any("do not attest real execution" in item for item in checked.limitations)
    assert any(runtime_contract.candidate.platform_id in item for item in checked.limitations)
    missing = result(runtime_contract, evidence=(), observations=())
    assert missing.verdict is VerificationVerdict.UNKNOWN
    assert missing.unmet_conditions == (runtime_contract.trigger.constraints[0].id,)
    assert type(missing).model_validate_json(missing.model_dump_json()) == missing
    with pytest.raises(ValidationError):
        clone(checked, unmet_conditions=(runtime_contract.trigger.constraints[0].id,))
    with pytest.raises(ValidationError):
        clone(missing, unmet_conditions=("condition:unrelated-target",))
    with pytest.raises(ValidationError):
        clone(missing, unmet_conditions=missing.unmet_conditions * 2)
    changed_explanation = clone(missing, limitations=("additional producer annotation",))
    assert changed_explanation.id == missing.id


def test_emulator_explanation_explicitly_limits_physical_hardware_claim(
    runtime_contract: RuntimeContract,
) -> None:
    execution = clone(
        runtime_contract.execution,
        source=SourceKind.EMULATOR,
        scope=ExecutionScope.EMULATOR_RUNTIME,
    )
    raw = runtime_contract.evidence
    reference = clone(
        raw.reference, provenance=clone(raw.reference.provenance, source=SourceKind.EMULATOR)
    )
    evidence = RawEvidence(reference=reference, content=raw.content)
    observation = clone(
        runtime_contract.observation,
        evidence_ids=(reference.id,),
        provenance=clone(runtime_contract.observation.provenance, source=SourceKind.EMULATOR),
    )
    checked = result(
        runtime_contract, execution=execution, evidence=(evidence,), observations=(observation,)
    )
    assert checked.unmet_conditions == ()
    assert any("not physical hardware" in item for item in checked.limitations)
