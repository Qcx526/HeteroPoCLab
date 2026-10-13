"""Generate deterministic contract examples; no compiler, device, or model is used."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeVar

from heteropoclab.core import Fact, FactStatus, SourceKind
from heteropoclab.domain import (
    ExecutionBudget,
    ExecutionIntent,
    ExecutionRequest,
    InputEvent,
    PoCCandidate,
    Provenance,
    TriggerConstraint,
    TriggerSpec,
    VerificationRequirement,
    VulnerabilitySpec,
)
from heteropoclab.mock import MockAdapter
from heteropoclab.platforms import ExecutionMode, PlatformCapability, PlatformRequirement, negotiate

T = TypeVar("T")


def declared(value: T) -> Fact[T]:
    return Fact(
        value=value,
        status=FactStatus.INFERRED,
        source=SourceKind.USER_REQUIREMENT,
        provenance="declared requirement scenario; no hardware or vendor manual inspected",
    )


def mock_flow() -> dict[str, object]:
    adapter = MockAdapter()
    capability = adapter.query_capability()
    provenance = Provenance(
        source=SourceKind.SYNTHETIC,
        method="deterministic offline contract example; no real vulnerability",
        scope=capability.id,
    )
    vulnerability = VulnerabilitySpec(
        name="Synthetic input condition",
        affected_component="fictional fixture parser",
        platform_id=capability.id,
        description="Contract example, not an actual vulnerability specification.",
        provenance=provenance,
    )
    condition = TriggerConstraint(
        name="declared input",
        expression="fixture receives the declared byte sequence",
        knowledge=declared(True),
    )
    trigger = TriggerSpec(
        vulnerability_id=vulnerability.id,
        platform_id=capability.id,
        constraints=(condition,),
    )
    requirement = PlatformRequirement(platform_id=capability.id)
    candidate = PoCCandidate(
        vulnerability_id=vulnerability.id,
        trigger_id=trigger.id,
        platform_id=capability.id,
        language="fixture-text",
        description="Non-executable text; Mock never evaluates this payload.",
        expected_behavior="Record input and Mock outcome; no verified vulnerability claim.",
        platform_requirement=requirement,
        payload="synthetic fixture payload",
        provenance=provenance,
    )
    budget = ExecutionBudget(
        timeout_seconds=1.0,
        max_memory_bytes=4096,
        max_output_bytes=4096,
        max_input_bytes=128,
    )
    request = ExecutionRequest(
        run_id="example-mock-run-001",
        candidate_id=candidate.id,
        platform_id=capability.id,
        budget=budget,
        intent=ExecutionIntent.PREPARE_AND_RUN,
        verification_requirement=VerificationRequirement.CONDITION_LINKED_RUNTIME,
    )
    preparation = adapter.prepare(candidate, requirement, budget)
    if preparation.prepared is None:
        raise RuntimeError("Known Mock example must prepare successfully.")
    event = InputEvent(run_id=request.run_id, name="mock.input", data=b"fixture\x00\xff")
    receipt = adapter.inject_input(preparation.prepared, request, event)
    execution = adapter.execute(preparation.prepared, request)
    evidence = adapter.collect_raw_evidence(execution)
    result = adapter.verify(candidate, trigger, execution)
    return {
        "schema_version": "1.0",
        "notice": "Offline Mock contract only; no candidate code or hardware was executed.",
        "platform_capability": capability.model_dump(mode="json"),
        "vulnerability_spec": vulnerability.model_dump(mode="json"),
        "trigger_spec": trigger.model_dump(mode="json"),
        "poc_candidate": candidate.model_dump(mode="json"),
        "execution_request": request.model_dump(mode="json"),
        "preparation_summary": {
            "status": preparation.status.value,
            "content_id": preparation.prepared.content_id,
            "decision": preparation.decision.model_dump(mode="json"),
            "notice": "Random preparation instance ID omitted from this deterministic example.",
        },
        "input_event": event.model_dump(mode="json"),
        "input_receipt": receipt.model_dump(mode="json"),
        "execution_record": execution.model_dump(mode="json"),
        "raw_evidence": [raw.model_dump(mode="json") for raw in evidence],
        "observations": [
            observation.model_dump(mode="json") for observation in result.observations
        ],
        "verification_result": result.model_dump(mode="json"),
    }


def c8000_scenario() -> dict[str, object]:
    capability = PlatformCapability(
        label="C8000 PowerPC requirement scenario; hardware NOT_CHECKED",
        category=declared("embedded"),
        isa=declared("PowerPC"),
        isa_profile=Fact[str](),
        os=declared("baremetal"),
        binary_format=declared("ELF"),
        endian=Fact[str](),
        bit_width=Fact[int](),
        abi=Fact[str](),
        toolchain=declared("self-developed environment; version NOT_CHECKED"),
        runtime=declared("baremetal hardware runtime; version NOT_CHECKED"),
        execution_mode=declared(ExecutionMode.REAL_HARDWARE),
        debug_capabilities=declared(("vendor.non-GDB.JTAG",)),
        observation_capabilities=declared(("UART", "hardware.result")),
        input_interfaces=declared(("UART", "CAN", "SPI", "GPIO")),
        evidence_capabilities=declared(("serial.bytes", "hardware.result")),
    )
    requirement = PlatformRequirement(
        platform_id=capability.id,
        isa=declared("PowerPC"),
        isa_profile=Fact[str](),
        os=declared("baremetal"),
        binary_format=declared("ELF"),
        endian=Fact[str](),
        bit_width=Fact[int](),
        abi=Fact[str](),
        debug_capabilities=declared(("vendor.non-GDB.JTAG",)),
        input_interfaces=declared(("UART", "CAN", "SPI", "GPIO")),
    )
    decision = negotiate(capability, requirement)
    return {
        "schema_version": "1.0",
        "notice": "Requirements only; no C8000 toolchain, debugger, bus, or hardware was tested.",
        "platform_capability": capability.model_dump(mode="json"),
        "platform_requirement": requirement.model_dump(mode="json"),
        "compatibility_decision": decision.model_dump(mode="json"),
        "can_execute": decision.can_execute,
        "manual_provenance_requirement": {
            "status": "NOT_CHECKED",
            "source": "DOCUMENTATION",
            "sha256": None,
            "note": "A future inspected vendor manual must supply its actual hash and provenance.",
        },
    }


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def examples() -> dict[str, str]:
    return {
        "m0a-mock-flow.json": encode(mock_flow()),
        "m0a-c8000-capability.json": encode(c8000_scenario()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs" / "examples"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check existing examples without writing."
    )
    arguments = parser.parse_args()
    generated = examples()
    if arguments.check:
        for name, content in generated.items():
            path = arguments.output_dir / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                print(f"Example differs or is missing: {name}")
                return 1
        print("Deterministic JSON examples match.")
        return 0
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        (arguments.output_dir / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"Generated {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
