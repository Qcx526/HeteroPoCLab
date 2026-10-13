"""CPU-only fixtures: every platform fact is declared synthetic, never measured."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import TypeVar

import pytest

from heteropoclab.core import Fact, FactStatus, SourceKind
from heteropoclab.domain import (
    EvidenceKind,
    EvidenceReference,
    ExecutionRecord,
    ExecutionScope,
    Observation,
    ObservationKind,
    PoCCandidate,
    Provenance,
    RawEvidence,
    TerminationStatus,
    TriggerConstraint,
    TriggerSpec,
    VulnerabilitySpec,
)
from heteropoclab.platforms import ExecutionMode, PlatformCapability

T = TypeVar("T")


def known(value: T) -> Fact[T]:
    return Fact(
        value=value,
        status=FactStatus.INFERRED,
        source=SourceKind.SYNTHETIC,
        provenance="synthetic offline contract fixture; no hardware measurement",
    )


def capability_fixture(
    platform_id: str,
    *,
    category: str,
    isa: str,
    profile: str,
    os: str,
    toolchain: str,
    runtime: str,
    mode: ExecutionMode,
    debug: tuple[str, ...] = (),
    inputs: tuple[str, ...] = (),
    observations: tuple[str, ...] = (),
) -> PlatformCapability:
    return PlatformCapability(
        label=platform_id,
        category=known(category),
        isa=known(isa),
        isa_profile=known(profile),
        os=known(os),
        binary_format=known("ELF"),
        endian=known("little"),
        bit_width=known(32),
        abi=known("fixture-ABI"),
        toolchain=known(toolchain),
        runtime=known(runtime),
        execution_mode=known(mode),
        debug_capabilities=known(debug),
        observation_capabilities=known(observations),
        input_interfaces=known(inputs),
        evidence_capabilities=known(("runtime.bytes",)),
    )


@pytest.fixture
def platform_fixtures() -> dict[str, PlatformCapability]:
    gpu = capability_fixture(
        "fixture.cuda.sm75.driver617.42",
        category="GPU",
        isa="NVIDIA-PTX",
        profile="sm_75",
        os="Linux-WSL2",
        toolchain="nvcc-12.8",
        runtime="CUDA-12.8/driver-617.42",
        mode=ExecutionMode.REAL_HARDWARE,
        debug=("compute-sanitizer",),
        inputs=("kernel.arguments",),
        observations=("host.stdout", "compute-sanitizer.memcheck"),
    )
    arm = capability_fixture(
        "fixture.arm.cortex-m",
        category="MCU",
        isa="ARM",
        profile="Cortex-M4",
        os="baremetal",
        toolchain="arm-none-eabi-gcc",
        runtime="baremetal",
        mode=ExecutionMode.SIMULATOR,
        debug=("GDB",),
        inputs=("UART", "GPIO"),
        observations=("UART",),
    )
    riscv_baremetal = capability_fixture(
        "fixture.riscv.baremetal",
        category="MCU",
        isa="RISC-V",
        profile="RV32IMAC",
        os="baremetal",
        toolchain="riscv-none-elf-gcc",
        runtime="baremetal",
        mode=ExecutionMode.SIMULATOR,
        debug=("GDB",),
        inputs=("UART", "SPI"),
        observations=("UART",),
    )
    riscv_rtos = capability_fixture(
        "fixture.riscv.rtos",
        category="MCU",
        isa="RISC-V",
        profile="RV32IMAC",
        os="fixture-RTOS",
        toolchain="riscv-none-elf-gcc",
        runtime="fixture-RTOS-1",
        mode=ExecutionMode.SIMULATOR,
        debug=("GDB",),
        inputs=("CAN", "GPIO"),
        observations=("UART",),
    )
    c8000 = capability_fixture(
        "fixture.c8000.requirements",
        category="embedded",
        isa="PowerPC",
        profile="placeholder",
        os="baremetal",
        toolchain="self-developed-toolchain",
        runtime="baremetal",
        mode=ExecutionMode.MOCK,
        debug=("vendor.non-GDB.JTAG",),
        inputs=("UART", "CAN", "SPI", "GPIO"),
        observations=("UART",),
    )
    c8000_data = c8000.model_dump(exclude={"id"})
    for field in ("isa_profile", "endian", "bit_width", "abi"):
        c8000_data[field] = Fact(status=FactStatus.UNKNOWN)
    # C8000 facts are user requirements; no vendor manual was inspected.
    c8000_data["execution_mode"] = Fact[ExecutionMode](
        value=ExecutionMode.MOCK,
        status=FactStatus.OBSERVED,
        source=SourceKind.MOCK,
        provenance="declared mock fixture mode",
    )
    c8000_data["isa"] = Fact[str](
        value="PowerPC",
        status=FactStatus.INFERRED,
        source=SourceKind.USER_REQUIREMENT,
        provenance="declared C8000 requirement; no vendor manual inspected",
    )
    c8000 = PlatformCapability.model_validate(c8000_data)
    return {
        "gpu": gpu,
        "arm": arm,
        "riscv_baremetal": riscv_baremetal,
        "riscv_rtos": riscv_rtos,
        "c8000": c8000,
    }


@pytest.fixture
def known_fact() -> Callable[[T], Fact[T]]:
    return known


# These are deliberately fictional REAL_EXECUTION declarations used to test the
# structural contract. They are not hardware attestation or vulnerability proof.
@dataclass(frozen=True)
class RuntimeContract:
    vulnerability: VulnerabilitySpec
    trigger: TriggerSpec
    candidate: PoCCandidate
    execution: ExecutionRecord
    evidence: RawEvidence
    observation: Observation


@pytest.fixture
def runtime_contract(platform_fixtures: dict[str, PlatformCapability]) -> RuntimeContract:
    scope = platform_fixtures["gpu"].id
    vulnerability = VulnerabilitySpec(
        name="synthetic contract target",
        platform_id=scope,
        affected_component="fixture component",
        description="CPU-only fabricated test declaration, never real vulnerability evidence",
        provenance=Provenance(
            source=SourceKind.SYNTHETIC, method="offline contract fixture", scope=scope
        ),
    )
    condition = TriggerConstraint(
        name="fixture condition", expression="fixture_input == 7", knowledge=known(True)
    )
    trigger = TriggerSpec(
        vulnerability_id=vulnerability.id, platform_id=scope, constraints=(condition,)
    )
    candidate = PoCCandidate(
        vulnerability_id=vulnerability.id,
        trigger_id=trigger.id,
        platform_id=scope,
        language="fixture-text",
        payload="synthetic non-executable contract payload",
        description="synthetic contract candidate",
        expected_behavior="fictional condition declaration",
        provenance=Provenance(
            source=SourceKind.SYNTHETIC, method="offline contract fixture", scope=scope
        ),
    )
    run_id = "fictional-run-A"
    provenance = Provenance(
        source=SourceKind.REAL_EXECUTION,
        method="fictional runtime declaration for contract test only",
        scope=scope,
        source_sha256=candidate.source_sha256,
        run_id=run_id,
    )
    content = b"synthetic runtime bytes: fixture condition=true\x00\xff"
    evidence = RawEvidence(
        reference=EvidenceReference(
            run_id=run_id,
            candidate_id=candidate.id,
            platform_id=scope,
            kind=EvidenceKind.RUNTIME_LOG,
            sha256=sha256(content).hexdigest(),
            byte_length=len(content),
            provenance=provenance,
        ),
        content=content,
    )
    observation = Observation(
        run_id=run_id,
        candidate_id=candidate.id,
        platform_id=scope,
        kind=ObservationKind.CONDITION,
        status=FactStatus.OBSERVED,
        satisfied=True,
        condition_id=condition.id,
        evidence_ids=(evidence.reference.id,),
        provenance=provenance,
    )
    execution = ExecutionRecord(
        run_id=run_id,
        candidate_id=candidate.id,
        platform_id=scope,
        source=SourceKind.REAL_EXECUTION,
        scope=ExecutionScope.TARGET_RUNTIME,
        completed=True,
        termination=TerminationStatus.NORMAL,
        exit_code=0,
    )
    return RuntimeContract(vulnerability, trigger, candidate, execution, evidence, observation)
