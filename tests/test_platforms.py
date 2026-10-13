"""Conservative negotiation across declared CPU-only platform fixtures."""

import pytest
from conftest import known
from pydantic import ValidationError

from heteropoclab.core import Fact, FactStatus, SourceKind
from heteropoclab.platforms import (
    CompatibilityStatus,
    ExecutionMode,
    PlatformCapability,
    PlatformRequirement,
    negotiate,
)


@pytest.mark.parametrize("platform", ["gpu", "arm", "riscv_baremetal", "riscv_rtos"])
def test_platform_dimensions_roundtrip_and_matching_requirement(
    platform_fixtures: dict[str, PlatformCapability],
    platform: str,
) -> None:
    cap = platform_fixtures[platform]
    assert PlatformCapability.model_validate_json(cap.model_dump_json()) == cap
    request = PlatformRequirement(
        platform_id=cap.id,
        isa=cap.isa,
        isa_profile=cap.isa_profile,
        os=cap.os,
        binary_format=cap.binary_format,
        toolchain=cap.toolchain,
        runtime=cap.runtime,
        execution_mode=cap.execution_mode,
        input_interfaces=cap.input_interfaces,
        observation_capabilities=cap.observation_capabilities,
    )
    decision = negotiate(cap, request)
    assert decision.status is CompatibilityStatus.COMPATIBLE
    assert decision.can_execute
    assert decision.issues == ()


def test_c8000_preserves_declared_unknowns_and_independent_interfaces(
    platform_fixtures: dict[str, PlatformCapability],
) -> None:
    cap = platform_fixtures["c8000"]
    assert cap.isa.value == "PowerPC"
    assert cap.isa.source is SourceKind.USER_REQUIREMENT
    assert cap.isa.provenance == "declared C8000 requirement; no vendor manual inspected"
    for name in ("isa_profile", "endian", "bit_width", "abi"):
        fact = getattr(cap, name)
        assert fact.status is FactStatus.UNKNOWN
        assert fact.value is None
    assert cap.toolchain.value == "self-developed-toolchain"
    assert cap.debug_capabilities.value == ("vendor.non-GDB.JTAG",)
    assert cap.input_interfaces.value == ("UART", "CAN", "SPI", "GPIO")
    assert cap.observation_capabilities.value == ("UART",)
    requested = PlatformRequirement(
        isa=known("PowerPC"),
        os=known("baremetal"),
        binary_format=known("ELF"),
        toolchain=known("self-developed-toolchain"),
        debug_capabilities=known(("vendor.non-GDB.JTAG",)),
        input_interfaces=known(("UART", "CAN", "SPI", "GPIO")),
        observation_capabilities=known(("UART",)),
        isa_profile=Fact[str](),
        endian=Fact[str](),
        bit_width=Fact[int](),
        abi=Fact[str](),
    )
    decision = negotiate(cap, requested)
    assert decision.status is CompatibilityStatus.UNKNOWN
    assert not decision.can_execute
    assert {issue.dimension for issue in decision.issues} == {
        "isa_profile",
        "endian",
        "bit_width",
        "abi",
    }
    assert PlatformCapability.model_validate_json(cap.model_dump_json()) == cap


@pytest.mark.parametrize(
    "dimension,requested",
    [
        ("category", "CPU"),
        ("isa", "RISC-V"),
        ("isa_profile", "sm_80"),
        ("os", "Windows"),
        ("binary_format", "PE"),
        ("endian", "big"),
        ("bit_width", 64),
        ("abi", "other-ABI"),
        ("toolchain", "other-compiler"),
        ("runtime", "CUDA-12.8/driver-999.0"),
        ("execution_mode", ExecutionMode.SIMULATOR),
        ("debug_capabilities", ("GDB",)),
        ("observation_capabilities", ("UART",)),
        ("input_interfaces", ("SPI",)),
        ("evidence_capabilities", ("trace",)),
    ],
)
def test_every_platform_dimension_rejects_a_definite_mismatch(
    platform_fixtures: dict[str, PlatformCapability],
    dimension: str,
    requested: object,
) -> None:
    requirement = PlatformRequirement(**{dimension: known(requested)})
    decision = negotiate(platform_fixtures["gpu"], requirement)
    assert decision.status is CompatibilityStatus.UNSUPPORTED
    assert not decision.can_execute
    assert [issue.dimension for issue in decision.issues] == [dimension]


@pytest.mark.parametrize("status", [FactStatus.UNKNOWN, FactStatus.NOT_CHECKED])
def test_required_unknown_is_never_wildcard(
    platform_fixtures: dict[str, PlatformCapability],
    status: FactStatus,
) -> None:
    request = PlatformRequirement(isa=Fact[str](status=status))
    decision = negotiate(platform_fixtures["arm"], request)
    assert decision.status is CompatibilityStatus.UNKNOWN
    assert not decision.can_execute


@pytest.mark.parametrize("status", [FactStatus.UNKNOWN, FactStatus.NOT_CHECKED])
def test_missing_capability_does_not_inherit_isa_defaults(
    platform_fixtures: dict[str, PlatformCapability],
    status: FactStatus,
) -> None:
    data = platform_fixtures["arm"].model_dump(exclude={"id"})
    data["endian"] = Fact[str](status=status)
    cap = PlatformCapability.model_validate(data)
    decision = negotiate(cap, PlatformRequirement(endian=known("little")))
    assert decision.status is CompatibilityStatus.UNKNOWN
    assert not decision.can_execute


def test_set_requirements_are_subsets_but_identifiers_are_exact(
    platform_fixtures: dict[str, PlatformCapability],
) -> None:
    cap = platform_fixtures["c8000"]
    assert negotiate(cap, PlatformRequirement(input_interfaces=known(("CAN", "GPIO")))).can_execute
    assert negotiate(cap, PlatformRequirement(input_interfaces=known(()))).can_execute
    assert not negotiate(cap, PlatformRequirement(input_interfaces=known(("uart",)))).can_execute
    assert not negotiate(cap, PlatformRequirement(debug_capabilities=known(("GDB",)))).can_execute
    assert not negotiate(
        cap, PlatformRequirement(observation_capabilities=known(("CAN",)))
    ).can_execute


def test_runtime_and_execution_modes_cannot_cross_scope(
    platform_fixtures: dict[str, PlatformCapability],
) -> None:
    assert not negotiate(
        platform_fixtures["arm"],
        PlatformRequirement(
            execution_mode=known(ExecutionMode.REAL_HARDWARE),
        ),
    ).can_execute
    assert not negotiate(
        platform_fixtures["gpu"],
        PlatformRequirement(
            runtime=known("CUDA-12.8/driver-617.43"),
        ),
    ).can_execute
    assert not negotiate(
        platform_fixtures["riscv_baremetal"],
        PlatformRequirement(
            os=known("fixture-RTOS"),
            runtime=known("fixture-RTOS-1"),
        ),
    ).can_execute


def test_unsupported_overrides_unknown_and_platform_identity_must_match(
    platform_fixtures: dict[str, PlatformCapability],
) -> None:
    decision = negotiate(
        platform_fixtures["c8000"],
        PlatformRequirement(
            platform_id="another-platform",
            abi=Fact[str](),
        ),
    )
    assert decision.status is CompatibilityStatus.UNSUPPORTED
    assert not decision.can_execute
    assert {issue.dimension for issue in decision.issues} == {"platform_id", "abi"}


@pytest.mark.parametrize("bad", [True, "32", 32.0])
def test_bit_width_has_no_silent_coercion(bad: object) -> None:
    with pytest.raises(ValidationError):
        PlatformRequirement(bit_width=known(bad))


def test_platform_schema_rejects_unknown_fields_and_mutation() -> None:
    with pytest.raises(ValidationError):
        PlatformRequirement(ISA=known("ARM"))
    request = PlatformRequirement(isa=known("ARM"))
    with pytest.raises(ValidationError):
        request.isa = known("RISC-V")


def test_platform_content_identity_binds_driver_and_rejects_a_stale_id(
    platform_fixtures: dict[str, PlatformCapability],
) -> None:
    capability = platform_fixtures["gpu"]
    data = capability.model_dump(exclude={"id"})
    data["runtime"] = known("CUDA-12.8/driver-617.43")
    changed = PlatformCapability.model_validate(data)
    assert changed.label == capability.label
    assert changed.id != capability.id
    assert not negotiate(changed, PlatformRequirement(platform_id=capability.id)).can_execute
    with pytest.raises(ValidationError):
        PlatformCapability.model_validate({**data, "id": capability.id})
