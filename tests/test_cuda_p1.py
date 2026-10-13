"""CPU-only CUDA-P1 contract tests; no CUDA executable is invoked."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from heteropoclab.adapters.cuda import (
    CudaAdapter,
    CudaExperiment,
    CudaOracle,
    _application_exit,
    deterministic_trials,
    generate_candidates,
    load_experiment,
    make_vulnerability_and_trigger,
    parse_sanitizer_version,
    sanitizer_error_count,
    target_diagnostic_count,
)
from heteropoclab.protocol import PlatformAdapter

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "examples/cuda_p1/experiment.json"
FIXTURE_PATH = ROOT / "examples/cuda_p1/oob_write.cu"

MATCHING_REPORT = """========= Invalid __global__ write of size 4 bytes
=========     at <unnamed>::bounded_write_kernel(int *, int, int, int)+0xc0 in oob_write.cu:27
=========     by thread (0,0,0) in block (1,0,0)
=========     Address 0x704600080 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x704600000 of size 128 bytes
========= Saved host backtrace
========= Program hit cudaErrorLaunchFailure (error 719)
"""


def test_declared_matrix_is_deterministic_and_bounded() -> None:
    experiment = load_experiment(SPEC_PATH)
    trials = deterministic_trials(experiment)
    assert len(trials) == 8
    assert [(item.case_id, item.variant) for item in trials] == [
        (case.case_id, variant) for case in experiment.cases for variant in experiment.variants
    ]
    vulnerability, trigger = make_vulnerability_and_trigger(
        "platform:test-cuda", hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    )
    first = generate_candidates(
        experiment,
        vulnerability.id,
        trigger.id,
        "platform:test-cuda",
        hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
    )
    second = generate_candidates(
        experiment,
        vulnerability.id,
        trigger.id,
        "platform:test-cuda",
        hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
    )
    assert [candidate.id for candidate, _ in first] == [candidate.id for candidate, _ in second]
    assert len({candidate.id for candidate, _ in first}) == 8
    assert len(trigger.constraints) == 2


def test_experiment_rejects_unbounded_or_unreviewed_parameter_range() -> None:
    experiment = load_experiment(SPEC_PATH)
    data = experiment.model_dump()
    data["allowed_extra_elements"] = (0, 1, 4096)
    with pytest.raises(ValidationError):
        CudaExperiment.model_validate(data)
    data = experiment.model_dump()
    data["limits"]["max_parallel_runs"] = 2
    with pytest.raises(ValidationError):
        CudaExperiment.model_validate(data)


def test_memcheck_oracle_requires_diagnostic_kernel_source_and_allocation() -> None:
    oracle = load_experiment(SPEC_PATH).target_error_oracle
    assert target_diagnostic_count(MATCHING_REPORT, oracle) == 1
    assert target_diagnostic_count(MATCHING_REPORT + MATCHING_REPORT, oracle) == 2
    assert sanitizer_error_count(MATCHING_REPORT + "========= ERROR SUMMARY: 3 errors\n") == 3


@pytest.mark.parametrize(
    "report",
    [
        MATCHING_REPORT.replace("Invalid __global__ write", "Invalid __global__ read"),
        MATCHING_REPORT.replace("bounded_write_kernel", "unrelated_kernel"),
        MATCHING_REPORT.replace("oob_write.cu:27", "other.cu:27"),
        MATCHING_REPORT.replace("size 128 bytes", "size 256 bytes"),
        MATCHING_REPORT.replace("is out of bounds", "is within bounds"),
        "========= Program hit cudaErrorLaunchFailure (error 719)\n",
    ],
)
def test_unrelated_sanitizer_and_cuda_api_errors_do_not_match_oracle(report: str) -> None:
    oracle = load_experiment(SPEC_PATH).target_error_oracle
    assert target_diagnostic_count(report, oracle) == 0


def test_oracle_schema_rejects_silent_semantic_substitution() -> None:
    data = load_experiment(SPEC_PATH).target_error_oracle.model_dump()
    data["allocation_bytes"] = 256
    with pytest.raises(ValidationError):
        CudaOracle.model_validate(data)


def test_tool_version_and_application_exit_policy_are_explicit() -> None:
    version_output = (
        "NVIDIA (R) Compute Sanitizer\n"
        "Copyright (c) NVIDIA Corporation\n"
        "Version 2025.1.0.0 (build 35583870) (public-release)\n"
    )
    assert parse_sanitizer_version(version_output) == "2025.1.0.0"
    assert _application_exit(86, 3, 0, False) is None
    assert _application_exit(86, 3, 5, False) is None
    assert _application_exit(0, 0, 0, False) == 0
    assert _application_exit(86, 0, 0, False) is None
    assert _application_exit(1, 0, 0, False) is None
    assert _application_exit(86, 3, 0, True) is None


def test_cuda_adapter_has_the_shared_platform_protocol_shape() -> None:
    assert issubclass(CudaAdapter, PlatformAdapter)
