#!/usr/bin/env python3
"""Run the bounded CUDA-P1 experiment only on the scoped WSL GTX 1660 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from heteropoclab.adapters.cuda import (
    CudaAdapter,
    CudaTrial,
    generate_candidates,
    load_experiment,
    make_vulnerability_and_trigger,
    probe_cuda_environment,
    write_private_file,
)
from heteropoclab.core import FactStatus
from heteropoclab.domain import (
    ExecutionBudget,
    ExecutionIntent,
    ExecutionRequest,
    InputEvent,
    VerificationRequirement,
    VerificationVerdict,
)
from heteropoclab.platforms import PlatformRequirement

ROOT = Path(__file__).resolve().parents[1]
SPEC_DEFAULT = ROOT / "examples/cuda_p1/experiment.json"
SOURCE_DEFAULT = ROOT / "examples/cuda_p1/oob_write.cu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_one(adapter: CudaAdapter, candidate, trial: CudaTrial, prepared, budget, trigger):
    run_id = "gpu-p1-" + uuid4().hex
    request = ExecutionRequest(
        run_id=run_id,
        candidate_id=candidate.id,
        platform_id=prepared.platform_id,
        budget=budget,
        intent=ExecutionIntent.PREPARE_AND_RUN,
        verification_requirement=VerificationRequirement.CONDITION_LINKED_RUNTIME,
    )
    receipt = adapter.inject_input(
        prepared,
        request,
        InputEvent(
            run_id=run_id,
            name="kernel.arguments",
            data=candidate.payload.encode("utf-8"),
        ),
    )
    if not receipt.accepted or receipt.evidence is None:
        raise RuntimeError(f"input was not accepted for {candidate.id}")
    execution = adapter.execute(prepared, request)
    evidence = adapter.collect_raw_evidence(execution)
    verification = adapter.verify(candidate, trigger, execution)
    metadata_raw = next(item for item in evidence if item.reference.kind.value == "RUNTIME_LOG")
    metadata = json.loads(metadata_raw.content)
    target_expected = (
        trial.variant == "vulnerable" and trial.logical_count == 32 and trial.extra_elements >= 1
    )
    target_observed = metadata["target_oracle_match_count"] > 0
    expected_verdict = (
        VerificationVerdict.VERIFIED if target_expected else VerificationVerdict.NOT_VERIFIED
    )
    tool_exit = metadata["compute_sanitizer_exit_code"]
    app_exit = metadata["application_exit_code"]
    expected_termination = "ERROR_EXIT" if target_expected else "NORMAL"
    expected_supervisor_exit = 86 if target_expected else 0
    expected_application_exit = None if target_expected else 0
    expected_exit_intent = 5 if target_expected else 0
    run_ok = (
        execution.completed
        and execution.termination.value == expected_termination
        and execution.exit_code == expected_supervisor_exit
        and app_exit == expected_application_exit
        and metadata["application_exit_intent_marker"] == expected_exit_intent
        and target_observed == target_expected
        and tool_exit == expected_supervisor_exit
        and verification.verdict is expected_verdict
        and not metadata["output_budget_exceeded"]
    )
    stdout_raw = next(item for item in evidence if item.reference.kind.value == "STDOUT")
    stderr_raw = next(item for item in evidence if item.reference.kind.value == "STDERR")
    host_text = stdout_raw.content.decode("utf-8", errors="replace") + stderr_raw.content.decode(
        "utf-8", errors="replace"
    )
    cpu_match = re.search(r"(?m)^APP_CPU_COMPARE_MISMATCHES=(\d+)\s*$", host_text)
    cpu_mismatches = int(cpu_match.group(1)) if cpu_match else None
    if not target_expected and cpu_mismatches != 0:
        run_ok = False
    return {
        "run_id": run_id,
        "candidate_id": candidate.id,
        "case_id": trial.case_id,
        "variant": trial.variant,
        "input": {
            "capacity": 32,
            "logical_count": trial.logical_count,
            "extra_elements": trial.extra_elements,
            "trigger_input_predicate": trial.logical_count == 32 and trial.extra_elements >= 1,
        },
        "expected_target_error": target_expected,
        "target_error_match_count": metadata["target_oracle_match_count"],
        "compute_sanitizer_exit_code": tool_exit,
        "compute_sanitizer_error_summary_count": metadata["compute_sanitizer_error_summary_count"],
        "nvcc_exit_code": metadata["nvcc_exit_code"],
        "application_exit_code": app_exit,
        "application_exit_intent_marker": metadata["application_exit_intent_marker"],
        "application_exit_basis": metadata["application_exit_basis"],
        "host_supervisor_exit_code": metadata["host_supervisor_exit_code"],
        "execution_record_exit_code": execution.exit_code,
        "cuda_api_statuses": metadata["cuda_api_statuses"],
        "termination": execution.termination.value,
        "cpu_comparison_mismatches": cpu_mismatches,
        "verification_verdict": verification.verdict.value,
        "verification_reason": verification.reason,
        "evidence": [
            {
                "kind": item.reference.kind.value,
                "locator": item.reference.locator,
                "sha256": item.reference.sha256,
                "byte_length": item.reference.byte_length,
            }
            for item in evidence
        ],
        "hashes": metadata["hashes"],
        "passed": run_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    args = parser.parse_args()
    os.umask(0o077)

    experiment = load_experiment(args.spec)
    source_sha = sha256(SOURCE_DEFAULT)
    probe = probe_cuda_environment()
    output_parent = ROOT / "output/gpu-p1"
    output_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_parent, 0o700)
    label = datetime.now(timezone.utc).strftime("acceptance-%Y%m%dT%H%M%SZ")
    label += "-" + uuid4().hex[:8]
    output_dir = output_parent / label
    output_dir.mkdir(mode=0o700)
    os.chmod(output_dir, 0o700)

    write_private_file(
        output_dir / "environment.json",
        (probe.environment.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    write_private_file(
        output_dir / "version-probes.json",
        (
            json.dumps(
                {
                    "exit_codes": probe.probe_exit_codes,
                    "stdout": probe.raw_outputs,
                    "not_checked": probe.not_checked,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
    capability_probe = CudaAdapter(probe, experiment, SOURCE_DEFAULT, output_dir)
    capability = capability_probe.query_capability()
    if (
        capability.execution_mode.value is not None
        and capability.execution_mode.value.value != "REAL_HARDWARE"
    ):
        raise RuntimeError("adapter did not establish real-hardware execution mode")
    vulnerability, trigger = make_vulnerability_and_trigger(capability.id, source_sha)
    adapter = capability_probe
    adapter.bind_trigger(trigger)
    candidates = generate_candidates(
        experiment, vulnerability.id, trigger.id, capability.id, source_sha
    )
    budget = ExecutionBudget(
        timeout_seconds=float(experiment.limits.run_timeout_seconds),
        max_memory_bytes=experiment.limits.max_device_allocation_bytes,
        max_output_bytes=65536,
        max_input_bytes=256,
    )

    prepared_by_id = {}
    failures = []
    for candidate, trial in candidates:
        prepared_result = adapter.prepare(candidate, PlatformRequirement(), budget)
        if prepared_result.status.value != "READY" or prepared_result.prepared is None:
            failures.append(
                {
                    "candidate_id": candidate.id,
                    "case_id": trial.case_id,
                    "variant": trial.variant,
                    "phase": "compile",
                    "reason": prepared_result.reason,
                }
            )
        else:
            prepared_by_id[candidate.id] = (candidate, trial, prepared_result.prepared)

    run_results = []
    candidate_run_ids = {}
    for candidate, trial in candidates:
        entry = prepared_by_id.get(candidate.id)
        if entry is None:
            continue
        try:
            result = run_one(adapter, candidate, trial, entry[2], budget, trigger)
        except Exception as exc:
            failures.append(
                {
                    "candidate_id": candidate.id,
                    "case_id": trial.case_id,
                    "variant": trial.variant,
                    "phase": "execute_or_verify",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        run_results.append(result)
        candidate_run_ids[(trial.case_id, trial.variant)] = result["run_id"]

    repeat = next(
        (
            entry
            for entry in prepared_by_id.values()
            if entry[1].case_id == experiment.repeat_case_id and entry[1].variant == "vulnerable"
        ),
        None,
    )
    if repeat is None:
        failures.append(
            {"phase": "repeat", "reason": "vulnerable repeat candidate did not compile"}
        )
    else:
        try:
            repeated = run_one(adapter, repeat[0], repeat[1], repeat[2], budget, trigger)
            repeated["repeat"] = True
            run_results.append(repeated)
        except Exception as exc:
            failures.append({"phase": "repeat", "reason": f"{type(exc).__name__}: {exc}"})

    matrix = run_results[:-1] if run_results and run_results[-1].get("repeat") else run_results
    positive_candidate_ids = sorted(
        {
            item["candidate_id"]
            for item in matrix
            if item["expected_target_error"] and item["target_error_match_count"] > 0
        }
    )
    repeat_result = run_results[-1] if run_results and run_results[-1].get("repeat") else None
    reproducible = bool(
        repeat_result
        and repeat_result["passed"]
        and repeat_result["candidate_id"] in positive_candidate_ids
        and repeat_result["hashes"]["binary_sha256"]
        == next(item for item in matrix if item["candidate_id"] == repeat_result["candidate_id"])[
            "hashes"
        ]["binary_sha256"]
    )
    passed = (
        not failures
        and len(candidates) == experiment.limits.max_candidates
        and len(prepared_by_id) == len(candidates)
        and len(run_results) == experiment.limits.max_gpu_executions
        and all(item["passed"] for item in run_results)
        and len(positive_candidate_ids) == 2
        and reproducible
    )
    failed_candidate_ids = sorted(
        {item["candidate_id"] for item in run_results if not item["passed"]}
        | {item["candidate_id"] for item in failures if item.get("candidate_id") is not None}
    )
    report = {
        "schema_version": "1.0",
        "experiment_id": experiment.experiment_id,
        "notice": experiment.notice,
        "status": "PASSED" if passed else "FAILED",
        "scope": {
            "gpu": probe.environment.gpu_name.value,
            "compute_capability": probe.environment.compute_capability.value,
            "driver": probe.environment.driver_version.value,
            "toolkit_release": probe.environment.toolkit_release.value,
            "nvcc": probe.environment.nvcc_version.value,
            "compute_sanitizer": probe.environment.sanitizer_version.value,
            "driver_cuda_compatibility": (
                probe.environment.driver_cuda_compatibility.value
                if probe.environment.driver_cuda_compatibility.status is FactStatus.OBSERVED
                else "NOT_CHECKED"
            ),
            "driver_cuda_umd_version": (
                probe.environment.driver_cuda_umd_version.value
                if probe.environment.driver_cuda_umd_version.status is FactStatus.OBSERVED
                else "NOT_CHECKED"
            ),
            "python_user_non_root": os.geteuid() != 0,
            "run_mode": "WSL2; serialized; Compute Sanitizer wraps every GPU execution",
        },
        "counts": {
            "deterministically_generated_candidates": len(candidates),
            "successful_compiles": len(prepared_by_id),
            "matrix_gpu_executions": len(matrix),
            "repeat_gpu_executions": 1 if repeat_result else 0,
            "total_gpu_executions": len(run_results),
            "target_error_hit_trials": sum(
                1 for item in run_results if item["target_error_match_count"] > 0
            ),
            "target_error_miss_trials": sum(
                1 for item in run_results if item["target_error_match_count"] == 0
            ),
            "distinct_verified_candidates": len(positive_candidate_ids),
        },
        "reproducibility": {
            "repeat_case": experiment.repeat_case_id,
            "same_candidate_and_binary_retested": reproducible,
            "first_run_id": next(
                (
                    item["run_id"]
                    for item in matrix
                    if item["candidate_id"] == (repeat_result or {}).get("candidate_id")
                ),
                None,
            ),
            "repeat_run_id": repeat_result["run_id"] if repeat_result else None,
        },
        "successful_candidates": positive_candidate_ids,
        "failed_candidates": sorted(
            {item["candidate_id"] for item in run_results if not item["passed"]}
            | {item["candidate_id"] for item in failures if item.get("candidate_id") is not None}
        ),
        "candidates": [
            {
                "candidate_id": candidate.id,
                "case_id": trial.case_id,
                "variant": trial.variant,
                "candidate_outcome": (
                    "FAILED"
                    if candidate.id in failed_candidate_ids
                    else "TARGET_VERIFIED"
                    if candidate.id in positive_candidate_ids
                    else "NO_TARGET_DIAGNOSTIC"
                ),
                "nvcc_exit_code": next(
                    (
                        item["nvcc_exit_code"]
                        for item in run_results
                        if item["candidate_id"] == candidate.id
                    ),
                    None,
                ),
                "parameters": {
                    "logical_count": trial.logical_count,
                    "extra_elements": trial.extra_elements,
                },
                "run_ids": [
                    item["run_id"] for item in run_results if item["candidate_id"] == candidate.id
                ],
            }
            for candidate, trial in candidates
        ],
        "runs": run_results,
        "failures": failures,
        "limitations": [
            "Self-built bounded CUDA defect; no historical CVE or third-party target.",
            "Evidence applies only to this WSL2 GTX 1660, driver, toolkit, binary, "
            "and memcheck run.",
            "Sanitizer zero-error controls do not establish coverage of other defect classes.",
            "The triggered CUDA context error skips CPU/GPU value comparison "
            "for positive OOB runs.",
            "This is a deterministic parameter sweep, not a general PoC-generation system.",
        ],
        "artifact_directory": "output/gpu-p1/" + label,
        "fixture_source_sha256": source_sha,
        "spec_sha256": sha256(args.spec),
    }
    report_bytes = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_private_file(output_dir / "report.json", report_bytes)
    console_summary = {
        "status": report["status"],
        "artifact_directory": report["artifact_directory"],
        "scope": report["scope"],
        "counts": report["counts"],
        "reproducibility": report["reproducibility"],
        "runs": [
            {
                "case_id": item["case_id"],
                "variant": item["variant"],
                "run_id": item["run_id"],
                "sanitizer_exit": item["compute_sanitizer_exit_code"],
                "sanitizer_errors": item["compute_sanitizer_error_summary_count"],
                "target_matches": item["target_error_match_count"],
                "application_exit": item["application_exit_code"],
                "termination": item["termination"],
                "verdict": item["verification_verdict"],
                "passed": item["passed"],
            }
            for item in run_results
        ],
        "report": "report.json",
    }
    print(json.dumps(console_summary, sort_keys=True, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
