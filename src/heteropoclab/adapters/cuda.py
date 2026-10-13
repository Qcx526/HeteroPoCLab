"""Bounded CUDA-P1 fixture adapter; only the reviewed local CUDA source is compiled."""

from __future__ import annotations

import hashlib
import json
import os
import platform as host_platform
import re
import shutil
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar
from uuid import uuid4

from pydantic import Field, model_validator

from ..core import Fact, FactStatus, NonEmptyStr, SourceKind, StrictFrozenModel, canonical_json
from ..domain import (
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
    SourceTrust,
    TerminationStatus,
    TriggerConstraint,
    TriggerSpec,
    VerificationRequirement,
    VerificationResult,
    VerificationVerdict,
    VulnerabilitySpec,
)
from ..domain import (
    verify as verify_result,
)
from ..platforms import (
    CompatibilityDecision,
    CompatibilityStatus,
    ExecutionMode,
    PlatformCapability,
    PlatformRequirement,
    negotiate,
)
from ..protocol import (
    AdapterContractError,
    InputReceipt,
    PreparationResult,
    PreparationStatus,
    PreparedRun,
)

T = TypeVar("T")
Variant = Literal["vulnerable", "fixed"]


class CudaCase(StrictFrozenModel):
    case_id: NonEmptyStr
    logical_count: int = Field(ge=1, le=32)
    extra_elements: int = Field(ge=0, le=2)


class CudaLimits(StrictFrozenModel):
    max_candidates: int = Field(ge=1, le=8)
    max_gpu_executions: int = Field(ge=1, le=9)
    max_parallel_runs: Literal[1]
    max_device_allocation_bytes: Literal[128]
    compile_timeout_seconds: int = Field(ge=1, le=60)
    run_timeout_seconds: int = Field(ge=1, le=30)


class CudaOracle(StrictFrozenModel):
    tool: Literal["compute-sanitizer"]
    tool_name: Literal["memcheck"]
    diagnostic: Literal["Invalid __global__ write of size 4 bytes"]
    kernel: Literal["bounded_write_kernel"]
    source_filename: Literal["oob_write.cu"]
    allocation_bytes: Literal[128]
    allocation_relation: Literal["address after nearest allocation"]
    condition: Literal["logical_count == capacity and extra_elements >= 1"]


class CudaExperiment(StrictFrozenModel):
    experiment_id: NonEmptyStr
    title: NonEmptyStr
    notice: NonEmptyStr
    capacity: Literal[32]
    thread_block_size: Literal[32]
    allowed_extra_elements: tuple[int, ...]
    cases: tuple[CudaCase, ...]
    variants: tuple[Variant, ...]
    repeat_case_id: NonEmptyStr
    limits: CudaLimits
    target_error_oracle: CudaOracle

    @model_validator(mode="after")
    def validate_bounded_fixture(self) -> CudaExperiment:
        case_ids = tuple(case.case_id for case in self.cases)
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("case IDs must be nonempty and unique")
        if tuple(sorted(set(self.allowed_extra_elements))) != self.allowed_extra_elements:
            raise ValueError("allowed extra-element values must be sorted and unique")
        if any(value not in (0, 1, 2) for value in self.allowed_extra_elements):
            raise ValueError("only the reviewed fixture's 0..2 parameter range is allowed")
        if any(case.extra_elements not in self.allowed_extra_elements for case in self.cases):
            raise ValueError("every case must be within the allowed parameter range")
        if self.variants != ("vulnerable", "fixed"):
            raise ValueError("both vulnerable and fixed variants are required in that order")
        if self.repeat_case_id not in case_ids:
            raise ValueError("repeat_case_id must reference a defined case")
        if len(self.cases) * len(self.variants) > self.limits.max_candidates:
            raise ValueError("candidate matrix exceeds its declared limit")
        if len(self.cases) * len(self.variants) + 1 > self.limits.max_gpu_executions:
            raise ValueError("matrix plus repeat exceeds the declared execution limit")
        if self.capacity * 4 != self.limits.max_device_allocation_bytes:
            raise ValueError("fixture allocation differs from the strict memory limit")
        if not self.notice.startswith("Self-built fixture; not a CVE"):
            raise ValueError("fixture must identify itself as self-built and not a CVE")
        return self


class CudaEnvironment(StrictFrozenModel):
    gpu_name: Fact[str]
    driver_version: Fact[str]
    memory_mib: Fact[int]
    compute_capability: Fact[str]
    nvcc_version: Fact[str]
    toolkit_release: Fact[str]
    sanitizer_version: Fact[str]
    driver_cuda_compatibility: Fact[str]
    driver_cuda_umd_version: Fact[str]
    host_architecture: Fact[str]
    host_endian: Fact[str]
    host_bits: Fact[int]
    kernel_release: Fact[str]


@dataclass(frozen=True)
class CudaProbe:
    environment: CudaEnvironment
    nvcc_path: Path
    sanitizer_path: Path
    nvidia_smi_path: Path
    raw_outputs: dict[str, str] = field(repr=False)
    probe_exit_codes: dict[str, int]
    not_checked: dict[str, str]


@dataclass(frozen=True)
class CudaTrial:
    case_id: str
    logical_count: int
    extra_elements: int
    variant: Variant


@dataclass
class _Run:
    request: ExecutionRequest
    directory: Path
    input_bytes: bytes | None = None
    input_evidence: RawEvidence | None = None
    execution: ExecutionRecord | None = None
    evidence: tuple[RawEvidence, ...] = ()
    observations: tuple[Observation, ...] = ()
    telemetry: dict[str, object] = field(default_factory=dict)


@dataclass
class _Prepared:
    prepared: PreparedRun
    candidate: PoCCandidate
    trial: CudaTrial
    binary_path: Path
    binary_sha256: str
    source_sha256: str
    build_stdout: bytes
    build_stderr: bytes
    build_exit_code: int
    runs: dict[str, _Run] = field(default_factory=dict)


def _observed(value: T, provenance: str) -> Fact[T]:
    return Fact(
        value=value,
        status=FactStatus.OBSERVED,
        source=SourceKind.REAL_EXECUTION,
        provenance=provenance,
    )


def _probe(command: list[str], timeout: int = 10) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, capture_output=True, check=False, timeout=timeout)


def _check_probe(label: str, result: subprocess.CompletedProcess[bytes]) -> bytes:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{label} failed with exit {result.returncode}: {detail}")
    return result.stdout + result.stderr


def probe_cuda_environment() -> CudaProbe:
    """Read local CUDA facts without installing or configuring host software."""
    nvcc = Path(shutil.which("nvcc") or "/usr/local/cuda-12.8/bin/nvcc")
    sanitizer = Path(
        shutil.which("compute-sanitizer") or "/usr/local/cuda-12.8/bin/compute-sanitizer"
    )
    nvidia_smi = Path(shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi")
    for label, path in (
        ("nvcc", nvcc),
        ("compute-sanitizer", sanitizer),
        ("nvidia-smi", nvidia_smi),
    ):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"required CUDA-P1 tool is unavailable: {label}")

    query = _probe(
        [
            str(nvidia_smi),
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    query_text = (
        _check_probe("nvidia-smi GPU query", query).decode("utf-8", errors="replace").strip()
    )
    rows = [line.strip() for line in query_text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("CUDA-P1 requires exactly one visible GPU; probe was ambiguous")
    fields = [value.strip() for value in rows[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError("nvidia-smi returned an unexpected GPU query format")
    gpu_name, driver, memory_text, compute_cap = fields
    if "GTX 1660" not in gpu_name or compute_cap != "7.5":
        raise RuntimeError(
            f"CUDA-P1 is scoped to GTX 1660 / CC 7.5; observed {gpu_name} / {compute_cap}"
        )
    memory_match = re.fullmatch(r"(\d+)\s*MiB", memory_text)
    if memory_match is None:
        raise RuntimeError("nvidia-smi memory value was not parseable")

    nvcc_result = _probe([str(nvcc), "--version"])
    nvcc_text = _check_probe("nvcc version", nvcc_result).decode("utf-8", errors="replace")
    nvcc_match = re.search(r"release\s+(\d+\.\d+),\s+V(\d+\.\d+\.\d+)", nvcc_text)
    if nvcc_match is None:
        raise RuntimeError("nvcc version output was not parseable")
    toolkit_release, nvcc_version = nvcc_match.groups()

    sanitizer_result = _probe([str(sanitizer), "--version"])
    sanitizer_text = _check_probe("Compute Sanitizer version", sanitizer_result).decode(
        "utf-8", errors="replace"
    )
    sanitizer_version = parse_sanitizer_version(sanitizer_text)
    if sanitizer_version is None:
        raise RuntimeError("Compute Sanitizer version output was not parseable")

    # The standard GPU query avoids collecting the full process table.
    compatibility_fact: Fact[str] = Fact(status=FactStatus.NOT_CHECKED)
    umd_fact: Fact[str] = Fact(status=FactStatus.NOT_CHECKED)
    kernel_release = host_platform.release()
    if "microsoft" not in kernel_release.lower() and "wsl" not in kernel_release.lower():
        raise RuntimeError("CUDA-P1 requires a positively identified WSL2 Linux kernel")
    basis = "observed using bounded local executable probes"
    env = CudaEnvironment(
        gpu_name=_observed(gpu_name, basis),
        driver_version=_observed(driver, basis),
        memory_mib=_observed(int(memory_match.group(1)), basis),
        compute_capability=_observed(compute_cap, basis),
        nvcc_version=_observed(nvcc_version, basis),
        toolkit_release=_observed(toolkit_release, basis),
        sanitizer_version=_observed(sanitizer_version, basis),
        driver_cuda_compatibility=compatibility_fact,
        driver_cuda_umd_version=umd_fact,
        host_architecture=_observed(host_platform.machine(), basis),
        host_endian=_observed(sys.byteorder, basis),
        host_bits=_observed(8 * struct.calcsize("P"), basis),
        kernel_release=_observed(kernel_release, basis),
    )
    raw_outputs = {
        "nvidia_smi_query": query_text + "\n",
        "nvcc_version": nvcc_text,
        "compute_sanitizer_version": sanitizer_text,
    }
    exit_codes = {
        "nvidia_smi_gpu_query": query.returncode,
        "nvcc_version": nvcc_result.returncode,
        "compute_sanitizer_version": sanitizer_result.returncode,
    }
    not_checked = {
        "driver_cuda_compatibility": (
            "Not queried because the privacy-limited nvidia-smi GPU query omits this field."
        ),
        "driver_cuda_umd_version": (
            "Not queried because the privacy-limited nvidia-smi GPU query omits this field."
        ),
    }
    return CudaProbe(env, nvcc, sanitizer, nvidia_smi, raw_outputs, exit_codes, not_checked)


def parse_sanitizer_version(text: str) -> str | None:
    match = re.search(r"(?m)^Version\s+([\w.\-]+)", text)
    return match.group(1) if match else None


def load_experiment(path: Path) -> CudaExperiment:
    return CudaExperiment.model_validate_json(path.read_text(encoding="utf-8"))


def deterministic_trials(experiment: CudaExperiment) -> tuple[CudaTrial, ...]:
    return tuple(
        CudaTrial(case.case_id, case.logical_count, case.extra_elements, variant)
        for case in experiment.cases
        for variant in experiment.variants
    )


def _trial_payload(trial: CudaTrial) -> str:
    return canonical_json(
        {
            "schema_version": "1.0",
            "case_id": trial.case_id,
            "logical_count": trial.logical_count,
            "extra_elements": trial.extra_elements,
            "variant": trial.variant,
        }
    )


def generate_candidates(
    experiment: CudaExperiment,
    vulnerability_id: str,
    trigger_id: str,
    platform_id: str,
    fixture_sha256: str,
) -> tuple[tuple[PoCCandidate, CudaTrial], ...]:
    results: list[tuple[PoCCandidate, CudaTrial]] = []
    for trial in deterministic_trials(experiment):
        payload = _trial_payload(trial)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        candidate = PoCCandidate(
            vulnerability_id=vulnerability_id,
            trigger_id=trigger_id,
            platform_id=platform_id,
            language="CUDA C++",
            description=f"Deterministic {trial.variant} {trial.case_id} parameter candidate.",
            expected_behavior=(
                "Memcheck reports a bounded-write out-of-bounds store"
                if trial.variant == "vulnerable"
                else "Bounds guard prevents an out-of-bounds store"
            ),
            payload=payload,
            provenance=Provenance(
                source=SourceKind.STATIC_ANALYSIS,
                method="deterministic bounded parameter generator",
                scope=platform_id,
                source_sha256=digest,
            ),
            platform_requirement=PlatformRequirement(platform_id=platform_id),
        )
        if digest != candidate.source_sha256 or fixture_sha256 == "0" * 64:
            raise ValueError("candidate or fixture identity hash is invalid")
        results.append((candidate, trial))
    if len(results) > experiment.limits.max_candidates:
        raise ValueError("generated candidates exceed the experiment limit")
    return tuple(results)


def target_diagnostic_count(log: str, oracle: CudaOracle) -> int:
    """Count only complete reports matching every target-specific oracle field."""
    starts = list(re.finditer(r"(?m)^========= Invalid __global__ write of size 4 bytes\s*$", log))
    count = 0
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(log)
        block = log[start.start() : end]
        if (
            re.search(
                rf"(?m)^=========\s+at .*{re.escape(oracle.kernel)}.*"
                rf"{re.escape(oracle.source_filename)}:\d+\s*$",
                block,
            )
            and re.search(r"(?m)^=========\s+Address .* is out of bounds\s*$", block)
            and re.search(
                rf"(?m)^=========\s+and is \d+ bytes after the nearest allocation "
                rf"at .* of size {oracle.allocation_bytes} bytes\s*$",
                block,
            )
        ):
            count += 1
    return count


def sanitizer_error_count(log: str) -> int | None:
    match = re.search(r"(?m)^========= ERROR SUMMARY:\s+(\d+) errors?\s*$", log)
    return int(match.group(1)) if match else None


def _application_exit(
    tool_exit: int | None, error_count: int | None, exit_intent: int | None, timed_out: bool
) -> int | None:
    if timed_out or tool_exit is None or error_count is None or exit_intent is None:
        return None
    # The wrapper's configured error-exitcode can mask a nonzero child status.
    # Only infer a successful child exit when both tool and application checks pass.
    if tool_exit == 0 and error_count == 0 and exit_intent == 0:
        return 0
    return None


def _api_statuses(text: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, code, error_name in re.findall(
        r"(?m)^(CUDA_[A-Z0-9_]+|APP_EXPECTED_SYNC_ERROR)=(\d+):([A-Za-z0-9_]+)$", text
    ):
        result[name] = {"code": int(code), "name": error_name}
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_private(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_bytes(content)
    os.chmod(path, 0o600)


class CudaAdapter:
    """CUDA-P1 implementation of the shared PlatformAdapter protocol."""

    def __init__(
        self, probe: CudaProbe, experiment: CudaExperiment, source_path: Path, output_root: Path
    ) -> None:
        self._probe = probe
        self._experiment = experiment
        self._source_path = source_path.resolve()
        self._repo_root = self._source_path.parents[2]
        self._build_root = Path("/tmp/heteropoclab-gpu-p1-build-v1")
        self._output_root = output_root.resolve()
        self._source_sha256 = _file_sha256(self._source_path)
        self._capability = self._make_capability(probe.environment)
        self._preparations: dict[str, _Prepared] = {}
        self._runs: dict[str, _Run] = {}
        self._trigger: TriggerSpec | None = None

    @staticmethod
    def _make_capability(environment: CudaEnvironment) -> PlatformCapability:
        required = (
            environment.gpu_name,
            environment.driver_version,
            environment.memory_mib,
            environment.compute_capability,
            environment.nvcc_version,
            environment.toolkit_release,
            environment.sanitizer_version,
        )
        if any(fact.status is not FactStatus.OBSERVED for fact in required):
            raise AdapterContractError("CUDA-P1 requires observed GPU and tool version facts")
        compute_capability = environment.compute_capability.value
        if compute_capability is None:
            raise AdapterContractError("compute capability is missing despite observed status")
        profile = "sm_" + compute_capability.replace(".", "")
        gpu_name = environment.gpu_name.value
        memory_mib = environment.memory_mib.value
        kernel_release = environment.kernel_release.value
        umd_version = environment.driver_cuda_umd_version.value
        if gpu_name is None or memory_mib is None or kernel_release is None:
            raise AdapterContractError("device identity facts are incomplete")
        device_slug = re.sub(r"[^a-z0-9]+", "-", gpu_name.lower()).strip("-")
        return PlatformCapability(
            label=(
                f"{device_slug}.{memory_mib}mib.cuda.{compute_capability}."
                f"driver-{environment.driver_version.value}."
                f"umd-{umd_version or 'unknown'}.toolkit-{environment.toolkit_release.value}"
            ),
            category=_observed("GPU", "nvidia-smi and CUDA execution probes"),
            isa=_observed("NVIDIA CUDA", "CUDA runtime and compiler probes"),
            isa_profile=_observed(profile, "nvidia-smi compute capability query"),
            os=_observed(f"Linux-WSL2/{kernel_release}", "observed WSL2 kernel release"),
            binary_format=_observed("ELF", "WSL2 Linux executable"),
            endian=environment.host_endian,
            bit_width=environment.host_bits,
            abi=_observed(f"{environment.host_architecture.value}-linux", "Linux host probe"),
            toolchain=_observed(f"nvcc-{environment.nvcc_version.value}", "nvcc --version"),
            runtime=_observed(
                f"CUDA-{environment.toolkit_release.value}/driver-{environment.driver_version.value}"
                f"/umd-{umd_version or 'unknown'}",
                "nvcc and nvidia-smi probes",
            ),
            execution_mode=_observed(ExecutionMode.REAL_HARDWARE, "WSL2 NVIDIA device query"),
            debug_capabilities=_observed(("compute-sanitizer.memcheck",), "installed tool probe"),
            observation_capabilities=_observed(
                ("cuda.api.status", "host.stdout", "compute-sanitizer.memcheck"),
                "CUDA-P1 adapter implementation",
            ),
            input_interfaces=_observed(("kernel.arguments",), "reviewed fixture interface"),
            evidence_capabilities=_observed(
                ("stdout", "stderr", "sanitizer.log", "build.log", "sha256"),
                "CUDA-P1 local evidence writer",
            ),
        )

    def query_capability(self) -> PlatformCapability:
        return PlatformCapability.model_validate(self._capability.model_dump())

    def negotiate(self, requirement: PlatformRequirement) -> CompatibilityDecision:
        return negotiate(self._capability, requirement)

    def prepare(
        self, candidate: PoCCandidate, requirement: PlatformRequirement, budget: ExecutionBudget
    ) -> PreparationResult:
        candidate = PoCCandidate.model_validate(candidate.model_dump())
        if self._trigger is not None and (
            candidate.trigger_id != self._trigger.id
            or candidate.vulnerability_id != self._trigger.vulnerability_id
        ):
            decision = CompatibilityDecision(
                platform_id=self._capability.id, status=CompatibilityStatus.UNSUPPORTED
            )
            return PreparationResult(
                status=PreparationStatus.REJECTED,
                decision=decision,
                reason=(
                    "Candidate vulnerability or trigger identity differs from the bound experiment."
                ),
            )
        requirement = PlatformRequirement.model_validate(requirement.model_dump())
        budget = ExecutionBudget.model_validate(budget.model_dump())
        if candidate.platform_id != self._capability.id:
            decision = CompatibilityDecision(
                platform_id=self._capability.id, status=CompatibilityStatus.UNSUPPORTED
            )
            return PreparationResult(
                status=PreparationStatus.REJECTED,
                decision=decision,
                reason="Candidate platform identity differs from the probed CUDA adapter.",
            )
        candidate_decision = self.negotiate(candidate.platform_requirement)
        effective = PlatformRequirement(
            platform_id=requirement.platform_id or self._capability.id,
            category=requirement.category,
            isa=requirement.isa,
            isa_profile=requirement.isa_profile,
            os=requirement.os,
            binary_format=requirement.binary_format,
            endian=requirement.endian,
            bit_width=requirement.bit_width,
            abi=requirement.abi,
            toolchain=requirement.toolchain,
            runtime=requirement.runtime,
            execution_mode=requirement.execution_mode,
            debug_capabilities=requirement.debug_capabilities,
            observation_capabilities=requirement.observation_capabilities,
            input_interfaces=requirement.input_interfaces,
            evidence_capabilities=requirement.evidence_capabilities,
        )
        decision = self.negotiate(effective)
        if not candidate_decision.can_execute or not decision.can_execute:
            return PreparationResult(
                status=PreparationStatus.REJECTED,
                decision=decision if not decision.can_execute else candidate_decision,
                reason="Candidate or requested platform is not exactly compatible.",
            )
        if (
            budget.max_memory_bytes != self._experiment.limits.max_device_allocation_bytes
            or budget.max_output_bytes > 65536
            or budget.max_input_bytes > 256
        ):
            return PreparationResult(
                status=PreparationStatus.FAILED,
                decision=decision,
                reason="Device allocation exceeds the declared device-memory budget.",
            )
        if budget.timeout_seconds > float(self._experiment.limits.run_timeout_seconds):
            return PreparationResult(
                status=PreparationStatus.REJECTED,
                decision=decision,
                reason="Request timeout exceeds the experiment hard limit.",
            )
        try:
            data = json.loads(candidate.payload)
            trial = CudaTrial(
                case_id=data["case_id"],
                logical_count=data["logical_count"],
                extra_elements=data["extra_elements"],
                variant=data["variant"],
            )
            declared = next(
                case for case in self._experiment.cases if case.case_id == trial.case_id
            )
            if (
                trial.logical_count != declared.logical_count
                or trial.extra_elements != declared.extra_elements
                or trial.variant not in self._experiment.variants
                or candidate.payload != _trial_payload(trial)
            ):
                raise ValueError("candidate does not match a declared deterministic trial")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, StopIteration) as exc:
            return PreparationResult(
                status=PreparationStatus.REJECTED,
                decision=CompatibilityDecision(
                    platform_id=self._capability.id, status=CompatibilityStatus.UNSUPPORTED
                ),
                reason=f"candidate payload is outside the declared search: {exc}",
            )

        artifact_dir = self._output_root / "artifacts" / candidate.id.replace(":", "_")
        artifact_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(artifact_dir, 0o700)
        binary = artifact_dir / f"{trial.variant}.out"
        self._build_root.mkdir(mode=0o700, parents=False, exist_ok=True)
        if self._build_root.is_symlink():
            raise AdapterContractError("private CUDA build directory cannot be a symbolic link")
        build_stat = self._build_root.stat()
        if (
            not stat.S_ISDIR(build_stat.st_mode)
            or build_stat.st_uid != os.geteuid()
            or stat.S_IMODE(build_stat.st_mode) & 0o077
        ):
            raise AdapterContractError(
                "private CUDA build directory must be owned by this user with mode 0700"
            )
        build_source = self._build_root / "oob_write.cu"
        if build_source.is_symlink():
            raise AdapterContractError("private CUDA source copy cannot be a symbolic link")
        if (
            not build_source.exists()
            or _file_sha256(build_source) != self._source_sha256
            or build_source.stat().st_uid != os.geteuid()
            or stat.S_IMODE(build_source.stat().st_mode) & 0o077
        ):
            build_source.write_bytes(self._source_path.read_bytes())
            os.chmod(build_source, 0o600)
        temporary_dir = self._build_root / "artifacts" / candidate.id.replace(":", "_")
        temporary_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if temporary_dir.is_symlink() or temporary_dir.stat().st_uid != os.geteuid():
            raise AdapterContractError("private CUDA candidate build directory is not trusted")
        os.chmod(temporary_dir, 0o700)
        temporary_binary = temporary_dir / f"{trial.variant}.out"
        temporary_binary.unlink(missing_ok=True)
        relative_binary = temporary_binary.relative_to(self._build_root).as_posix()
        command = [
            str(self._probe.nvcc_path),
            "--threads",
            "1",
            "-arch=sm_75",
            "-std=c++14",
            "-O2",
            "-lineinfo",
            "-Xcompiler=-Wall,-Wextra",
            f"-DGPU_P1_FIXED={1 if trial.variant == 'fixed' else 0}",
            "oob_write.cu",
            "-o",
            relative_binary,
        ]
        stable_compile_flags = [
            "--threads",
            "1",
            "-arch=sm_75",
            "-std=c++14",
            "-O2",
            "-lineinfo",
            "-Xcompiler=-Wall,-Wextra",
            f"-DGPU_P1_FIXED={1 if trial.variant == 'fixed' else 0}",
        ]
        build_env = os.environ.copy()
        build_env["OMP_NUM_THREADS"] = "1"
        try:
            build = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=self._experiment.limits.compile_timeout_seconds,
                env=build_env,
                cwd=self._build_root,
            )
        except subprocess.TimeoutExpired:
            return PreparationResult(
                status=PreparationStatus.FAILED,
                decision=decision,
                reason=f"nvcc exceeded {self._experiment.limits.compile_timeout_seconds}s.",
            )
        if build.returncode != 0 or not temporary_binary.is_file():
            return PreparationResult(
                status=PreparationStatus.FAILED,
                decision=decision,
                reason=(
                    f"nvcc failed with exit {build.returncode}; logs retained in ignored output."
                ),
            )
        shutil.copyfile(temporary_binary, binary)
        os.chmod(binary, 0o700)
        binary_hash = _file_sha256(binary)
        temporary_binary.unlink(missing_ok=True)
        content_id = (
            "prepared:"
            + hashlib.sha256(
                canonical_json(
                    {
                        "candidate_id": candidate.id,
                        "platform_id": self._capability.id,
                        "binary_sha256": binary_hash,
                        "compile_flags": stable_compile_flags,
                    }
                ).encode("utf-8")
            ).hexdigest()
        )
        prepared = PreparedRun(
            preparation_id="cuda-prep-" + uuid4().hex,
            content_id=content_id,
            candidate_id=candidate.id,
            platform_id=self._capability.id,
            requirement=effective,
            budget=budget,
        )
        self._preparations[prepared.preparation_id] = _Prepared(
            prepared,
            candidate,
            trial,
            binary,
            binary_hash,
            self._source_sha256,
            build.stdout,
            build.stderr,
            build.returncode,
        )
        return PreparationResult(
            status=PreparationStatus.READY,
            decision=decision,
            prepared=prepared,
            reason="Compiled reviewed CUDA-P1 source with explicit sm_75 target.",
        )

    def _bind(self, prepared: PreparedRun, request: ExecutionRequest) -> tuple[_Prepared, _Run]:
        request = ExecutionRequest.model_validate(request.model_dump())
        registration = self._preparations.get(prepared.preparation_id)
        if registration is None or registration.prepared != prepared:
            raise AdapterContractError("unknown or modified CUDA preparation")
        if request.intent is ExecutionIntent.PREPARE:
            raise AdapterContractError("PREPARE-only request cannot inject or execute")
        if (
            request.candidate_id != prepared.candidate_id
            or request.platform_id != prepared.platform_id
            or request.budget != prepared.budget
        ):
            raise AdapterContractError(
                "request candidate, platform, or budget differs from preparation"
            )
        existing = self._runs.get(request.run_id)
        if existing is not None:
            if existing.request != request or request.run_id not in registration.runs:
                raise AdapterContractError("run ID was reused with a different preparation")
            return registration, existing
        directory = self._output_root / "runs" / request.run_id
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(directory, 0o700)
        run = _Run(request, directory)
        registration.runs[request.run_id] = run
        self._runs[request.run_id] = run
        return registration, run

    def bind_trigger(self, trigger: TriggerSpec) -> None:
        if trigger.platform_id != self._capability.id or len(trigger.constraints) != 2:
            raise AdapterContractError("CUDA-P1 trigger must contain its two reviewed conditions")
        self._trigger = TriggerSpec.model_validate(trigger.model_dump())

    def _raw(
        self,
        registration: _Prepared,
        run: _Run,
        content: bytes,
        kind: EvidenceKind,
        method: str,
        locator: str,
    ) -> RawEvidence:
        provenance = Provenance(
            source=SourceKind.REAL_EXECUTION,
            method=method,
            scope=self._capability.id,
            source_sha256=registration.candidate.source_sha256,
            run_id=run.request.run_id,
        )
        reference = EvidenceReference(
            run_id=run.request.run_id,
            candidate_id=registration.candidate.id,
            platform_id=self._capability.id,
            kind=kind,
            sha256=_sha256(content),
            byte_length=len(content),
            provenance=provenance,
            locator=locator,
        )
        return RawEvidence(reference=reference, content=content)

    def inject_input(
        self, prepared: PreparedRun, request: ExecutionRequest, event: InputEvent
    ) -> InputReceipt:
        event = InputEvent.model_validate(event.model_dump())
        registration, run = self._bind(prepared, request)
        if run.execution is not None or run.input_evidence is not None:
            raise AdapterContractError("one input event is allowed before each execution")
        if event.run_id != request.run_id or event.name != "kernel.arguments":
            raise AdapterContractError("input must use this run's kernel.arguments interface")
        if len(event.data) > request.budget.max_input_bytes:
            raise AdapterContractError("input exceeds the configured input-byte limit")
        expected = _trial_payload(registration.trial).encode("utf-8")
        if event.data != expected:
            raise AdapterContractError("input differs from deterministic candidate parameters")
        _write_private(run.directory / "input.json", event.data + b"\n")
        run.input_bytes = event.data
        run.input_evidence = self._raw(
            registration,
            run,
            event.data,
            EvidenceKind.INPUT,
            "exact generated input accepted before execution",
            f"runs/{request.run_id}/input.json",
        )
        return InputReceipt(
            status=FactStatus.OBSERVED,
            accepted=True,
            evidence=run.input_evidence,
            reason="Input bytes were bound to the run before execution.",
        )

    def execute(self, prepared: PreparedRun, request: ExecutionRequest) -> ExecutionRecord:
        registration, run = self._bind(prepared, request)
        if run.execution is not None:
            raise AdapterContractError("run identity cannot be executed twice")
        if run.input_bytes is None or run.input_evidence is None:
            raise AdapterContractError("accepted input is required before execution")
        trial = registration.trial
        sanitizer_path = run.directory / "sanitizer.log"
        _write_private(sanitizer_path, b"")
        command = [
            str(self._probe.sanitizer_path),
            "--tool",
            "memcheck",
            "--error-exitcode",
            "86",
            "--check-exit-code",
            "yes",
            "--log-file",
            str(sanitizer_path),
            str(registration.binary_path),
            str(self._experiment.capacity),
            str(trial.logical_count),
            str(trial.extra_elements),
        ]
        run_env = os.environ.copy()
        run_env["CUDA_VISIBLE_DEVICES"] = "0"
        run_env["OMP_NUM_THREADS"] = "1"
        timeout = min(
            int(request.budget.timeout_seconds), self._experiment.limits.run_timeout_seconds
        )
        timed_out = False
        host_exit: int | None
        try:
            process = subprocess.run(
                command, capture_output=True, check=False, timeout=timeout, env=run_env
            )
            host_exit, stdout, stderr = process.returncode, process.stdout, process.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            host_exit = None
            stdout = exc.stdout if isinstance(exc.stdout, bytes) else b""
            stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
        _write_private(run.directory / "stdout.log", stdout)
        _write_private(run.directory / "stderr.log", stderr)
        sanitizer_bytes = sanitizer_path.read_bytes()
        sanitizer_text = sanitizer_bytes.decode("utf-8", errors="replace")
        host_text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
        summary_count = sanitizer_error_count(sanitizer_text)
        target_count = target_diagnostic_count(sanitizer_text, self._experiment.target_error_oracle)
        intent_match = re.search(r"(?m)^APP_EXIT_INTENT=(\d+)\s*$", host_text)
        exit_intent = int(intent_match.group(1)) if intent_match else None
        output_bytes = len(stdout) + len(stderr) + len(sanitizer_bytes)
        output_exceeded = output_bytes > request.budget.max_output_bytes
        application_exit = _application_exit(
            host_exit, summary_count, exit_intent, timed_out or output_exceeded
        )
        if timed_out:
            termination = TerminationStatus.TIMEOUT
        elif output_exceeded or host_exit is None:
            termination = TerminationStatus.UNKNOWN
        elif host_exit == 0:
            termination = TerminationStatus.NORMAL
        else:
            # ExecutionRecord preserves the directly observed Compute Sanitizer process
            # outcome. The inferred child application exit remains separate metadata.
            termination = TerminationStatus.ERROR_EXIT
        api_statuses = _api_statuses(host_text)
        metadata: dict[str, object] = {
            "schema_version": "1.0",
            "experiment_id": self._experiment.experiment_id,
            "candidate_id": registration.candidate.id,
            "run_id": request.run_id,
            "platform_id": self._capability.id,
            "variant": trial.variant,
            "case_id": trial.case_id,
            "input": {
                "capacity": self._experiment.capacity,
                "logical_count": trial.logical_count,
                "extra_elements": trial.extra_elements,
                "trigger_input_predicate": (
                    trial.logical_count == self._experiment.capacity and trial.extra_elements >= 1
                ),
            },
            "commands": {
                "compile": [
                    str(self._probe.nvcc_path),
                    "--threads",
                    "1",
                    "-arch=sm_75",
                    "-std=c++14",
                    "-O2",
                    "-lineinfo",
                    "-Xcompiler=-Wall,-Wextra",
                    f"-DGPU_P1_FIXED={1 if trial.variant == 'fixed' else 0}",
                    "oob_write.cu",
                    (
                        Path("artifacts")
                        / registration.candidate.id.replace(":", "_")
                        / f"{trial.variant}.out"
                    ).as_posix(),
                ],
                "execute": [
                    str(self._probe.sanitizer_path),
                    "--tool",
                    "memcheck",
                    "--error-exitcode",
                    "86",
                    "--check-exit-code",
                    "yes",
                    "--log-file",
                    sanitizer_path.relative_to(self._source_path.parents[2]).as_posix(),
                    registration.binary_path.relative_to(self._source_path.parents[2]).as_posix(),
                    str(self._experiment.capacity),
                    str(trial.logical_count),
                    str(trial.extra_elements),
                ],
                "compile_working_directory": str(self._build_root),
                "timeout_seconds": timeout,
            },
            "host_supervisor_exit_code": host_exit,
            "application_exit_code": application_exit,
            "application_exit_basis": (
                "inferred zero from clean wrapper exit, zero sanitizer errors, and fixture marker"
                if application_exit == 0
                else (
                    "UNKNOWN: configured Compute Sanitizer error-exitcode masks the child OS "
                    "status; APP_EXIT_INTENT is only the selected return-path marker"
                    if host_exit == 86 and summary_count is not None and summary_count > 0
                    else "UNKNOWN: wrapper status, summary, or marker ambiguous"
                )
            ),
            "application_exit_intent_marker": exit_intent,
            "compute_sanitizer_exit_code": host_exit,
            "nvcc_exit_code": registration.build_exit_code,
            "compute_sanitizer_error_summary_count": summary_count,
            "target_oracle_match_count": target_count,
            "target_oracle_matched": target_count > 0,
            "cuda_api_statuses": api_statuses,
            "timeout": timed_out,
            "output_byte_count": output_bytes,
            "output_budget_bytes": request.budget.max_output_bytes,
            "output_budget_exceeded": output_exceeded,
            "process_reaped": True,
            "termination": termination.value,
            "hashes": {
                "fixture_source_sha256": registration.source_sha256,
                "candidate_payload_sha256": registration.candidate.source_sha256,
                "binary_sha256": registration.binary_sha256,
                "input_sha256": _sha256(run.input_bytes or b""),
                "stdout_sha256": _sha256(stdout),
                "stderr_sha256": _sha256(stderr),
                "sanitizer_log_sha256": _sha256(sanitizer_bytes),
            },
        }
        metadata_bytes = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _write_private(run.directory / "run-metadata.json", metadata_bytes)
        execution = ExecutionRecord(
            run_id=request.run_id,
            candidate_id=registration.candidate.id,
            platform_id=self._capability.id,
            source=SourceKind.REAL_EXECUTION,
            scope=ExecutionScope.TARGET_RUNTIME,
            completed=True,
            termination=termination,
            exit_code=(
                host_exit
                if termination in (TerminationStatus.NORMAL, TerminationStatus.ERROR_EXIT)
                else None
            ),
        )
        run.execution = execution
        run.telemetry = metadata
        return execution

    def collect_raw_evidence(self, execution: ExecutionRecord) -> tuple[RawEvidence, ...]:
        run = self._runs.get(execution.run_id)
        if run is None or run.execution != execution:
            raise AdapterContractError("execution is not a completed CUDA-P1 run")
        registration = next(
            prepared
            for prepared in self._preparations.values()
            if execution.run_id in prepared.runs
        )
        files: tuple[tuple[str, bytes, EvidenceKind], ...] = (
            ("build.stdout", registration.build_stdout, EvidenceKind.BUILD_LOG),
            ("build.stderr", registration.build_stderr, EvidenceKind.BUILD_LOG),
            ("stdout.log", (run.directory / "stdout.log").read_bytes(), EvidenceKind.STDOUT),
            ("stderr.log", (run.directory / "stderr.log").read_bytes(), EvidenceKind.STDERR),
            (
                "sanitizer.log",
                (run.directory / "sanitizer.log").read_bytes(),
                EvidenceKind.SANITIZER_LOG,
            ),
            (
                "run-metadata.json",
                (run.directory / "run-metadata.json").read_bytes(),
                EvidenceKind.RUNTIME_LOG,
            ),
        )
        evidence = [run.input_evidence] if run.input_evidence is not None else []
        for name, content, kind in files:
            evidence.append(
                self._raw(
                    registration,
                    run,
                    content,
                    kind,
                    f"captured CUDA-P1 {name}",
                    f"runs/{execution.run_id}/{name}",
                )
            )
            if name.startswith("build."):
                _write_private(run.directory / name, content)
        run.evidence = tuple(item for item in evidence if item is not None)

        stdout = (run.directory / "stdout.log").read_text(encoding="utf-8", errors="replace")
        stderr = (run.directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
        output = stdout + "\n" + stderr
        params_match = (
            f"APP_VARIANT={registration.trial.variant}" in output
            and f"APP_CAPACITY={self._experiment.capacity}" in output
            and f"APP_LOGICAL_COUNT={registration.trial.logical_count}" in output
            and f"APP_EXTRA_ELEMENTS={registration.trial.extra_elements}" in output
        )
        parameter_evidence = tuple(
            raw.reference.id
            for raw in run.evidence
            if raw.reference.kind in (EvidenceKind.STDOUT, EvidenceKind.STDERR)
        )
        sanitizer_text = (run.directory / "sanitizer.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if self._trigger is None or len(self._trigger.constraints) != 2:
            raise AdapterContractError("bind the reviewed TriggerSpec before collecting evidence")
        provenance = Provenance(
            source=SourceKind.REAL_EXECUTION,
            method="strict parser over captured host and Compute Sanitizer output",
            scope=self._capability.id,
            source_sha256=registration.candidate.source_sha256,
            run_id=execution.run_id,
        )
        input_matches = (
            registration.trial.logical_count == self._experiment.capacity
            and registration.trial.extra_elements >= 1
        )
        input_observation = Observation(
            run_id=execution.run_id,
            candidate_id=execution.candidate_id,
            platform_id=self._capability.id,
            kind=ObservationKind.CONDITION,
            status=FactStatus.OBSERVED if params_match else FactStatus.UNKNOWN,
            provenance=provenance,
            satisfied=input_matches if params_match else None,
            condition_id=self._trigger.constraints[0].id,
            evidence_ids=parameter_evidence,
        )
        sanitizer_summary = sanitizer_error_count(sanitizer_text)
        sanitizer_observed = sanitizer_summary is not None and execution.termination not in (
            TerminationStatus.TIMEOUT,
            TerminationStatus.UNKNOWN,
        )
        target_observation = Observation(
            run_id=execution.run_id,
            candidate_id=execution.candidate_id,
            platform_id=self._capability.id,
            kind=ObservationKind.SANITIZER_ERROR,
            status=FactStatus.OBSERVED if sanitizer_observed else FactStatus.UNKNOWN,
            provenance=provenance,
            satisfied=(
                target_diagnostic_count(sanitizer_text, self._experiment.target_error_oracle) > 0
                if sanitizer_observed
                else None
            ),
            condition_id=self._trigger.constraints[1].id,
            evidence_ids=tuple(
                raw.reference.id
                for raw in run.evidence
                if raw.reference.kind is EvidenceKind.SANITIZER_LOG
            ),
        )
        run.observations = (input_observation, target_observation)
        return run.evidence

    def verify(
        self, candidate: PoCCandidate, trigger: TriggerSpec, execution: ExecutionRecord
    ) -> VerificationResult:
        if self._trigger is None:
            self.bind_trigger(trigger)
        if trigger != self._trigger or candidate.id != execution.candidate_id:
            raise AdapterContractError("verification trigger or candidate differs from this run")
        run = self._runs.get(execution.run_id)
        if run is None or run.execution != execution:
            raise AdapterContractError("verification requires this adapter's execution")
        evidence = run.evidence or self.collect_raw_evidence(execution)
        if run.request.verification_requirement is VerificationRequirement.NONE:
            return VerificationResult(
                candidate=candidate,
                trigger=trigger,
                execution=execution,
                evidence=evidence,
                observations=run.observations,
                verdict=VerificationVerdict.NOT_CHECKED,
                reason="This execution request did not request condition-linked verification.",
                scope=execution.scope,
            )
        return verify_result(candidate, trigger, execution, evidence, run.observations)


def make_vulnerability_and_trigger(
    platform_id: str, fixture_sha256: str
) -> tuple[VulnerabilitySpec, TriggerSpec]:
    vulnerability = VulnerabilitySpec(
        name="CUDA bounded-write off-by-one fixture",
        affected_component="examples/cuda_p1/oob_write.cu bounded_write_kernel",
        source_trust=SourceTrust.UNREVIEWED,
        platform_id=platform_id,
        description=(
            "Self-built training defect: the vulnerable kernel omits a capacity guard; "
            "the fixed control adds it. This is not a CVE or third-party defect."
        ),
        provenance=Provenance(
            source=SourceKind.SYNTHETIC,
            method="local fixture source hash",
            scope=platform_id,
            source_sha256=fixture_sha256,
        ),
        source_sha256=fixture_sha256,
    )
    analysis_source = "static analysis of versioned self-built CUDA fixture"
    trigger = TriggerSpec(
        vulnerability_id=vulnerability.id,
        platform_id=platform_id,
        constraints=(
            TriggerConstraint(
                name="input_boundary",
                expression="logical_count == capacity and extra_elements >= 1",
                knowledge=Fact[bool](
                    value=True,
                    status=FactStatus.INFERRED,
                    source=SourceKind.STATIC_ANALYSIS,
                    provenance=analysis_source,
                ),
            ),
            TriggerConstraint(
                name="target_memcheck_diagnostic",
                expression=(
                    "memcheck reports bounded_write_kernel's 4-byte out-of-bounds write "
                    "past the 128-byte device allocation"
                ),
                knowledge=Fact[bool](
                    value=True,
                    status=FactStatus.INFERRED,
                    source=SourceKind.STATIC_ANALYSIS,
                    provenance=analysis_source,
                ),
            ),
        ),
    )
    return vulnerability, trigger


def write_private_file(path: Path, content: bytes) -> None:
    """Public helper used by the runner for local, permission-restricted probe records."""
    _write_private(path, content)


def file_sha256(path: Path) -> str:
    return _file_sha256(path)
