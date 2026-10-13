# GPU-P1: bounded CUDA trigger and evidence baseline

## Scope

GPU-P1 exercises a self-built, deterministic CUDA bounds defect. It is not a historical CVE, a third-party vulnerability, or a general PoC generator. The fixture has a vulnerable kernel and a fixed control; its candidate generator enumerates the declared normal, capacity-boundary, minimum-trigger, and maximum-trigger inputs. All GPU executions are serialized under Compute Sanitizer memcheck.

The run requires the existing WSL2 CUDA environment and runs as a non-root user:

```bash
.venv/bin/python scripts/run_gpu_p1.py --spec examples/cuda_p1/experiment.json
```

The runner uses the installed CUDA tools; it does not install software, use a model API, or run the fixture outside Compute Sanitizer. Compilation explicitly targets `sm_75`. The device allocation is 128 bytes, inputs are restricted to the experiment schema, compile timeout is 60 seconds, runtime timeout is 30 seconds, captured output above 64 KiB fails acceptance, and only one GPU run may execute at a time.

## Recorded real-device result

The completed run used WSL2 Ubuntu and a non-root user on an NVIDIA GeForce GTX 1660 (Compute Capability 7.5, 6144 MiB), driver 617.42, CUDA Toolkit 12.8 / nvcc 12.8.93, and Compute Sanitizer 2025.1.0.0. The privacy-limited nvidia-smi query observed the GPU and driver fields; generic CUDA compatibility and CUDA UMD fields remain NOT_CHECKED and are not conflated with the installed Toolkit release.

| Measure | Observed |
| --- | ---: |
| Deterministic candidates | 8 |
| Successful `nvcc` compilations (exit 0) | 8 |
| Compute Sanitizer wrapped GPU executions | 9 |
| Exact target OOB diagnostic hits | 3 trials |
| Runs with no matching target diagnostic | 6 |
| Distinct candidates with condition-linked `VERIFIED` results | 2 |
| Repeat of vulnerable minimum-trigger candidate | Passed |

The vulnerable minimum-trigger input (`capacity=32, logical_count=32, extra=1`) produced one exact target diagnostic. The maximum-trigger input (`extra=2`) produced two. Both fixed controls and both vulnerable non-trigger controls had zero target matches. All six negative controls completed CPU/GPU element comparisons with zero mismatches.

The positive runs returned the configured Compute Sanitizer exit code 86 with error-summary counts 3 and 4. The target-specific parser counted 1 and 2 out-of-bounds writes respectively; the remaining summary entries were follow-on CUDA API errors after the device context fault. Kernel launch status was CUDA success, device synchronization returned 719 / `cudaErrorLaunchFailure`, and the fixture skipped CPU comparison. The fixture logged `APP_EXIT_INTENT=5` after selecting its nonzero return path, but the child application's OS exit code is recorded as UNKNOWN because the sanitizer wrapper returned its configured error code and does not expose the child's status separately in this run. The directly observed wrapper exit remains `ERROR_EXIT` / 86 in the shared ExecutionRecord. The six clean controls have inferred application exit 0 based on wrapper exit 0, zero sanitizer errors, and the fixture marker. This preserves the abnormal positive process status instead of normalizing it away. See the [NVIDIA Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) for `--error-exitcode` and `--check-exit-code` semantics.

The M0-A verifier returned `VERIFIED` for the two positive candidates while preserving `ERROR_EXIT` / 86. It required objective input-boundary evidence linked to the captured host output, plus an observed target-specific sanitizer diagnostic linked to the exact raw sanitizer log. The CUDA API error code, process termination, sanitizer summary total, and app's trigger-intent marker could not substitute for the target oracle.

## Evidence and reproduction

The successful acceptance artifacts are stored locally under the Git-ignored directory:

`output/gpu-p1/acceptance-20261013T094452Z-cda39666/`

`report.json` contains each candidate/run identity, compile and sanitizer exit codes, application exit basis, CUDA API statuses, exact target match count, verification result, and SHA-256 references. Each run directory keeps the input, compile stdout/stderr, target stdout/stderr, Compute Sanitizer log, and run metadata separately. The same candidate and binary were rerun with a new run ID; its input hash and target match count agreed. The report retains the wrapper's exit 86 and leaves the positive child-process exit UNKNOWN.

To repeat the full matrix on the same supported environment, run the command above. The runner creates a new private output directory and returns nonzero if compilation, execution, expected positive/negative behavior, or repetition checks fail. Raw logs and compiled binaries remain local and are ignored by Git. Only this reviewed redacted summary and the deterministic fixture/runner are intended for publication.

## Limits

These results support only the specific self-built kernel, compiled binary, WSL2 GTX 1660, driver/toolkit combination, input matrix, and Compute Sanitizer memcheck mode recorded here. Zero-error control runs do not show that memcheck covers every memory-safety defect or that this method transfers to other GPU models, drivers, CUDA versions, applications, or hardware. No CUDA vulnerability backend for third-party code, DeepSeek API, LangGraph workflow, historical CVE, or MCU backend is implemented by GPU-P1. CPU-only CI validates contracts and parsing only; it does not claim real GPU execution.
