# HeteroPoCLab contributor instructions

## Global requirements

- Keep this repository independent of ChipChainV3.
- Never fabricate evidence. Record observed results, inferences, failures and checks not performed accurately.
- Keep raw host logs, binaries and local host inventory under ignored output/. Publish only reviewed source, reproducible scripts and redacted summaries.
- Never commit secrets, credentials or sensitive host identity. Before committing, audit tracked files, staged diff and generated evidence for sensitive information and accidental artifacts. .gitignore cannot replace this audit or protect secrets already tracked or committed.
- Run dangerous experiments in isolated environments with least privilege and bounded execution time and resources.
- Execute real vulnerability samples or exploitation PoCs only with explicit task authorization and within its approved scope.
- Work on a feature branch. Preserve existing history and unrelated changes. Force-pushing, stable tags and PR merges require explicit authorization.

## Preflight-01C run boundaries

The following boundaries apply to Preflight-01C and its review revisions. Later explicitly authorized tasks may conduct GPU/MCU PoC experiments and DeepSeek API validation under the global requirements above.

- This stage covers only a small deterministic normal CUDA vector addition program and its execution evidence.
- Run CUDA compilation, GPU execution and Compute Sanitizer in WSL2 Linux as a non-root user; Windows tool output cannot substitute for Linux execution.
- Target sm_75 explicitly. Check CUDA API, launch, synchronization and CPU/GPU comparison results. Use bounded commands and small allocations.
- During this stage, do not install drivers, change networking, call model APIs, execute vulnerability samples or exploits, force-push, create stable tags or merge pull requests.
- For runtime changes, run applicable syntax checks and the smoke runner. For documentation and ignore-rule revisions only, run relevant static checks and explicitly record that GPU tests were not rerun.

## GPU-P1 run boundaries

These constraints apply to the GPU-P1 acceptance fixture and runner. Later stages may expand the experiment scope only with explicit authorization and a new reviewed plan.

- GPU-P1 compiles and executes only the self-built vulnerable/fixed fixture in `examples/cuda_p1/oob_write.cu`; it does not target third-party software or historical CVEs.
- Run CUDA compilation, GPU execution, and Compute Sanitizer under WSL2 as a non-root user on the scoped GTX 1660 / Compute Capability 7.5 environment.
- Run every GPU trial through Compute Sanitizer memcheck. Keep one trial at a time, cap the device allocation at 128 bytes, compilation at 60 seconds, execution at 30 seconds, input at 256 bytes, and reject runs whose captured output exceeds 64 KiB.
- Do not call model APIs or start agent workflows in this stage.
- Keep raw logs and binaries in Git-ignored `output/gpu-p1/`; commit only reviewed source, reproducible scripts, and redacted conclusions.
- CPU-only CI validates contracts and parsing; it does not count as a real GPU execution.
