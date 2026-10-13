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
