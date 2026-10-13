# HeteroPoCLab contributor instructions

- Keep this repository independent of ChipChainV3.
- Preflight-01C covers only a small deterministic normal CUDA vector addition program and its execution evidence.
- Run CUDA compilation, GPU execution and Compute Sanitizer in WSL2 Linux as a non-root user; Windows tool output cannot substitute for Linux execution.
- Target sm_75 explicitly. Check CUDA API, launch, synchronization and CPU/GPU comparison results. Use bounded commands and small allocations.
- Keep raw logs, binaries and local host inventory under ignored output/. Publish only reviewed source, reproducible scripts and redacted summaries.
- Do not install drivers, change networking, call model APIs, execute exploits, force-push, create stable tags or merge pull requests.
- Work on a feature branch. Preserve existing history and unrelated changes.
- Before committing, run applicable syntax checks and the smoke runner; inspect git diff and tracked files for privacy and accidental artifacts.
