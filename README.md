# HeteroPoCLab

Independent research on evidence-driven PoC generation and validation for heterogeneous GPU and MCU systems. This project is independent of ChipChainV3.

## Current stage: Preflight-01C

This stage establishes a normal CUDA execution baseline: a small deterministic vector addition kernel, an element-by-element CPU reference comparison, and Compute Sanitizer memcheck. No vulnerability samples or model API calls are involved.

Run in an initialized WSL2 Ubuntu environment as a non-root user:

```bash
bash scripts/run_cuda_smoke.sh
```

The runner targets `sm_75` explicitly and records compiler/tool versions, exact commands, exit codes, source/binary hashes, and separate raw stdout/stderr. It discovers installed tools from PATH or the existing CUDA 12.8 system directory; it does not install or reconfigure them. `NVCC` and `SANITIZER` can select existing executable paths.

Each run writes a unique directory under `output/preflight-01c/`. Raw local evidence and compiled binaries are ignored by Git. Keep that directory private; paths and host details in raw logs are not intended for public publication. Only the reviewed source, runner, and redacted acceptance summary are committed.

See [Preflight-01C evidence and limits](docs/preflight/preflight-01c.md) for the actual recorded validation results. A clean memcheck result only describes this normal example in its recorded environment and selected tool mode.
