# HeteroPoCLab

Independent research on evidence-driven PoC generation and validation for heterogeneous GPU and MCU systems. This project is independent of ChipChainV3.

## Accepted baseline: Preflight-01C

This stage establishes a normal CUDA execution baseline: a small deterministic vector addition kernel, an element-by-element CPU reference comparison, and Compute Sanitizer memcheck. No vulnerability samples or model API calls are involved.

Run in an initialized WSL2 Ubuntu environment as a non-root user:

```bash
bash scripts/run_cuda_smoke.sh
```

The runner targets `sm_75` explicitly and records compiler/tool versions, exact commands, exit codes, source/binary hashes, and separate raw stdout/stderr. It discovers installed tools from PATH or the existing CUDA 12.8 system directory; it does not install or reconfigure them. `NVCC` and `SANITIZER` can select existing executable paths.

Each run writes a unique directory under `output/preflight-01c/`. Raw local evidence and compiled binaries are ignored by Git. Keep that directory private; paths and host details in raw logs are not intended for public publication. Only the reviewed source, runner, and redacted acceptance summary are committed.

See [Preflight-01C evidence and limits](docs/preflight/preflight-01c.md) for the actual recorded validation results. A clean memcheck result only describes this normal example in its recorded environment and selected tool mode.

## M0-A: domain and platform contracts

The Preflight-01C baseline was squash-merged as `ed5a90892fe63e1fd7e2044a05094c56a2191479`. Its CUDA source, runner and recorded acceptance evidence remain unchanged.

M0-A provides a Python 3.10+ package with strict Pydantic v2 models, deterministic content identities, multi-dimensional platform compatibility and an in-memory Mock Adapter. CPU-only tests include CUDA, ARM Cortex-M, RISC-V and a hypothetical PowerPC/C8000 contract. These tests do not establish real hardware compatibility or vulnerability triggering.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests examples/m0a
.venv/bin/ruff format --check src tests examples/m0a
.venv/bin/mypy
```

See [M0-A architecture, JSON flow and scientific limits](docs/m0a-domain-contracts.md). Raw local check logs and dependency wheels belong under ignored `output/m0a/`; do not publish host paths or credentials.
