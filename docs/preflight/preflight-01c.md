# Preflight-01C：CUDA 真实执行与验证基线

日期：2026-10-13，Asia/Shanghai。这里只验证一个正常 CUDA 程序；所有 CUDA 编译、GPU 执行与 Compute Sanitizer 均在 WSL2 Linux 下由非 root 用户完成。

## 实际环境

以下来自两轮实际运行的 `summary.json`、工具版本原始日志和 GPU 查询，标记为 OBSERVED：

| 项目 | 观测值 |
| --- | --- |
| 执行环境 | WSL2，Ubuntu 22.04.5 LTS，x86_64，内核数字版本 6.6.87.2 |
| 用户权限 | non_root=true；初始会话 UID 检查为非零 |
| 实际运行设备 | NVIDIA GeForce GTX 1660，CC 7.5，6144 MiB 显存 |
| NVIDIA Windows KMD / GPU 查询驱动 | 617.42 |
| 驱动报告 CUDA 兼容能力 | 13.4；不代表 Toolkit 版本 |
| CUDA Toolkit / nvcc | 12.8 / 12.8.93 |
| Compute Sanitizer | 2025.1.0.0 |
| 主机 C++ 编译器 | GCC/G++ 11.4.0，实际解析为 `/usr/bin/x86_64-linux-gnu-g++-11` |

Windows CPU/OS 的完整环境审计与 VMware/MCU 均不是本次检查内容，未用 Windows CUDA 输出代替 Linux 执行证据。

## 样例与命令

`examples/cuda_smoke/vector_add.cu` 使用 4096 个确定性元素：`a[i]=i%251`，`b[i]=(i*7)%127`。这些小整数及和都能由 float 精确表示，因此逐元素采用精确比较。设备数组总分配 49152 bytes（48 KiB）；一次 kernel 使用 32 blocks × 128 threads，并有边界判断。没有压力测试或长时间 kernel。

程序检查设备计数、设备选择与属性、分配、拷贝、kernel 启动、同步以及释放错误；只在 4096 个元素全部比对通过且清理成功时返回 0。其他退出码：1=CUDA 操作失败，2=无设备或 CC 不是 7.5，3=比较失败/未完成，4=仅清理失败。先前非零失败不会被清理阶段覆盖。

实际编译参数如下；仓库和运行目录在公开文本中以变量脱敏，原始绝对 argv 在各轮私有 `commands.json` 中：

```bash
/usr/local/cuda-12.8/bin/nvcc \
  -arch=sm_75 -std=c++14 -O2 \
  --compiler-bindir /usr/bin/x86_64-linux-gnu-g++-11 \
  "$REPO/examples/cuda_smoke/vector_add.cu" -o "$RUN/vector_add"

"$RUN/vector_add"
/usr/local/cuda-12.8/bin/compute-sanitizer \
  --tool memcheck --error-exitcode 86 "$RUN/vector_add"
```

外层 GNU timeout 限制编译 90 秒、直接运行 30 秒、memcheck 60 秒，TERM 后最多 5 秒清理。工具版本探测各限制 15 秒。两轮均未超时。CUDA 12.8 官方 nvcc 支持 `sm_75`；此处架构参数是实际执行参数，而非仅文档推导。[CUDA 12.8 nvcc 文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)

## 两轮实测验收

| 检查项 | 第一轮 | 第二轮 | 证据边界 |
| --- | --- | --- | --- |
| CUDA 编译 | PASS，退出 0 | PASS，退出 0 | nvcc 实际编译，显式 sm_75 |
| 真实 GPU kernel 执行 | PASS，程序退出 0 | PASS，程序退出 0 | CUDA 启动、同步、D2H 与实际设备输出均成功 |
| CPU/GPU 结果正确性 | PASS，4096/4096，差异 0 | PASS，4096/4096，差异 0 | 每个元素均检查 |
| Compute Sanitizer memcheck | PASS，工具退出 0，0 errors | PASS，工具退出 0，0 errors | 正常样例在本环境和该模式下的观测 |
| 执行证据 | PASS | PASS | 独立 stdout/stderr、argv、版本、时间、退出码、哈希 |
| 同一脚本重复执行 | PASS | PASS | 两轮源码与 runner 哈希相同，两次 runner 均退出 0 |

第一轮开始时间：2026-10-13 11:16:43（UTC+08:00）。第二轮开始时间：2026-10-13 11:17:18（UTC+08:00）。精确 UTC 时间、单步起止和耗时见各轮摘要。

两轮原始程序日志均实际包含：

```text
DEVICE_NAME=NVIDIA GeForce GTX 1660
DEVICE_CC=7.5
ELEMENTS=4096
CHECKED_ELEMENTS=4096
MISMATCH_COUNT=0
RESULT=PASS
PROGRAM_EXIT_CODE=0
```

两轮 Sanitizer 原始日志均实际包含：

```text
========= COMPUTE-SANITIZER
...
========= ERROR SUMMARY: 0 errors
```

直接运行的退出码 0 是程序的 OS 进程退出状态。memcheck 的工具退出码 0 是 Compute Sanitizer 的 OS 进程退出状态；被插桩程序输出的 `PROGRAM_EXIT_CODE=0` 是程序自身标记，**不是独立取得的被插桩进程 OS 退出码**。runner 同时要求工具退出 0、唯一程序成功标记、完整 CPU/GPU 比对、唯一零错误摘要及无异常工具诊断。

`--error-exitcode 86` 的语义由本机实际安装的官方随包 v2025.1.0 文档核对：`/usr/local/cuda-12.8/compute-sanitizer/docs/ComputeSanitizer/index.html`。该选项针对程序原本成功但 Sanitizer 检测到错误的情况；不能只凭工具退出码推断所有被测程序行为。网上 canonical 文档会更新，随包版本和实际日志是本次依据。[Compute Sanitizer 文档入口](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

**本次能够得出的结论仅是：当前正常样例在该执行环境中未发现指定的内存错误。** 这不证明漏洞检测系统已经开发完成，也不证明对未来全部漏洞类别有效。

## 本地证据与哈希

所有原始日志与二进制均留在仓库根目录下的私有、Git 忽略目录，未提交公开仓库：

- 第一轮：`output/preflight-01c/20261013T031643.091502Z-fcdd90a9/`
- 第二轮：`output/preflight-01c/20261013T031718.515566Z-4ce947e5/`

每轮包含 `summary.json`、`commands.json`、`vector_add`，以及 `nvcc_version`、`sanitizer_version`、`gxx_version`、`nvidia_version`、`gpu_query`、`compile`、`direct_run`、`sanitizer_run` 的独立 `.stdout.log` / `.stderr.log`；还有 GNU timeout 版本记录。目录权限实际为 700，日志和摘要为 600，`git check-ignore` 已确认摘要和二进制被忽略。

| 产物 | SHA-256 |
| --- | --- |
| 两轮共同 CUDA 源码 | `5a1a714660617a1f6f99de452120d3f0886f60b3b4d58c5cc39194c5ba75b136` |
| 两轮共同 runner | `7738375c1ab4c1fd98e33c4d9b6ddeae6f8c2d04af699065588694e25f19d84f` |
| 第一轮二进制 | `11244c2d72e7fd258a4bc56dd87d3a1dfff1b154a832e393f00fba2a9c561e66` |
| 第二轮二进制 | `d6ed7120ea7b971bec8a30ccf848d87ef8476c6e78618e9484500938d4ea2df6` |

源码在运行过程中未变化。二进制哈希不同；本次验证的是重复执行通过，没有宣称构建产物按位一致。原始日志包含本地绝对路径，应保持私有；公开摘要只使用相对目录与必要的硬件/版本字段，不包含身份、凭证、硬件 UUID 或网络配置。

## 复现、项目检查与限制

在同一 WSL2 Ubuntu、非 root、已有相应工具的环境中，从仓库根目录运行：

```bash
bash scripts/run_cuda_smoke.sh
```

runner 检查 WSL2、非 root、工具版本、源文件和 GPU；可通过 `NVCC` / `SANITIZER` 选择现有工具。`OUT_DIR` 只允许仓库 `output/` 范围内的位置，每次创建唯一子目录。脚本不安装工具、修改 PATH/驱动/网络或调用模型 API。未执行的步骤标记 `NOT_CHECKED`，执行失败标记 `FAILED`，只有全部所需检查通过才返回成功。

本次已完成 Bash `-n`、嵌入 Python AST、UTF-8/LF 检查、独立源码/runner 只读审阅、两轮实际 CUDA + memcheck、Git diff whitespace 检查以及 output 忽略和权限核验。仓库当前没有其他构建或测试框架；没有把缺失的 CI/GPU runner 当作已通过检查。

未验证内容均为 NOT_CHECKED：真实漏洞/阳性错误检出、racecheck/initcheck/synccheck、其他 GPU/CC/OS、Windows 原生 CUDA、长时间/大规模负载、多 GPU、MCU/QEMU/Renode、真实开发板、通用 Agent、PoC 自动生成、DeepSeek API，以及构建产物按位可复现的原因分析。

Git 传输限制：WSL 的 HTTPS Git 推送遇到 TLS 握手失败/超时（FAILED，未更改网络设置）。现有 Windows Git 能访问同一仓库并用于 Git 对象传输；这不是 CUDA 验收的替代执行环境。初始远端没有分支，只建立了最小 README 的 main 提交；后续源码、脚本和文档都在 `preflight/01c-cuda-smoke` 分支，保留 main 历史。未 force push、创建 stable tag 或自行合并 PR。
