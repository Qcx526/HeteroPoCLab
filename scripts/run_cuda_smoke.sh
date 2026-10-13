#!/usr/bin/env bash
# WSL2, non-root, small normal CUDA smoke test. No installs or network calls.
# NVCC / SANITIZER: optional executable path or command name.
# OUT_DIR: optional base within repo output/; every run creates a unique child.
# Raw logs and exact argv are private evidence under ignored output/.
set -eu
umask 077
exec python3 -I -B - "${BASH_SOURCE[0]}" <<'PY'
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


if os.name != 'posix' or os.geteuid() == 0:
    print('Refused: run in WSL2 Linux as a non-root user; no commands or files created.', file=sys.stderr)
    sys.exit(2)
os.umask(0o077)
runner_path = Path(sys.argv[1]).resolve()
repo = runner_path.parent.parent
output_base = repo / 'output'
if output_base.is_symlink() or not inside(output_base.resolve(), repo):
    print('Refused: repository output/ resolves outside the repository.', file=sys.stderr)
    sys.exit(2)
requested = os.environ.get('OUT_DIR')
base = Path(requested) if requested else output_base / 'preflight-01c'
if not base.is_absolute():
    base = repo / base
base = base.resolve()
invalid_output = not inside(base, output_base.resolve())
if invalid_output:
    base = output_base / 'preflight-01c'
leaf = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8]
output = base / leaf
output.mkdir(mode=0o700, parents=True, exist_ok=False)
source = repo / 'examples' / 'cuda_smoke' / 'vector_add.cu'
binary = output / 'vector_add'
checks = dict.fromkeys(('preconditions', 'compile', 'run', 'correctness', 'sanitizer', 'evidence'), 'NOT_CHECKED')
summary = {
    'schema_version': 1, 'started_at_utc': now(), 'ended_at_utc': None,
    'output_directory': output.relative_to(repo).as_posix(), 'checks': checks,
    'overall': 'FAILED', 'target_architecture': 'sm_75', 'language_standard': 'c++14',
    'runner': {'file': 'scripts/run_cuda_smoke.sh', 'sha256': sha256(runner_path)},
    'source': {'file': 'examples/cuda_smoke/vector_add.cu', 'sha256': None, 'unchanged_during_run': None},
    'binary': {'file': 'vector_add', 'sha256': None},
    'environment': {}, 'versions': {}, 'probes': {}, 'execution': {}, 'failures': [],
    'provenance': {
        'commands': 'summary argv redact repository/user paths; commands.json stores exact local argv',
        'raw_logs': 'separate stdout/stderr files under ignored output/; created with private umask 077',
        'sanitizer_program_exit': 'PROGRAM_EXIT_CODE is a program-reported marker under instrumentation, not an independent OS exit status',
        'scope': 'small normal CUDA vector addition; no vulnerability PoC or stress workload',
    },
}
exact_commands = []
child_env = os.environ.copy()
child_env.update({'LC_ALL': 'C', 'LANG': 'C', 'PYTHONDONTWRITEBYTECODE': '1'})
for name in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'NVCC_CCBIN'):
    child_env.pop(name, None)


def redact(arg):
    text = str(arg)
    if text.startswith(str(repo) + '/'):
        return '<REPO>/' + text[len(str(repo)) + 1:]
    if re.match(r'^/(?:home|root)/', text) or re.match(r'^/mnt/[a-z]/Users/', text, re.I):
        return '<USER_PATH>/' + Path(text).name
    return text


def resolve_tool(name, env_name=None, fallback=None):
    configured = os.environ.get(env_name) if env_name else None
    candidate = configured or shutil.which(name) or fallback
    if not candidate:
        return None
    found = shutil.which(candidate) if '/' not in candidate else candidate
    if not found:
        return None
    path = Path(found).expanduser().resolve()
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def execute(stage, argv, limit, bounded=True):
    launched = ([timeout_tool, '--verbose', '--signal=TERM', '--kill-after=5s', str(limit) + 's'] + argv) if bounded else argv
    exact_commands.append({'stage': stage, 'argv': argv, 'launched_argv': launched})
    out_name, err_name = stage + '.stdout.log', stage + '.stderr.log'
    item = {'argv': [redact(x) for x in argv], 'launched_argv': [redact(x) for x in launched],
            'started_at_utc': now(), 'ended_at_utc': None, 'elapsed_seconds': None,
            'timeout_seconds': limit, 'timed_out': False, 'exit_code': None,
            'stdout_log': out_name, 'stderr_log': err_name}
    start = time.monotonic()
    try:
        with (output / out_name).open('wb') as out, (output / err_name).open('wb') as err:
            process = subprocess.Popen(launched, cwd=repo, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, env=child_env, start_new_session=True)
            try:
                item['exit_code'] = process.wait(timeout=limit + 7 if bounded else limit)
            except subprocess.TimeoutExpired:
                item['timed_out'] = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                item['exit_code'] = process.wait(timeout=2)
        if bounded and item['exit_code'] == 124:
            item['timed_out'] = True
    except OSError:
        item['launch_failed'] = True
    finally:
        item['ended_at_utc'] = now()
        item['elapsed_seconds'] = round(time.monotonic() - start, 6)
    stdout = (output / out_name).read_text(errors='replace') if (output / out_name).exists() else ''
    stderr = (output / err_name).read_text(errors='replace') if (output / err_name).exists() else ''
    if bounded and re.search(r'timeout: sending signal (?:TERM|KILL) to command', stderr):
        item['timed_out'] = True
    return item, stdout, stderr


def version(stage, executable, parser):
    item, stdout, stderr = execute(stage, [executable, '--version'], 15)
    summary['probes'][stage] = item
    parsed = parser(stdout + stderr)
    summary['versions'][stage] = parsed
    return item['exit_code'] == 0 and not item['timed_out'] and parsed is not None


def markers(stdout, stderr):
    raw = stdout + '\n' + stderr
    results = re.findall(r'^RESULT=([^\r\n]*)$', raw, re.M)
    exits = re.findall(r'^PROGRAM_EXIT_CODE=([^\r\n]*)$', raw, re.M)
    observed = {}
    for key in ('DEVICE_NAME', 'DEVICE_CC', 'ELEMENTS', 'CHECKED_ELEMENTS', 'MISMATCH_COUNT'):
        values = re.findall(r'^' + key + r'=([^\r\n]*)$', raw, re.M)
        value = values[0] if len(values) == 1 else None
        if key in ('ELEMENTS', 'CHECKED_ELEMENTS', 'MISMATCH_COUNT'):
            value = int(value) if value and re.fullmatch(r'\d+', value) else None
        elif value and not re.fullmatch(r'[A-Za-z0-9 ()+._/-]{1,100}', value):
            value = None
        observed[key.lower()] = value
    comparison = observed['elements'] == observed['checked_elements'] == 4096 and observed['mismatch_count'] == 0
    return {'result_markers': [x if x in ('PASS', 'FAIL') else 'UNRECOGNIZED' for x in results],
            'program_exit_markers': [int(x) if re.fullmatch(r'-?\d+', x) else 'UNRECOGNIZED' for x in exits],
            'valid_pass_markers': results == ['PASS'] and exits == ['0'],
            'observed_program_output': observed, 'cpu_gpu_comparison_verified': comparison}


def finish():
    summary['ended_at_utc'] = now()
    summary['elapsed_seconds'] = round(time.monotonic() - run_started, 6)
    summary['overall'] = 'PASS' if all(x == 'PASS' for x in checks.values()) else 'FAILED'
    (output / 'commands.json').write_text(json.dumps(exact_commands, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('CUDA smoke: ' + summary['overall'])
    print('Evidence: ' + summary['output_directory'] + '/summary.json')
    print('Checks: ' + ', '.join(k + '=' + v for k, v in checks.items()))
    return 0 if summary['overall'] == 'PASS' else 1


run_started = time.monotonic()
timeout_tool = resolve_tool('timeout', fallback='/usr/bin/timeout')
try:
    kernel = Path('/proc/sys/kernel/osrelease').read_text().strip()
    is_wsl2 = 'microsoft' in kernel.lower() and ('wsl2' in kernel.lower() or 'microsoft-standard' in kernel.lower())
    summary['environment'].update({'execution_environment': 'WSL2' if is_wsl2 else 'UNSUPPORTED',
                                   'kernel_numeric_release': re.match(r'[0-9.]+', kernel).group(0),
                                   'non_root': True, 'architecture': os.uname().machine})
    distro = {}
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            value = value.strip().strip(chr(34)).strip(chr(39))
            if key in ('ID', 'VERSION_ID', 'PRETTY_NAME') and re.fullmatch(r'[A-Za-z0-9 .()_+:/-]{1,100}', value):
                distro[key.lower()] = value
    summary['environment']['distribution'] = distro
    if invalid_output:
        raise RuntimeError('OUT_DIR must resolve within repository output/; evidence was saved in the safe default location.')
    if not is_wsl2:
        raise RuntimeError('WSL2 kernel markers are required; no CUDA command executed.')
    if not timeout_tool:
        raise RuntimeError('GNU timeout executable is required.')
    timeout_probe, stdout, stderr = execute('timeout_version', [timeout_tool, '--version'], 15, bounded=False)
    summary['probes']['timeout_version'] = timeout_probe
    if timeout_probe['exit_code'] != 0 or 'GNU coreutils' not in stdout + stderr:
        raise RuntimeError('GNU timeout version could not be verified.')
    nvcc = resolve_tool('nvcc', 'NVCC', '/usr/local/cuda-12.8/bin/nvcc')
    sanitizer = resolve_tool('compute-sanitizer', 'SANITIZER', '/usr/local/cuda-12.8/bin/compute-sanitizer')
    nvidia = resolve_tool('nvidia-smi', fallback='/usr/lib/wsl/lib/nvidia-smi')
    gcc = resolve_tool('g++')
    if not all((nvcc, sanitizer, nvidia, gcc)):
        raise RuntimeError('Required nvcc, compute-sanitizer, nvidia-smi, or g++ executable is unavailable.')
    if not source.is_file() or not inside(source.resolve(), repo):
        raise RuntimeError('Reviewed CUDA source is absent or resolves outside the repository.')
    summary['source']['sha256'] = sha256(source)
    def matched(pattern):
        return lambda text: (re.search(pattern, text, re.I).group(1) if re.search(pattern, text, re.I) else None)
    def nvcc_version(text):
        m = re.search(r'\brelease\s+(\d+\.\d+),\s+V([0-9.]+)', text)
        return {'toolkit_release': m.group(1), 'compiler_version': m.group(2)} if m else None
    version_ok = [
        version('nvcc_version', nvcc, nvcc_version),
        version('sanitizer_version', sanitizer, matched(r'\b(?:version|V)\s*:?\s*(\d+(?:\.\d+){1,4})')),
        version('gxx_version', gcc, matched(r'\b(\d+(?:\.\d+){1,3})\b')),
        version('nvidia_version', nvidia, matched(r'CUDA\s+(?:UMD\s+)?Version\s*:\s*(\d+\.\d+)')),
    ]
    summary['versions']['nvidia_version_meaning'] = 'driver CUDA compatibility; not installed Toolkit version'
    gpu_item, stdout, stderr = execute('gpu_query', [nvidia, '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader,nounits'], 15)
    summary['probes']['gpu_query'] = gpu_item
    devices = []
    for row in csv.reader(io.StringIO(stdout)):
        row = [x.strip() for x in row]
        if len(row) == 4 and re.fullmatch(r'[A-Za-z0-9 ()+._/-]{1,100}', row[0]) and re.fullmatch(r'[0-9.]+', row[1]) and re.fullmatch(r'[0-9.]+', row[2]) and re.fullmatch(r'\d+\.\d+', row[3]):
            devices.append({'model': row[0], 'driver_version': row[1], 'memory_total_mib': float(row[2]), 'compute_capability': row[3]})
    summary['environment']['gpus'] = devices
    summary['environment']['host_cpp_compiler'] = redact(gcc)
    if not all(version_ok) or gpu_item['exit_code'] != 0 or not devices:
        raise RuntimeError('Tool versions or accessible NVIDIA GPU inventory could not be verified.')
    checks['preconditions'] = 'PASS'
    compile_item, _, _ = execute('compile', [nvcc, '-arch=sm_75', '-std=c++14', '-O2', '--compiler-bindir', gcc, str(source), '-o', str(binary)], 90)
    summary['execution']['compile'] = compile_item
    compiled = compile_item['exit_code'] == 0 and not compile_item['timed_out'] and binary.is_file()
    checks['compile'] = 'PASS' if compiled else 'FAILED'
    if not compiled:
        raise RuntimeError('CUDA compilation failed or timed out; later execution checks were not performed.')
    summary['binary']['sha256'] = sha256(binary)
    direct_item, stdout, stderr = execute('direct_run', [str(binary)], 30)
    direct_markers = markers(stdout, stderr)
    direct_item.update(direct_markers)
    direct_item['exit_code_provenance'] = 'OS process exit status of the directly executed compiled program'
    summary['execution']['direct_run'] = direct_item
    direct_ok = direct_item['exit_code'] == 0 and not direct_item['timed_out'] and direct_markers['valid_pass_markers']
    checks['run'] = 'PASS' if direct_ok else 'FAILED'
    checks['correctness'] = 'PASS' if direct_ok and direct_markers['cpu_gpu_comparison_verified'] else 'FAILED'
    sanitized_item, stdout, stderr = execute('sanitizer_run', [sanitizer, '--tool', 'memcheck', '--error-exitcode', '86', str(binary)], 60)
    sanitized_item.update(markers(stdout, stderr))
    raw = stdout + '\n' + stderr
    errors = re.findall(r'^\s*=*\s*ERROR SUMMARY:\s*(\d+)\s+errors?\b.*$', raw, re.M)
    unexpected = re.findall(r'^\s*=+\s*(?:Warning\b|Error(?!\s+SUMMARY\b)\b|Fatal\b|Internal error\b).*$', raw, re.M | re.I)
    sanitized_item['memcheck_error_summary_counts'] = [int(x) for x in errors]
    sanitized_item['unexpected_sanitizer_diagnostic_count'] = len(unexpected)
    sanitized_item['tool_exit_code'] = sanitized_item['exit_code']
    sanitized_item['exit_code_provenance'] = 'OS exit status of Compute Sanitizer; PROGRAM_EXIT_CODE is only a program-reported marker'
    summary['execution']['sanitizer_run'] = sanitized_item
    checks['sanitizer'] = 'PASS' if sanitized_item['exit_code'] == 0 and not sanitized_item['timed_out'] and sanitized_item['valid_pass_markers'] and sanitized_item['cpu_gpu_comparison_verified'] and errors == ['0'] and not unexpected else 'FAILED'
    summary['source']['unchanged_during_run'] = sha256(source) == summary['source']['sha256']
    log_names = [x[k] for x in list(summary['probes'].values()) + list(summary['execution'].values()) for k in ('stdout_log', 'stderr_log')]
    checks['evidence'] = 'PASS' if summary['source']['unchanged_during_run'] and summary['binary']['sha256'] and all((output / x).is_file() for x in log_names) else 'FAILED'
    if checks['evidence'] == 'FAILED':
        summary['failures'].append('Source changed during execution or required evidence files are incomplete.')
    if checks['correctness'] == 'FAILED':
        summary['failures'].append('CPU/GPU output did not uniquely confirm all 4096 elements checked with zero mismatches.')
    if checks['run'] == 'FAILED':
        summary['failures'].append('Direct execution failed, timed out, or lacked unique successful result/exit markers.')
    if checks['sanitizer'] == 'FAILED':
        summary['failures'].append('Memcheck did not complete with tool exit 0, one zero-error summary, unique successful program markers, and no unexpected diagnostics.')
except RuntimeError as error:
    if checks['preconditions'] == 'NOT_CHECKED':
        checks['preconditions'] = 'FAILED'
    summary['failures'].append(str(error))
except Exception:
    if checks['preconditions'] == 'NOT_CHECKED':
        checks['preconditions'] = 'FAILED'
    summary['failures'].append('Unexpected local runner failure; raw paths and exception details withheld.')
    for key in ('compile', 'run', 'sanitizer'):
        if key in summary['execution'] and checks[key] == 'NOT_CHECKED':
            checks[key] = 'FAILED'
sys.exit(finish())
PY
