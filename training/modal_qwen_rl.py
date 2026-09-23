"""Bounded fresh Qwen gameplay RL on one H100, with native tools and no dataset.

Launch: modal run --detach training/modal_qwen_rl.py --run-id qwen27b-rl-01
The entrypoint privately forwards the already authenticated W&B account.
"""
from __future__ import annotations

import hashlib
import json
import netrc
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path('/opt/thunderhill')
RUNS = Path('/runs')
PYTHON = '/opt/thunderhill/training/.venv/bin/python'
GODOT = '/usr/local/bin/godot'
GODOT_SOURCE = Path(os.environ.get('THUNDERHILL_GODOT', str(
    Path.home() / '.local/share/thunderhill-tools/godot-4.7.2/Godot_v4.7.2-stable_linux.x86_64')))
ARTIFACT_VOLUME = 'thunderhill-runs-v2'
CACHE_VOLUME = 'thunderhill-huggingface-v2'
FUNCTION_TIMEOUT = 7200
CHILD_TIMEOUT = 6900
# Explicit runtime source allowlist. No adapter, teacher output, or dataset is mounted.
TRAINING_FILES = (
    'train_full_lap_grpo.py', 'agent_harness.py', 'batched_policy.py',
    'experiment_tracking.py', 'functiongemma_tools.py', 'greedy_cache.py',
    'lap_audit.py', 'lap_episode.py', 'lap_policy.py', 'lap_prefix.py',
    'lap_rollout.py', 'model_runtime.py', 'native_constraints.py', 'native_tools.py',
    'qwen_tools.py', 'trajectory_update.py', 'video_jobs.py',
)
TOOL_FILES = ('check_agent.py', 'check_parallel.py')

app = modal.App('thunderhill-qwen-gameplay-rl')
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True, version=2)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)
image = (
    modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04', add_python='3.12')
    .apt_install('build-essential', 'git')
    .pip_install('uv==0.10.9')
    .add_local_file(str(ROOT / 'training/pyproject.toml'), str(REMOTE_ROOT / 'training/pyproject.toml'), copy=True)
    .add_local_file(str(ROOT / 'training/uv.lock'), str(REMOTE_ROOT / 'training/uv.lock'), copy=True)
    .run_commands('uv sync --frozen --project /opt/thunderhill/training --no-dev')
    # Compile the convolution extension against the pinned CUDA 13 PyTorch wheel.
    .env({'CUDA_HOME': '/usr/local/cuda', 'TORCH_CUDA_ARCH_LIST': '9.0',
          'CAUSAL_CONV1D_FORCE_BUILD': 'TRUE', 'MAX_JOBS': '4',
          'CC': '/usr/bin/gcc', 'CXX': '/usr/bin/g++', 'CUDAHOSTCXX': '/usr/bin/g++'})
    .run_commands(
        'uv pip install --python /opt/thunderhill/training/.venv/bin/python '
        'ninja==1.13.2 setuptools==84.0.0 wheel==0.48.0',
        'uv pip install --python /opt/thunderhill/training/.venv/bin/python '
        '--no-build-isolation torch==2.14.0 causal-conv1d==1.7.0 flash-linear-attention==0.5.2',
        '/opt/thunderhill/training/.venv/bin/python -c "import causal_conv1d; '
        'from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule"',
    )
    .add_local_file(str(GODOT_SOURCE), GODOT, copy=True)
    .run_commands('chmod 755 /usr/local/bin/godot', '/usr/local/bin/godot --headless --version')
    .add_local_dir(str(ROOT / 'godot'), str(REMOTE_ROOT / 'godot'), copy=True,
                   ignore=['.godot', '**/.DS_Store'])
    .run_commands('/usr/local/bin/godot --headless --path /opt/thunderhill/godot --editor --import')
    .env({'HF_HOME': '/model-cache',
          'TORCHINDUCTOR_CACHE_DIR': '/model-cache/torchinductor/torch-2.14.0-h100-qwen',
          'TORCHINDUCTOR_FX_GRAPH_CACHE': '1', 'TOKENIZERS_PARALLELISM': 'false',
          'PYTHONUNBUFFERED': '1', 'PYTHONPATH': '/opt/thunderhill/training'})
)
for filename in TRAINING_FILES:
    image = image.add_local_file(str(ROOT / 'training' / filename), str(REMOTE_ROOT / 'training' / filename))
for filename in TOOL_FILES:
    image = image.add_local_file(str(ROOT / 'tools' / filename), str(REMOTE_ROOT / 'tools' / filename))
image = image.add_local_file(str(Path(__file__)), str(REMOTE_ROOT / 'training/modal_qwen_rl.py'))


def validate_run_id(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', run_id):
        raise ValueError('Run ID must be 1 to 80 letters, digits, underscores or hyphens')


def training_command(output):
    return [PYTHON, '-u', str(REMOTE_ROOT / 'training/train_full_lap_grpo.py'),
            '--godot', GODOT, '--output', str(output), '--qwen27b',
            '--generations', '1', '--initial-generation', '0',
            '--time-budget-seconds', '30', '--batch-candidates', '3',
            '--rollouts-per-generation', '4', '--evaluation-interval', '1',
            '--evaluation-rollouts', '2', '--temperature', '0.6',
            '--wandb-mode', 'online', '--wandb-entity', 'skeptrune-org',
            '--wandb-project', 'thunderhill-rl']


def source_manifest():
    files = [ROOT / 'training' / name for name in TRAINING_FILES]
    files += [ROOT / 'tools' / name for name in TOOL_FILES]
    files += [ROOT / 'training' / name for name in ('pyproject.toml', 'uv.lock', 'modal_qwen_rl.py')]
    return {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        'runtime_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in files},
        'godot_sha256': hashlib.sha256(GODOT_SOURCE.read_bytes()).hexdigest(),
    }


def wandb_secret():
    auth = netrc.netrc().authenticators('api.wandb.ai')
    if not auth or not auth[2]:
        raise RuntimeError('W&B authentication is required in the local netrc before launch')
    return modal.Secret.from_dict({'WANDB_API_KEY': auth[2]})


def execute_run(run_id, source):
    validate_run_id(run_id)
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Missing W&B secret; launch through the local entrypoint')
    artifacts.reload()
    root = RUNS / run_id
    root.mkdir(exist_ok=False)
    command = training_command(root / 'experiment')
    manifest = {
        'run_id': run_id, 'command': command, 'source': source,
        'initialization': 'fresh_pretrained_instruct_with_new_lora',
        'supervised_training_performed': False, 'source_checkpoint': None,
        'gpu_requested': 'H100', 'function_timeout_seconds': FUNCTION_TIMEOUT,
        'child_timeout_seconds': CHILD_TIMEOUT, 'artifact_volume': ARTIFACT_VOLUME,
        'video_rendering': 'Download the complete run and render its video_jobs locally',
    }
    (root / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
    artifacts.commit()
    started = time.monotonic()
    process = None
    status = {'ok': False, 'run_id': run_id}
    try:
        with (root / 'run.log').open('w', buffering=1) as log:
            process = subprocess.Popen(command, cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       start_new_session=True)
            def stream_output():
                for line in process.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
            reader = threading.Thread(target=stream_output, daemon=True)
            reader.start()
            deadline = time.monotonic() + CHILD_TIMEOUT
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, CHILD_TIMEOUT)
                    try:
                        returncode = process.wait(timeout=min(60, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        artifacts.commit()
            except subprocess.TimeoutExpired:
                # Let the trainer audit interrupted attempts while Godot stays alive.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status['timeout'] = True
                returncode = process.returncode
            reader.join(timeout=10)
            if reader.is_alive():
                raise RuntimeError('Child output remained open after termination')
        status.update(ok=returncode == 0 and not status.get('timeout', False),
                      returncode=returncode, elapsed_seconds=time.monotonic() - started)
        if not status['ok']:
            raise RuntimeError(f'Training exited {returncode}; inspect {run_id}/run.log')
        return status
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        artifacts.commit()
        cache.commit()


@app.function(image=image, gpu='H100', cpu=16, memory=131072, timeout=FUNCTION_TIMEOUT,
              max_containers=1, retries=0, include_source=False,
              volumes={str(RUNS): artifacts, '/model-cache': cache})
def run(run_id: str, source: dict):
    return execute_run(run_id, source)


@app.local_entrypoint()
def main(run_id: str):
    validate_run_id(run_id)
    source = source_manifest()
    secret = wandb_secret()
    result = run.with_options(secrets=[secret]).remote(run_id, source)
    print(json.dumps(result, indent=2))
