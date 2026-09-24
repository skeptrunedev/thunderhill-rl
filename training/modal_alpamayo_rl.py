"""Cost bounded Alpamayo generation; prepare all model files before GPU billing."""
from pathlib import Path
import hashlib
import json
import netrc
import os
import subprocess
import codecs
import selectors
import signal
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/opt/thunderhill'
PYTHON = REMOTE + '/training/alpamayo/.venv/bin/python'
MODEL = 'nvidia/Alpamayo-1.5-10B'
REVISION = '7aba8293c09993f2e125c6819df05d7fa3e873ea'
PROCESSOR = 'nvidia/Cosmos-Reason2-8B'
PROCESSOR_REVISION = 'a9fae2cf89dc64db96b12860417f0eb403013bb9'
GODOT_SOURCE = Path.home() / '.local/share/thunderhill-tools/godot-4.7.2/Godot_v4.7.2-stable_linux.x86_64'
SOURCES = ('train_alpamayo_rl.py', 'alpamayo_policy.py', 'driving_episode.py',
           'driving_trajectory.py', 'lap_episode.py', 'lap_policy.py', 'lap_audit.py',
           'lap_rollout.py', 'agent_harness.py', 'video_jobs.py',
           'experiment_tracking.py')
app = modal.App('thunderhill-alpamayo-gameplay-rl')
cache = modal.Volume.from_name('thunderhill-huggingface-v2', create_if_missing=True, version=2)
runs = modal.Volume.from_name('thunderhill-runs-v2', create_if_missing=True, version=2)
download_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('huggingface-hub==0.36.0')
    .env({'PYTHONPATH': REMOTE + '/training'})
    .add_local_file(str(Path(__file__)), REMOTE + '/training/modal_alpamayo_rl.py'))
image = (modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04', add_python='3.12')
    .apt_install('git', 'build-essential', 'xvfb', 'xauth', 'libgl1-mesa-dri', 'libglx-mesa0',
                 'libvulkan1', 'mesa-vulkan-drivers', 'libxcursor1', 'libxinerama1', 'libxi6', 'libxrandr2')
    .pip_install('uv==0.10.9')
    .run_commands(
        'git clone https://github.com/NVlabs/alpamayo.git /opt/alpamayo && cd /opt/alpamayo && git checkout 4cda35d22bb257f0936ac272397627b9309ca211',
        'git clone https://github.com/NVlabs/alpamayo-recipes.git /opt/alpamayo-recipes && cd /opt/alpamayo-recipes && git checkout 1d99fc50370c96637157455da386323a44663c9d',
        'git clone https://github.com/NVlabs/alpagym.git /opt/alpagym && cd /opt/alpagym && git checkout 972d160eed0e23d388497851504a3a233fec5879')
    .add_local_file(str(ROOT / 'training/alpamayo/pyproject.toml'), REMOTE + '/training/alpamayo/pyproject.toml', copy=True)
    .add_local_file(str(ROOT / 'training/alpamayo/uv.lock'), REMOTE + '/training/alpamayo/uv.lock', copy=True)
    .run_commands('uv sync --frozen --project ' + REMOTE + '/training/alpamayo')
    .add_local_file(str(GODOT_SOURCE), '/usr/local/bin/godot', copy=True)
    .add_local_dir(str(ROOT / 'godot'), REMOTE + '/godot', copy=True, ignore=['.godot', '**/.DS_Store'])
    .run_commands('chmod 755 /usr/local/bin/godot',
                  'godot --headless --path ' + REMOTE + '/godot --editor --import')
    .env({'PYTHONPATH': REMOTE + '/training:/opt/alpamayo/src:/opt/alpamayo-recipes/recipes:/opt/alpagym/packages/policies/alpamayo_r1/src',
          'HF_HOME': '/model-cache', 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
          'PYTHONUNBUFFERED': '1', 'LIBGL_ALWAYS_SOFTWARE': '1', 'ALPAMAYO_SOURCE_ROOT': '/opt'}))
for name in SOURCES:
    image = image.add_local_file(str(ROOT / 'training' / name), REMOTE + '/training/' + name)
for name in ('check_parallel.py', 'check_agent.py'):
    image = image.add_local_file(str(ROOT / 'tools' / name), REMOTE + '/tools/' + name)
image = image.add_local_file(str(Path(__file__)), REMOTE + '/training/modal_alpamayo_rl.py')


def download_model_files(cache_dir):
    """Verify both gated repositories before fetching any large weight shard."""
    from huggingface_hub import hf_hub_download, snapshot_download
    for model, revision in ((MODEL, REVISION), (PROCESSOR, PROCESSOR_REVISION)):
        hf_hub_download(model, 'config.json', revision=revision, cache_dir=cache_dir)
    processor = snapshot_download(PROCESSOR, revision=PROCESSOR_REVISION,
        cache_dir=cache_dir, allow_patterns=['*.json', '*.txt', '*.jinja', 'merges.txt'])
    checkpoint = snapshot_download(MODEL, revision=REVISION, cache_dir=cache_dir)
    return dict(checkpoint=checkpoint, processor_path=processor)


def signal_process_group(process, signum):
    """The wrapper can exit before descendants, so do not predicate on poll()."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def run_logged_process(command, *, cwd, log_path, timeout_seconds,
                       heartbeat, grace_seconds=60, heartbeat_seconds=30):
    """Drain logs in the supervising thread and always reap the owned group.

    No background log writer can outlive the log file. SIGINT allows the trainer
    to flush its episode; a fixed grace deadline then enforces process cleanup.
    """
    started = time.monotonic()
    next_heartbeat = started + heartbeat_seconds
    interrupt_at = None
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with log_path.open('w', buffering=1) as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)

            def drain(wait):
                for key, _ in selector.select(wait):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        text = decoder.decode(chunk)
                        log.write(text)
                        print(text, end='', flush=True)

            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if interrupt_at is None and now - started >= timeout_seconds:
                        interrupt_at = now
                        signal_process_group(process, signal.SIGINT)
                    if interrupt_at is not None and now - interrupt_at >= grace_seconds:
                        signal_process_group(process, signal.SIGKILL)
                        break
                    if now >= next_heartbeat:
                        heartbeat()
                        next_heartbeat = now + heartbeat_seconds
                    drain(min(.1, max(.01, timeout_seconds - (now - started))))
            finally:
                # Also kills inherited stdout holders after the wrapper exits.
                signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=10)
                deadline = time.monotonic() + 2
                while selector.get_map() and time.monotonic() < deadline:
                    drain(.05)
                text = decoder.decode(b'', final=True)
                log.write(text)
                if text:
                    print(text, end='', flush=True)
                process.stdout.close()
        if interrupt_at is not None:
            raise TimeoutError('GPU run reached its bounded runtime')
        if process.returncode != 0:
            raise RuntimeError(f'Training process exited {process.returncode}')
        return process.returncode


@app.function(image=download_image, cpu=2, memory=4096, timeout=1800, retries=0,
              volumes={'/model-cache': cache}, include_source=False)
def prepare_weights():
    cache.reload()
    paths = download_model_files('/model-cache/hub')
    cache.commit()
    return paths


@app.function(image=image, gpu='RTX-PRO-6000', cpu=8, memory=65536,
              timeout=1800, retries=0, max_containers=1, include_source=False,
              volumes={'/model-cache': cache, '/runs': runs})
def run_generation(run_id: str, paths: dict, source: dict):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid run ID')
    cache.reload()
    runs.reload()
    root = Path('/runs') / run_id
    root.mkdir(exist_ok=False)
    command = ['xvfb-run', '-a', '-s', '-screen 0 800x600x24', PYTHON,
        '-u', REMOTE + '/training/train_alpamayo_rl.py', '--checkpoint', paths['checkpoint'],
        '--processor-path', paths['processor_path'], '--godot', '/usr/local/bin/godot',
        '--output', str(root / 'experiment'), '--seconds', '12', '--rollouts', '2']
    (root / 'launch.json').write_text(json.dumps(dict(run_id=run_id, command=command, source=source,
        gpu='RTX-PRO-6000', child_timeout_seconds=1500, supervised_training_performed=False), indent=2))
    runs.commit()
    started = time.monotonic()
    status = dict(run_id=run_id, ok=False)
    try:
        run_logged_process(command, cwd=REMOTE, log_path=root / 'run.log',
            timeout_seconds=1500, heartbeat=runs.commit)
        campaign = json.loads((root / 'experiment/campaign.json').read_text())
        if not campaign['complete'] or len(campaign['generations']) != 1:
            raise RuntimeError('Missing completed generation evidence')
        status['ok'] = True
        return dict(status)
    except BaseException as error:
        status['error'] = repr(error)
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic() - started
        (root / 'status.json').write_text(json.dumps(status, indent=2))
        runs.commit()


@app.local_entrypoint()
def main(run_id: str):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid run ID')
    auth = netrc.netrc().authenticators('api.wandb.ai')
    if not auth or not auth[2]:
        raise RuntimeError('W&B authentication missing')
    credentials = {'WANDB_API_KEY': auth[2]}
    token_file = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'token'
    token = os.environ.get('HF_TOKEN') or (token_file.read_text().strip() if token_file.exists() else None)
    if token:
        credentials['HF_TOKEN'] = token
    secret = modal.Secret.from_dict(credentials)
    files = [ROOT / 'training' / name for name in (*SOURCES, 'modal_alpamayo_rl.py')]
    source = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        runtime_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    paths = prepare_weights.with_options(secrets=[secret]).remote()
    print(json.dumps({'model_files_prepared': paths}), flush=True)
    print(json.dumps(run_generation.with_options(secrets=[secret]).remote(run_id, paths, source)))
