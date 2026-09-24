"""Publish one completed Modal run atomically, then render every queued episode.

Downloads remain outside the renderer's watched tree until the Modal CLI has
finished. Failed training runs are archived and rendered exactly like successes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time

VOLUME = "thunderhill-runs-v2"
REPO = Path(__file__).resolve().parents[1]


def cli_json(arguments, *, execute=subprocess.run):
    result = execute(["modal", "volume", *arguments], check=True, capture_output=True,
                     text=True, timeout=60)
    return json.loads(result.stdout)


def finished_status(run_id, *, staging_root, execute=subprocess.run):
    # List before get: absence is pending, but authentication/network/CLI errors
    # propagate rather than masquerading as a job that is still running.
    roots = cli_json(["ls", VOLUME, "/", "--json"], execute=execute)
    if not any(row["filename"].strip("/") == run_id for row in roots):
        return None
    entries = cli_json(["ls", VOLUME, run_id, "--json"], execute=execute)
    if not any(row["filename"].strip("/") == f"{run_id}/status.json" for row in entries):
        return None
    # The get command appends a human completion footer even when its target is
    # stdout. Read the downloaded file instead of treating CLI output as JSON.
    with tempfile.TemporaryDirectory(prefix="status-", dir=staging_root) as directory:
        destination = Path(directory) / "status.json"
        execute(["modal", "volume", "get", VOLUME, f"{run_id}/status.json", str(destination)],
                check=True, capture_output=True, text=True, timeout=60)
        status = json.loads(destination.read_text())
    if status.get("run_id") != run_id or not isinstance(status.get("ok"), bool):
        raise ValueError("Modal status identity or terminal schema mismatch")
    return status


def download_volume_run(volume, run_id, stage, *, workers=4, resume=False):
    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    with (stage / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _download_volume_run(volume, run_id, stage, workers=workers, resume=resume)


def _download_volume_run(volume, run_id, stage, *, workers, resume):
    """Finish the inventory RPC before starting bounded, atomic file downloads."""
    if not 1 <= workers <= 16:
        raise ValueError('Download workers must be between 1 and 16')
    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    entries = list(volume.iterdir(run_id, recursive=True))
    inventory = []
    seen = set()
    for entry in entries:
        relative = PurePosixPath(entry.path.lstrip('/'))
        if ('..' in relative.parts or not relative.parts or relative.parts[0] != run_id
                or str(relative) in seen):
            raise ValueError(f'Unsafe or duplicate volume path: {entry.path}')
        seen.add(str(relative))
        destination = stage.joinpath(*relative.parts)
        components = [destination, *destination.parents]
        if any(part.is_symlink() for part in components if part.is_relative_to(stage)):
            raise ValueError(f'Unexpected local symlink: {entry.path}')
        if not destination.resolve().is_relative_to(stage):
            raise ValueError(f'Volume path escapes staging: {entry.path}')
        kind = int(entry.type)
        if kind not in (1, 2, 3):
            raise ValueError(f'Unsupported volume entry type: {entry.path}')
        if entry.size < 0:
            raise ValueError(f'Invalid volume file size: {entry.path}')
        inventory.append(dict(path=str(relative), type=kind, size=entry.size,
                              mtime=getattr(entry, 'mtime', None)))
    if not inventory:
        raise ValueError('Volume run inventory is empty')
    # Modal exposes link metadata but no target. W&B diagnostic links may point
    # outside the run (for example debug-core.log in a temporary directory).
    # Preserve their metadata explicitly; download every concrete file. A link
    # anywhere else could replace required training evidence and fails closed.
    wandb_root = PurePosixPath(run_id) / 'experiment' / 'wandb'
    for entry in inventory:
        if entry['type'] != 3:
            continue
        path = PurePosixPath(entry['path'])
        if not path.is_relative_to(wandb_root):
            raise ValueError(f"Unsupported volume symlink: {entry['path']}")
        entry.update(archive_action='metadata_only_wandb_diagnostic_symlink',
                     target_available=False)
    receipt_path = stage / 'download-inventory.json'
    if resume:
        if sorted(json.loads(receipt_path.read_text()), key=lambda item: item['path']) != sorted(
                inventory, key=lambda item: item['path']):
            raise ValueError('Remote inventory changed since the original download')
    else:
        with receipt_path.open('x') as receipt:
            json.dump(inventory, receipt, indent=2)
    pending = []
    completed_bytes = completed_files = 0
    for entry in inventory:
        destination = stage / entry['path']
        if entry['type'] == 3:
            continue
        if entry['type'] == 2:
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not resume:
                    raise FileExistsError(destination)
                if not destination.is_file() or destination.stat().st_size != entry['size']:
                    raise ValueError(f'Invalid completed staging file: {destination}')
                # These final names were created only after the original writer
                # fsynced and size checked its temporary file, using atomic link.
                completed_files += 1
                completed_bytes += entry['size']
            else:
                pending.append(entry)
    print(json.dumps({'resumed_files': completed_files, 'resumed_bytes': completed_bytes,
                      'pending_files': len(pending), 'download_workers': workers}), flush=True)

    def download(entry):
        destination = stage / entry['path']
        # Failed or interrupted files stay visibly partial in the staging tree.
        fd, partial = tempfile.mkstemp(prefix=destination.name + '.', suffix='.partial',
                                       dir=destination.parent)
        with os.fdopen(fd, 'wb') as stream:
            transferred = volume.read_file_into_fileobj(entry['path'], stream)
            stream.flush()
            os.fsync(stream.fileno())
        if transferred != entry['size'] or os.stat(partial).st_size != entry['size']:
            raise ValueError(f"Downloaded size mismatch: {entry['path']}")
        # Atomic publication without replacing any existing file.
        os.link(partial, destination)
        os.unlink(partial)
        return entry['size']

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(download, item) for item in pending]
        sizes = []
        for future in as_completed(futures):
            sizes.append(future.result())
            if len(sizes) % 256 == 0:
                print(json.dumps({'downloaded_files': len(sizes),
                                  'pending_files': len(pending) - len(sizes),
                                  'resumed_files': completed_files}), flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    print(json.dumps({'downloaded_files': len(sizes), 'downloaded_bytes': sum(sizes),
                      'download_workers': workers,
                      'wandb_aliases_metadata_only': sum(item['type'] == 3 for item in inventory)}), flush=True)


def download_with_sdk(run_id, stage, *, modal_python, execute=subprocess.run,
                      workers=4, resume=False):
    command = [str(modal_python), str(Path(__file__).resolve()), 'download',
               '--run-id', run_id, '--stage', str(stage), '--download-workers', str(workers)]
    if resume:
        command.append('--resume')
    execute(command, check=True)


def archive_run(*, run_id, staging_root, output, godot, ffmpeg, watch=False,
                execute=subprocess.run, sleep=time.sleep, modal_python=None, downloader=None,
                resume_stage=None, download_workers=4):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("Invalid run ID")
    if not 1 <= download_workers <= 16:
        raise ValueError('Download workers must be between 1 and 16')
    staging_root, output = Path(staging_root).resolve(), Path(output).resolve()
    watched = (REPO / "artifacts").resolve()
    if staging_root.is_relative_to(watched) or staging_root.is_relative_to(output.parent):
        raise ValueError("Staging must be outside the watched artifacts/output tree")
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if staging_root.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("Staging and output must share a filesystem for atomic publication")
    # All invocations of this helper share an output lock. Existing archives are
    # never replaced, and incomplete staging directories remain for diagnosis.
    with (output.parent / f".{output.name}.archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError(f"Archive already exists: {output}")
        while True:
            status = finished_status(run_id, staging_root=staging_root, execute=execute)
            if status is not None:
                break
            if not watch:
                raise RuntimeError("Run has no terminal status; use --watch to wait")
            print(json.dumps({"waiting_for_run": run_id}), flush=True)
            sleep(30)
        if resume_stage is not None:
            stage = Path(resume_stage).resolve()
            if (stage.parent != staging_root or not stage.is_dir()
                    or not stage.name.startswith(run_id + '-')):
                raise ValueError('Resume stage must be an existing run directory in staging root')
            local_status = json.loads((stage / run_id / 'status.json').read_text())
            local_launch = json.loads((stage / run_id / 'launch.json').read_text())
            if local_status != status or local_launch.get('run_id') != run_id:
                raise ValueError('Resume stage run status or launch identity mismatch')
            with tempfile.TemporaryDirectory(prefix='launch-', dir=staging_root) as directory:
                launch_path = Path(directory) / 'launch.json'
                execute(['modal', 'volume', 'get', VOLUME, f'{run_id}/launch.json', str(launch_path)],
                        check=True, capture_output=True, text=True, timeout=60)
                if json.loads(launch_path.read_text()) != local_launch:
                    raise ValueError('Resume stage launch differs from remote launch')
        else:
            stage = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=staging_root))
        if downloader is not None:
            downloader(run_id, stage)
        else:
            if modal_python is None:
                raise ValueError('Supply --modal-python with an interpreter containing the Modal SDK')
            download_with_sdk(run_id, stage, modal_python=modal_python, execute=execute,
                              workers=download_workers, resume=resume_stage is not None)
        downloaded = stage / run_id
        local_status = json.loads((downloaded / "status.json").read_text())
        launch = json.loads((downloaded / "launch.json").read_text())
        if local_status != status or launch.get("run_id") != run_id:
            raise ValueError("Downloaded run identity or status differs from terminal source")
        if output.exists():
            raise FileExistsError(f"Archive already exists: {output}")
        os.rename(stage, output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        print(json.dumps({"archive": str(output), "run_id": run_id,
                          "execution_ok": status["ok"],
                          "outcome": status.get("outcome", "validation")}), flush=True)
        if not status['ok'] and not (output / run_id / 'experiment').exists():
            print(json.dumps({'run_id': run_id, 'video_status': 'no_training_attempt_started'}),
                  flush=True)
            return output
        execute([sys.executable, str(REPO / "tools/render_video_queue.py"), str(output),
                 "--godot", godot, "--ffmpeg", ffmpeg, "--wait-for-lock",
                 "--recover-interrupted"], check=True)
        return output


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        parser = argparse.ArgumentParser(description='Download an inventoried Modal run')
        parser.add_argument('--run-id', required=True)
        parser.add_argument('--stage', type=Path, required=True)
        parser.add_argument('--download-workers', type=int, default=4)
        parser.add_argument('--resume', action='store_true')
        args = parser.parse_args(sys.argv[2:])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', args.run_id):
            raise ValueError('Invalid run ID')
        import modal
        volume = modal.Volume.from_name(VOLUME)
        download_volume_run(volume, args.run_id, args.stage,
                            workers=args.download_workers, resume=args.resume)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--resume-stage", type=Path)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--modal-python", type=Path, required=True)
    args = parser.parse_args()
    archive_run(**vars(args))


if __name__ == "__main__":
    main()
