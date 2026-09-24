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


def download_volume_run(volume, run_id, stage, *, workers=4):
    """Finish the inventory RPC before starting bounded, atomic file downloads."""
    if not 1 <= workers <= 8:
        raise ValueError('Download workers must be between 1 and 8')
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
    # Modal's public FileEntry exposes no link target. Do not dereference links.
    # W&B's three convenience aliases are redundant with concrete run logs;
    # retain their listing metadata and make the omission explicit in the receipt.
    wandb_root = PurePosixPath(run_id) / 'experiment' / 'wandb'
    for entry in inventory:
        if entry['type'] != 3:
            continue
        path = PurePosixPath(entry['path'])
        if path.parent != wandb_root or path.name not in ('debug.log', 'debug-internal.log', 'latest-run'):
            raise ValueError(f"Unsupported volume symlink: {entry['path']}")
        concrete = []
        for candidate in inventory:
            other = PurePosixPath(candidate['path'])
            if candidate['type'] != 1 or not other.is_relative_to(wandb_root):
                continue
            components = other.relative_to(wandb_root).parts
            if not components or not components[0].startswith(('run-', 'offline-run-')):
                continue
            if path.name == 'latest-run' or components[1:] == ('logs', path.name):
                concrete.append(candidate['path'])
        if not concrete:
            raise ValueError(f"W&B alias has no concrete archived log files: {entry['path']}")
        entry.update(archive_action='metadata_only_wandb_convenience_symlink',
                     target_available=False, concrete_run_files=concrete)
    with (stage / 'download-inventory.json').open('x') as receipt:
        json.dump(inventory, receipt, indent=2)
    for entry in inventory:
        destination = stage / entry['path']
        if entry['type'] == 3:
            continue
        if entry['type'] == 2:
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)

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
        futures = [pool.submit(download, item) for item in inventory if item['type'] == 1]
        sizes = [future.result() for future in as_completed(futures)]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    print(json.dumps({'downloaded_files': len(sizes), 'downloaded_bytes': sum(sizes),
                      'download_workers': workers,
                      'wandb_aliases_metadata_only': sum(item['type'] == 3 for item in inventory)}), flush=True)


def download_with_sdk(run_id, stage, *, modal_python, execute=subprocess.run):
    execute([str(modal_python), str(Path(__file__).resolve()), 'download',
             '--run-id', run_id, '--stage', str(stage)], check=True)


def archive_run(*, run_id, staging_root, output, godot, ffmpeg, watch=False,
                execute=subprocess.run, sleep=time.sleep, modal_python=None, downloader=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("Invalid run ID")
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
        stage = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=staging_root))
        if downloader is not None:
            downloader(run_id, stage)
        else:
            if modal_python is None:
                raise ValueError('Supply --modal-python with an interpreter containing the Modal SDK')
            download_with_sdk(run_id, stage, modal_python=modal_python, execute=execute)
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
        args = parser.parse_args(sys.argv[2:])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', args.run_id):
            raise ValueError('Invalid run ID')
        import modal
        volume = modal.Volume.from_name(VOLUME)
        download_volume_run(volume, args.run_id, args.stage)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--modal-python", type=Path, required=True)
    args = parser.parse_args()
    archive_run(**vars(args))


if __name__ == "__main__":
    main()
