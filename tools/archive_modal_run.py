"""Publish one completed Modal run atomically, then render every queued episode.

Downloads remain outside the renderer's watched tree until the Modal CLI has
finished. Failed training runs are archived and rendered exactly like successes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
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


def finished_status(run_id, *, execute=subprocess.run):
    # List before get: absence is pending, but authentication/network/CLI errors
    # propagate rather than masquerading as a job that is still running.
    roots = cli_json(["ls", VOLUME, "/", "--json"], execute=execute)
    if not any(row["filename"].strip("/") == run_id for row in roots):
        return None
    entries = cli_json(["ls", VOLUME, run_id, "--json"], execute=execute)
    if not any(row["filename"].strip("/") == f"{run_id}/status.json" for row in entries):
        return None
    status = cli_json(["get", VOLUME, f"{run_id}/status.json", "-"], execute=execute)
    if status.get("run_id") != run_id or not isinstance(status.get("ok"), bool):
        raise ValueError("Modal status identity or terminal schema mismatch")
    return status


def archive_run(*, run_id, staging_root, output, godot, ffmpeg, watch=False,
                execute=subprocess.run, sleep=time.sleep):
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
            status = finished_status(run_id, execute=execute)
            if status is not None:
                break
            if not watch:
                raise RuntimeError("Run has no terminal status; use --watch to wait")
            print(json.dumps({"waiting_for_run": run_id}), flush=True)
            sleep(30)
        stage = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=staging_root))
        # Modal 1.5.4 requires the directory to exist before recursive downloads.
        execute(["modal", "volume", "get", VOLUME, run_id, str(stage)], check=True)
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
                          "training_ok": status["ok"]}), flush=True)
        execute([sys.executable, str(REPO / "tools/render_video_queue.py"), str(output),
                 "--godot", godot, "--ffmpeg", ffmpeg, "--wait-for-lock"], check=True)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    archive_run(**vars(args))


if __name__ == "__main__":
    main()
