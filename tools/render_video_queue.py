"""Render every queued rollout, preserving failures and a montage index.

Run against the artifacts root with --watch, or a finished experiment without it.
An experiment has complete video coverage only when every job is in index.json.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from render_run_video import render_video


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def acquire_renderer_lock(lock, *, wait=False):
    """Wait only for an active renderer, never retry rendering failures."""
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if not wait:
                raise
            time.sleep(1)


def process_queue(directory, *, godot, ffmpeg, fps=30, retry_failed=False, wait_for_lock=False):
    """Lock one experiment, resume verified results, and retain failed attempts."""
    directory = Path(directory).resolve()
    root = directory.parent
    videos = root / "videos"
    videos.mkdir(exist_ok=True)
    with (directory / ".renderer.lock").open("a") as lock:
        acquire_renderer_lock(lock, wait=wait_for_lock)
        entries, failures = [], []
        for job_path in sorted(directory.glob("*.json")):
            job = json.loads(job_path.read_text())
            job_hash = digest(job_path)
            if job.get("schema_version") != 1 or job.get("full_episode") is not True:
                raise ValueError(f"Unsupported video job: {job_path}")
            source = (root / job["source"]).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Video source escapes experiment directory")
            target = videos / job_path.stem
            target.mkdir(exist_ok=True)
            receipt_path = target / "complete.json"
            if digest(source) != job["source_sha256"]:
                raise ValueError(f"Recording changed after queueing: {source}")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                movie = (root / receipt["video"]).resolve()
                if (
                    receipt["job_sha256"] != job_hash
                    or not movie.is_relative_to(target)
                    or digest(movie) != receipt["video_sha256"]
                ):
                    raise ValueError(f"Completed video provenance changed: {target}")
                entries.append(receipt)
                continue
            attempts = list(target.glob("attempt-*"))
            if attempts and not retry_failed:
                failures.append(
                    {
                        "job": job_path.name,
                        "reason": "Previous attempt incomplete; inspect logs then use --retry-failed",
                    }
                )
                continue
            attempt = target / f"attempt-{len(attempts) + 1:04d}"
            attempt.mkdir()
            output = attempt / "run.mp4"
            try:
                rendered = render_video(
                    source, output, godot=godot, ffmpeg=ffmpeg, fps=fps
                )
                if digest(source) != job["source_sha256"]:
                    raise ValueError("Recording changed while rendering")
                receipt = {
                    "episode_id": job["episode_id"],
                    "job": str(job_path.relative_to(root)),
                    "job_sha256": job_hash,
                    "source_sha256": job["source_sha256"],
                    "video": str(output.relative_to(root)),
                    "video_sha256": digest(output),
                    "policy_display": job["policy_display"],
                    "metadata": job["metadata"],
                    "render": rendered,
                }
                write_json(receipt_path, receipt)
                entries.append(receipt)
                print(
                    json.dumps(
                        {"video_complete": str(output), "episode_id": job["episode_id"]}
                    ),
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "job": job_path.name,
                    "error": str(error),
                    "type": type(error).__name__,
                }
                write_json(attempt / "error.json", failure)
                failures.append(failure)
        # Sort for montage assembly by generation and rollout, retaining both
        # evaluations and failures. Each entry includes its original identity.
        queued_episodes = {
            json.loads(path.read_text())["episode_id"]
            for path in directory.glob("*.json")
        }
        # Catch episodes left behind by interruption before queue publication.
        # Worker bootstrap and snapshot preparation have empty display metadata.
        recorded_episodes = set()
        for environment in root.glob("environment*"):
            if not environment.is_dir():
                continue
            for recording in environment.rglob("*.jsonl"):
                with recording.open() as stream:
                    first = stream.readline()
                if not first:
                    continue
                header = json.loads(first)
                if header.get("type") == "episode" and header.get("policy_display"):
                    recorded_episodes.add(header["episode_id"])
        missing = sorted(recorded_episodes - queued_episodes)
        failures.extend(
            {
                "episode_id": episode,
                "reason": "Recorded policy episode has no video job",
            }
            for episode in missing
        )
        entries.sort(
            key=lambda r: (
                r["policy_display"].get("generation", -1),
                r["policy_display"].get("rollout_number", -1),
                r["episode_id"],
            )
        )
        report = {
            "complete": not failures,
            "videos": entries,
            "failures": failures,
            "queued": len(queued_episodes),
            "unqueued_episodes": missing,
        }
        write_json(videos / "index.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--wait-for-lock", action="store_true",
                        help="Wait for another renderer to finish this queue")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Root must exist")
    if args.fps <= 0:
        parser.error("FPS must be positive")
    if args.watch and args.retry_failed:
        parser.error("Retry failed attempts once, without --watch")
    scanned = {}
    while True:
        failures = 0
        queues = (
            [args.root]
            if args.root.name == "video_jobs"
            else sorted(args.root.rglob("video_jobs"))
        )
        if not queues and not args.watch:
            parser.error("No video queues found")
        for directory in queues:
            stamp = directory.stat().st_mtime_ns
            if args.watch and scanned.get(directory) == stamp:
                continue
            try:
                report = process_queue(
                    directory,
                    godot=args.godot,
                    ffmpeg=args.ffmpeg,
                    fps=args.fps,
                    retry_failed=args.retry_failed,
                    wait_for_lock=args.wait_for_lock,
                )
                failures += len(report["failures"])
                if report["failures"]:
                    print(
                        json.dumps(
                            {"queue": str(directory), "failures": report["failures"]}
                        ),
                        flush=True,
                    )
                scanned[directory] = stamp
            except BlockingIOError:
                if not args.watch:
                    raise
        if not args.watch:
            print(json.dumps({"queues": len(queues), "failures": failures}))
            raise SystemExit(1 if failures else 0)
        time.sleep(10)


if __name__ == "__main__":
    main()
