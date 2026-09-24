"""Render every queued rollout, preserving failures and a montage index.

Run against the artifacts root with --watch, or a finished experiment without it.
An experiment has complete video coverage only when every job is in index.json.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from render_run_video import inspect_recording, render_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from video_jobs import enqueue_video


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


def preserve_bytes(path, payload):
    """Recovery is idempotent but cannot overwrite changed evidence."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Recovery evidence changed: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())


def verify_recovery(root, job, source):
    recovery = job["metadata"]["recovery"]
    paths = [(root / recovery[key]).resolve() for key in ("original", "tail")]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Recovery evidence escapes experiment directory")
    original, tail = (path.read_bytes() for path in paths)
    if (
        hashlib.sha256(original).hexdigest() != recovery["original_sha256"]
        or source.read_bytes() + tail != original
    ):
        raise ValueError("Interrupted recording recovery provenance changed")


def recover_interrupted(root):
    """Recover only archived terminal runs, keeping originals and every tail byte.

    A final row without a newline was not durably flushed and is excluded even
    if it happens to parse. Invalid terminated rows or discontinuities fail
    closed. No missing simulator state is synthesized.
    """
    root = Path(root).resolve()
    statuses = sorted(
        set(root.rglob("status.json")) | set(root.rglob("run_status.json"))
    )
    recovered = []
    terminal_runs = 0
    for status_path in statuses:
        run = status_path.parent
        native = status_path.name == "run_status.json"
        launch_path = run / ("launch_manifest.json" if native else "launch.json")
        if not launch_path.is_file():
            continue
        status, launch = (
            json.loads(status_path.read_text()),
            json.loads(launch_path.read_text()),
        )
        terminal = (
            status.get("state") in {"completed", "failed"}
            and status.get("runtime_stopped") is True
            if native
            else type(status.get("ok")) is bool
        )
        if (
            not terminal
            or not status.get("run_id")
            or status["run_id"] != launch.get("run_id")
        ):
            raise ValueError(
                "Recovery requires matching terminal run status and launch"
            )
        terminal_runs += 1
        with (run / ".recording-recovery.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for environment in sorted(run.rglob("environment*")):
                if not environment.is_dir():
                    continue
                attempt = environment.parent
                queue = attempt / "video_jobs"
                queued = {
                    json.loads(p.read_text())["episode_id"]
                    for p in queue.glob("*.json")
                }
                for original in sorted(environment.rglob("*.jsonl")):
                    payload = original.read_bytes()
                    first = payload.partition(b"\n")[0]
                    if not first:
                        continue
                    header = json.loads(first)
                    if (
                        header.get("type") != "episode"
                        or not header.get("policy_display")
                        or header["episode_id"] in queued
                    ):
                        continue
                    boundary = payload.rfind(b"\n") + 1
                    if not boundary:
                        raise ValueError("Interrupted episode has no flushed header")
                    identity = hashlib.sha256(header["episode_id"].encode()).hexdigest()
                    directory = attempt / "recovered_recordings" / identity
                    directory.mkdir(parents=True, exist_ok=True)
                    source, tail = (
                        directory / "flushed-prefix.jsonl",
                        directory / "unflushed-tail.bin",
                    )
                    preserve_bytes(source, payload[:boundary])
                    preserve_bytes(tail, payload[boundary:])
                    info = inspect_recording(source)
                    recovery = dict(
                        original=str(original.relative_to(attempt)),
                        original_sha256=hashlib.sha256(payload).hexdigest(),
                        tail=str(tail.relative_to(attempt)),
                        unflushed_tail_bytes=len(payload) - boundary,
                        recovered_transitions=info["transitions"],
                        final_tick=info["final_tick"],
                    )
                    metadata = dict(
                        kind="interrupted_attempt",
                        stop_reason="interrupted",
                        training_eligible=False,
                        episode_complete=False,
                        recovery=recovery,
                    )
                    job = enqueue_video(
                        attempt, source, metadata=metadata, full_episode=False
                    )
                    recovered.append(str(job.relative_to(root)))
                    queued.add(header["episode_id"])
    if not terminal_runs:
        raise ValueError("Recovery requires an archived terminal run")
    return recovered


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


def process_queue(
    directory, *, godot, ffmpeg, fps=30, retry_failed=False, wait_for_lock=False
):
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
            interrupted = (
                job.get("full_episode") is False
                and job.get("metadata", {}).get("stop_reason") == "interrupted"
                and job.get("metadata", {}).get("recovery")
            )
            if job.get("schema_version") != 1 or (
                job.get("full_episode") is not True and not interrupted
            ):
                raise ValueError(f"Unsupported video job: {job_path}")
            source = (root / job["source"]).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Video source escapes experiment directory")
            target = videos / job_path.stem
            target.mkdir(exist_ok=True)
            receipt_path = target / "complete.json"
            if digest(source) != job["source_sha256"]:
                raise ValueError(f"Recording changed after queueing: {source}")
            if interrupted:
                verify_recovery(root, job, source)
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
                    "full_episode": job["full_episode"],
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


def process_queues(directories, *, workers=1, watch=False, **kwargs):
    """Bound independent queues, retaining exclusive locks and safe completion.

    Only one queue per worker is submitted. On an error, currently running
    recordings finish before propagation, and no more queues are submitted.
    """
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("Video workers must be 1 or 2")

    def process(directory):
        try:
            return process_queue(directory, **kwargs)
        except BlockingIOError:
            if not watch:
                raise
            return None

    remaining = iter(directories)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def submit_next():
            directory = next(remaining, None)
            if directory is not None:
                pending[pool.submit(process, directory)] = directory

        for _ in range(workers):
            submit_next()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            results = [(pending.pop(future), future.result()) for future in completed]
            for directory, report in results:
                yield directory, report
                submit_next()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2),
        default=2,
        help="Independent video queues to render concurrently (default: 2)",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help="Queue preserved prefixes from an archived terminal run",
    )
    parser.add_argument(
        "--wait-for-lock",
        action="store_true",
        help="Wait for another renderer to finish this queue",
    )
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Root must exist")
    if args.fps <= 0:
        parser.error("FPS must be positive")
    if args.watch and args.retry_failed:
        parser.error("Retry failed attempts once, without --watch")
    if args.recover_interrupted:
        if args.watch:
            parser.error("Recovery is only for archived terminal runs, without --watch")
        print(
            json.dumps({"recovered_interrupted_jobs": recover_interrupted(args.root)}),
            flush=True,
        )
    scanned = {}
    while True:
        failures = 0
        queues = (
            [args.root]
            if args.root.name == "video_jobs"
            else sorted(args.root.rglob("video_jobs"))
        )
        if not queues and not args.watch:
            if args.recover_interrupted:
                print(
                    json.dumps({"queues": 0, "failures": 0, "state": "no_recordings"})
                )
                return
            parser.error("No video queues found")
        stamps = {directory: directory.stat().st_mtime_ns for directory in queues}
        pending = [
            directory
            for directory in queues
            if not args.watch or scanned.get(directory) != stamps[directory]
        ]
        for directory, report in process_queues(
            pending,
            workers=args.workers,
            watch=args.watch,
            godot=args.godot,
            ffmpeg=args.ffmpeg,
            fps=args.fps,
            retry_failed=args.retry_failed,
            wait_for_lock=args.wait_for_lock,
        ):
            if report is None:
                continue
            failures += len(report["failures"])
            if report["failures"]:
                print(
                    json.dumps(
                        {"queue": str(directory), "failures": report["failures"]}
                    ),
                    flush=True,
                )
            scanned[directory] = stamps[directory]
        if not args.watch:
            print(json.dumps({"queues": len(queues), "failures": failures}))
            raise SystemExit(1 if failures else 0)
        time.sleep(10)


if __name__ == "__main__":
    main()
