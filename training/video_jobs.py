"""Durable, immutable video work items for flushed simulator episodes.

Jobs contain paths relative to the run directory, so the entire run can move
between local storage and a Modal volume. Publishing never replaces a job.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def enqueue_video(out: Path, source: Path, *, metadata: dict, full_episode: bool = True) -> Path:
    """Publish one full episode render job, or verify an identical existing job.

    The caller must flush the recorder before enqueueing and must not append to
    it afterwards. The consumer must recheck source_sha256 before rendering.
    Conflicting identity, content, or metadata raises ValueError.
    """
    root = Path(out).resolve(strict=True)
    recording = Path(source).resolve(strict=True)
    try:
        relative = recording.relative_to(root)
    except ValueError as error:
        raise ValueError("Video source must be inside the run directory") from error
    if not recording.is_file():
        raise ValueError("Video source must be a recording file")
    if not isinstance(metadata, dict):
        raise ValueError("Video metadata must be an object")
    if type(full_episode) is not bool or (not full_episode and
            (metadata.get("stop_reason") != "interrupted" or not metadata.get("recovery"))):
        raise ValueError("Incomplete episodes require explicit interruption provenance")

    digest = hashlib.sha256()
    with recording.open("rb") as stream:
        before = os.fstat(stream.fileno())
        first = stream.readline()
        try:
            header = json.loads(first)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Video source has no valid episode header") from error
        if not isinstance(header, dict) or header.get("type") != "episode":
            raise ValueError("Video source must begin with an episode header")
        identity = header.get("episode_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Video source needs a nonempty episode_id")
        display = header.get("policy_display")
        if not isinstance(display, dict):
            raise ValueError("Video source needs a policy_display object")
        digest.update(first)
        last = first
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            last = chunk
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Video source changed while enqueueing")
    if not last.endswith(b"\n"):
        raise ValueError("Video source ends in an unflushed or partial record")

    job = {
        "schema_version": 1,
        "episode_id": identity,
        "source": relative.as_posix(),
        "source_sha256": digest.hexdigest(),
        "policy_display": display,
        "metadata": metadata,
        "full_episode": full_episode,
    }
    # Round trip also detaches caller owned nested metadata. Reject NaN/Infinity.
    payload = json.dumps(job, sort_keys=True, indent=2, allow_nan=False) + "\n"
    job = json.loads(payload)
    queue = root / "video_jobs"
    queue.mkdir(exist_ok=True)
    path = queue / (hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=queue, prefix=".pending-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link atomically publishes complete bytes and is exclusive,
            # unlike replace/rename which could overwrite concurrent enqueuers.
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("Existing video job is corrupt") from error
            if existing != job:
                raise ValueError("Conflicting video job for the same episode")
        directory = os.open(queue, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
