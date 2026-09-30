"""Serialize local Rev training and serving on the shared GPU."""

import contextlib
import fcntl
from pathlib import Path


@contextlib.contextmanager
def gpu_slot():
    path = Path.home() / ".cache/thunderhill-rev/gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
