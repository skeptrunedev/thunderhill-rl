#!/usr/bin/env python3
"""Forced SSH command for a Modal container's key on the rendering host.

Only two requests are accepted, both confined to a mirror directory: a Godot
launch from tools/remote_godot.py (container paths are mapped into the mirror,
the container's repository into a worktree at the requested revision) and an
rsync server for copying files between the container and the mirror.
Installed by tools/host_remote_godot.py; standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

def mirrored(mirror: Path, path: str) -> str:
    if not path.startswith("/") or ".." in Path(path).parts:
        raise ValueError(f"Expected a normalized absolute container path: {path!r}")
    return str(mirror / path.lstrip("/"))


def run_godot(payload: dict, mirror: Path, worktrees: Path, godot: str, repo: str) -> int:
    worktree = worktrees / payload["revision"]
    if not (worktree / "godot/project.godot").is_file():
        raise RuntimeError(f"No prepared worktree for revision {payload['revision']}")
    args = []
    iterator = iter(payload["args"])
    for arg in iterator:
        if arg == "--path":
            value = next(iterator)
            if not value.startswith(repo + "/"):
                raise ValueError(f"Godot project must be inside the repository: {value}")
            args += [arg, str(worktree / value[len(repo) + 1:])]
        elif arg == "--write-movie":
            value = mirrored(mirror, next(iterator))
            Path(value).parent.mkdir(parents=True, exist_ok=True)
            args += [arg, value]
        elif arg.startswith("--replay="):
            args.append("--replay=" + mirrored(mirror, arg.split("=", 1)[1]))
        elif arg.startswith("--agent-port="):
            # The container chose a port free in its own namespace; choose one free
            # here and tell the container which local port to forward to.
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                local_port = reservation.getsockname()[1]
            print(f"THUNDERHILL_REMOTE_PORT {local_port}", flush=True)
            args.append(f"--agent-port={local_port}")
        else:
            args.append(arg)
    env = dict(os.environ)
    for key, value in payload["env"].items():
        if not (key.startswith("THUNDERHILL_") or key == "XDG_DATA_HOME"):
            raise ValueError(f"Unexpected environment variable {key}")
        env[key] = value
    if "XDG_DATA_HOME" in payload["env"]:
        env["XDG_DATA_HOME"] = mirrored(mirror, payload["env"]["XDG_DATA_HOME"])
        Path(env["XDG_DATA_HOME"]).mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", godot, *args],
        env=env, stdin=subprocess.DEVNULL, start_new_session=True,
    )

    def watch_stdin():
        # The container closes the SSH session's stdin to stop the game. Raw reads:
        # a buffered reader blocked here would abort interpreter shutdown.
        while os.read(0, 65536):
            pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    threading.Thread(target=watch_stdin, daemon=True).start()
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--worktrees", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--container-repo", default="/opt/thunderhill")
    options = parser.parse_args()
    command = shlex.split(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
    if len(command) == 2 and command[0] == "godot-run":
        payload = json.loads(base64.b64decode(command[1]))
        return run_godot(payload, options.mirror, options.worktrees, options.godot,
                         options.container_repo)
    if command[:2] == ["rsync", "--server"]:
        # rsync's server form ends with ". <path>"; confine that path to the mirror.
        if len(command) < 4 or command[-2] != ".":
            raise ValueError("Unexpected rsync server command")
        target = mirrored(options.mirror, command[-1].rstrip("/")) + (
            "/" if command[-1].endswith("/") else "")
        Path(target.rstrip("/")).parent.mkdir(parents=True, exist_ok=True)
        os.execvp("rsync", command[:-1] + [target])
    print(f"Rejected remote command: {command[:1]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip finalization: the stdin watcher thread may still be blocked in read().
    os._exit(code)
