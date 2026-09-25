"""Run Godot on a remote rendering host over the reverse SSH tunnel.

Installed as ``godot`` on PATH inside a Modal container by
training/modal_remote_godot.py. Every invocation executes the same Godot
command on the host (tools/remote_godot_host.py maps container paths into its
mirror directory), streams its output back, forwards the agent port once the
game reports ready, and copies files the container reads back afterwards.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

TUNNEL_PORT = int(os.environ.get("THUNDERHILL_REMOTE_GODOT_PORT", "2222"))


def _ssh(*extra: str) -> list[str]:
    return [
        "ssh", "-p", str(TUNNEL_PORT), "-i", os.environ["THUNDERHILL_REMOTE_GODOT_KEY"],
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=15", *extra,
        os.environ["THUNDERHILL_REMOTE_GODOT_USER"] + "@127.0.0.1",
    ]


def _rsync(source: str, destination: str) -> None:
    subprocess.run(
        ["rsync", "-a", "--mkpath", "-e", " ".join(_ssh()[:-1]), source, destination],
        check=True, timeout=600,
    )


def _remote(real: str) -> str:
    return os.environ["THUNDERHILL_REMOTE_GODOT_USER"] + "@127.0.0.1:" + real


def pull_dir(path: Path) -> None:
    """Copy the host's copy of a container directory back into the container."""
    real = os.path.realpath(path)
    _rsync(_remote(real) + "/", real + "/")


def pull_file(path: Path) -> None:
    real = os.path.realpath(path)
    _rsync(_remote(real), real)


def sync_if_remote(path: Path) -> None:
    """Before reading Godot's data directory: pull it when Godot runs remotely."""
    if os.environ.get("THUNDERHILL_REMOTE_GODOT") == "1":
        pull_dir(path)


def push(path: Path) -> None:
    real = os.path.realpath(path)
    _rsync(real, _remote(real))


def _argument_paths(args: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Resolve path arguments; return (args, inputs to push, outputs to pull)."""
    resolved, inputs, outputs = [], [], []
    iterator = iter(args)
    for arg in iterator:
        if arg in ("--path", "--write-movie"):
            value = os.path.realpath(next(iterator))
            resolved += [arg, value]
            if arg == "--write-movie":
                outputs.append(value)
        elif arg.startswith("--replay="):
            value = os.path.realpath(arg.split("=", 1)[1])
            resolved.append("--replay=" + value)
            inputs.append(value)
        else:
            resolved.append(arg)
    return resolved, inputs, outputs


def main(argv: list[str]) -> int:
    args, inputs, outputs = _argument_paths(argv)
    env = {k: v for k, v in os.environ.items()
           if k.startswith("THUNDERHILL_") and not k.startswith("THUNDERHILL_REMOTE_GODOT")}
    data = os.environ.get("XDG_DATA_HOME")
    if data:
        env["XDG_DATA_HOME"] = os.path.realpath(data)
    for path in inputs:
        push(Path(path))
    payload = base64.b64encode(json.dumps(dict(
        args=args, env=env, revision=os.environ["THUNDERHILL_REMOTE_GODOT_REVISION"],
    )).encode()).decode()
    process = subprocess.Popen(_ssh() + ["godot-run " + payload], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = next((a.split("=", 1)[1] for a in args if a.startswith("--agent-port=")), None)
    forward = None
    terminated = threading.Event()

    def stop():
        # Closing stdin makes the host terminate its Godot process.
        if process.stdin and not process.stdin.closed:
            process.stdin.close()

    def terminate(*_):
        terminated.set()
        stop()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        for line in process.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            if port and forward is None and line.startswith(b"THUNDERHILL_READY"):
                # Forward only once the game listens; an early forward would accept
                # and immediately drop the caller's connection attempts.
                forward = subprocess.Popen(
                    _ssh("-N", "-o", "ExitOnForwardFailure=yes",
                         "-L", f"127.0.0.1:{port}:127.0.0.1:{port}"))
        returncode = process.wait()
    finally:
        stop()
        if forward is not None:
            forward.terminate()
            forward.wait()
    # A terminated caller (the bridge's worker allows five seconds) has already
    # pulled what it reads; only natural exits copy outputs back here.
    if not terminated.is_set():
        for path in outputs:
            pull_file(Path(path))
        if data:
            pull_dir(Path(data))
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
