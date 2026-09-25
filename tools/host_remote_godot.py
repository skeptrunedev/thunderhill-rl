#!/usr/bin/env python3
"""Serve Godot to a Modal run from this machine over a reverse SSH tunnel.

Waits for the run to print its ``REMOTE_GODOT`` endpoint, prepares a worktree at
the run's source revision (imported like the Modal image), authorizes the
container's key for tools/remote_godot_host.py only, then keeps
``ssh -R 2222 -> local sshd`` open until the app stops, reconnecting on drops.

Run: python3 tools/host_remote_godot.py --app-id ap-...
(training/modal_native_validation.py --remote-godot starts this automatically.)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home() / "git_projects/thunderhill-remote"
KEY = Path.home() / ".ssh/thunderhill_modal"
AUTHORIZED_KEYS = Path.home() / ".ssh/authorized_keys"
GODOT = Path.home() / ".local/share/thunderhill-tools/godot-4.7.2/Godot_v4.7.2-stable_linux.x86_64"
MODAL = ["uv", "run", "-q", "--no-project", "--python", "3.12", "--with", "modal==1.5.4", "modal"]
TUNNEL_PORT = 2222


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S ") + message, flush=True)


def app_running(app_id: str) -> bool:
    listing = subprocess.run(MODAL + ["app", "list", "--json"], capture_output=True, text=True,
                             cwd=ROOT)
    if listing.returncode != 0:
        return True  # Transient CLI failure: do not tear the tunnel down.
    for app in json.loads(listing.stdout or "[]"):
        if app.get("app_id") == app_id:
            return app.get("stopped_at") is None and "stopped" not in app.get("state", "")
    return False


def endpoint(app_id: str, timeout: float) -> dict:
    """Follow the app's log stream until it prints its endpoint."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stream = subprocess.Popen(MODAL + ["app", "logs", app_id], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
        try:
            for line in stream.stdout:
                if line.startswith("REMOTE_GODOT "):
                    return json.loads(line.split(" ", 1)[1])
                if time.monotonic() >= deadline:
                    break
        finally:
            stream.kill()
            stream.wait()
        if not app_running(app_id):
            raise RuntimeError(f"{app_id} stopped before publishing its SSH endpoint")
        time.sleep(5)
    raise TimeoutError(f"{app_id} did not publish its SSH endpoint")


def prepare_worktree(revision: str) -> Path:
    worktree = STATE / "worktrees" / revision
    if (worktree / ".imported").exists():
        return worktree
    if worktree.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(worktree)])
    subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin"], check=True)
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach", str(worktree),
                    revision], check=True)
    # The Modal image copies the working tree's build info and imports the project.
    build_info = ROOT / "godot/data/build-info.json"
    if build_info.exists():
        shutil.copy2(build_info, worktree / "godot/data/build-info.json")
    subprocess.run([str(GODOT), "--headless", "--path", str(worktree / "godot"), "--editor",
                    "--import"], check=True, timeout=1800,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (worktree / ".imported").touch()
    return worktree


def authorize(container_key: str, marker: str, worktree: Path) -> None:
    command = (f"python3 {worktree / 'tools/remote_godot_host.py'} --mirror {STATE / 'mirror'} "
               f"--worktrees {STATE / 'worktrees'} --godot {GODOT}")
    options = (f'restrict,port-forwarding,permitopen="127.0.0.1:*",from="127.0.0.1,::1",'
               f'command="{command}"')
    lines = AUTHORIZED_KEYS.read_text().splitlines() if AUTHORIZED_KEYS.exists() else []
    lines = [line for line in lines if not line.endswith(" " + marker)]
    lines.append(f"{options} {container_key.split()[0]} {container_key.split()[1]} {marker}")
    AUTHORIZED_KEYS.write_text("\n".join(lines) + "\n")


def revoke(marker: str) -> None:
    if AUTHORIZED_KEYS.exists():
        kept = [line for line in AUTHORIZED_KEYS.read_text().splitlines()
                if not line.endswith(" " + marker)]
        AUTHORIZED_KEYS.write_text("\n".join(kept) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--endpoint-timeout", type=float, default=1800)
    args = parser.parse_args()
    (STATE / "mirror").mkdir(parents=True, exist_ok=True)
    log(f"waiting for {args.app_id} to publish its SSH endpoint")
    remote = endpoint(args.app_id, args.endpoint_timeout)
    log(f"endpoint {remote['host']}:{remote['port']} revision {remote['revision']}")
    worktree = prepare_worktree(remote["revision"])
    marker = "thunderhill-remote-godot:" + args.app_id
    authorize(remote["container_key"], marker, worktree)
    try:
        while app_running(args.app_id):
            log("opening reverse tunnel")
            subprocess.run([
                "ssh", "-N", "-i", str(KEY), "-p", str(remote["port"]),
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                "-o", "ExitOnForwardFailure=yes",
                "-R", f"127.0.0.1:{TUNNEL_PORT}:127.0.0.1:22",
                f"root@{remote['host']}",
            ])
            time.sleep(5)
        log(f"{args.app_id} stopped")
    finally:
        revoke(marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
