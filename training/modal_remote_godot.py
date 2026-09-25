"""Route a Modal container's Godot launches to a rendering host over SSH.

Inside a Modal function: start sshd, expose it with modal.forward, print the
``REMOTE_GODOT`` endpoint that tools/host_remote_godot.py follows, wait for the
host's reverse tunnel, then put tools/remote_godot.py first on PATH as
``godot``. Every existing caller (bridge, camera preflight, video renderer)
keeps invoking ``godot`` unchanged.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from training.modal_native import REMOTE

TUNNEL_PORT = 2222
SSHD_CONFIG = """Port 22
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
AllowTcpForwarding yes
GatewayPorts no
ClientAliveInterval 15
ClientAliveCountMax 4
PidFile /run/sshd-thunderhill.pid
"""


@contextmanager
def remote_godot(revision: str, host_user: str, host_public_key: str,
                 wait_seconds: float = 1800):
    import modal

    subprocess.run(
        "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "openssh-server openssh-client rsync",
        shell=True, check=True, stdout=subprocess.DEVNULL,
    )
    ssh = Path("/root/.ssh")
    ssh.mkdir(mode=0o700, exist_ok=True)
    (ssh / "authorized_keys").write_text(host_public_key.strip() + "\n")
    (ssh / "authorized_keys").chmod(0o600)
    key = ssh / "id_ed25519"
    if not key.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
                       check=True)
    subprocess.run(["ssh-keygen", "-A"], check=True, stdout=subprocess.DEVNULL)
    Path("/run/sshd").mkdir(exist_ok=True)
    config = Path("/etc/ssh/sshd_thunderhill")
    config.write_text(SSHD_CONFIG)
    sshd = subprocess.Popen(["/usr/sbin/sshd", "-D", "-e", "-f", str(config)])
    try:
        with modal.forward(22, unencrypted=True) as tunnel:
            host, port = tunnel.tcp_socket
            print("REMOTE_GODOT " + json.dumps(dict(
                host=host, port=port, revision=revision,
                container_key=(ssh / "id_ed25519.pub").read_text().strip(),
            )), flush=True)
            deadline = time.monotonic() + wait_seconds
            while True:
                try:
                    socket.create_connection(("127.0.0.1", TUNNEL_PORT), timeout=2).close()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("The rendering host never opened its reverse tunnel")
                    time.sleep(2)
            shim = Path("/opt/remote-godot-bin")
            shim.mkdir(exist_ok=True)
            (shim / "godot").write_text(
                f'#!/bin/sh\nexec python3 {REMOTE}/tools/remote_godot.py "$@"\n')
            (shim / "godot").chmod(0o755)
            os.environ.update(
                PATH=f"{shim}:{os.environ['PATH']}",
                THUNDERHILL_REMOTE_GODOT="1",
                THUNDERHILL_REMOTE_GODOT_KEY=str(key),
                THUNDERHILL_REMOTE_GODOT_USER=host_user,
                THUNDERHILL_REMOTE_GODOT_REVISION=revision,
            )
            # The first request also waits for the host to finish importing the project.
            subprocess.run(["godot", "--headless", "--version"], check=True, timeout=1800)
            print(f"Remote Godot ready through {host}:{port}", flush=True)
            yield
    finally:
        sshd.terminate()


def start_host(app_id: str) -> dict:
    """Local side: serve this app's Godot from this machine; return its remote_host."""
    import getpass

    from training.modal_native import ROOT

    key = Path.home() / ".ssh/thunderhill_modal.pub"
    if not key.exists():
        raise FileNotFoundError(f"Create the tunnel key first: ssh-keygen -t ed25519 -f {key.with_suffix('')}")
    # A user unit outlives this launcher, like the detached Modal app it serves.
    subprocess.run([
        "systemd-run", "--user", "--collect", "--unit", f"thunderhill-remote-godot-{app_id}",
        f"--setenv=PATH={os.environ['PATH']}", f"--working-directory={ROOT}",
        "/usr/bin/python3", str(ROOT / "tools/host_remote_godot.py"), "--app-id", app_id,
    ], check=True)
    print(f"Serving Godot from this machine: journalctl --user -u thunderhill-remote-godot-{app_id} -f")
    return dict(host_user=getpass.getuser(), host_public_key=key.read_text().strip())
