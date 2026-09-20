# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exercise the real headless game transport and simulation lifecycle.

Run: uv run tools/check_agent.py --godot /path/to/godot
The process and recordings are isolated in a temporary user data directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.stream = connection.makefile("rwb")

    def request(self, command: dict) -> dict:
        self.stream.write(json.dumps(command, allow_nan=False).encode() + b"\n")
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise RuntimeError("Game disconnected before replying")
        response = json.loads(line)
        assert isinstance(response, dict), response
        return response


def exercise(client: Client) -> dict:
    episode = client.request({"op": "reset", "policy_id": "checkpoint-alpha"})
    assert episode["tick"] == 0 and episode["policy_id"] == "checkpoint-alpha", episode
    episode_id = episode["episode_id"]

    def observe() -> dict:
        return client.request({"op": "observe", "episode_id": episode_id})

    time.sleep(0.15)
    assert observe() == episode, "Wall time advanced an idle agent episode"
    command = {
        "op": "advance", "episode_id": episode_id, "expected_tick": 0,
        "action_id": "first", "controls": {"throttle": 0.6},
    }
    advanced = client.request(command)
    assert advanced["tick"] == 12 and len(advanced["transitions"]) == 12, advanced
    assert abs(advanced["sim_time"] - 0.1) < 1e-10, advanced
    for index, transition in enumerate(advanced["transitions"]):
        assert transition["previous_tick"] == index and transition["tick"] == index + 1
    assert client.request(command) == advanced, "Duplicate action was not identical"
    assert observe()["tick"] == 12, "Duplicate action advanced twice"
    altered = copy.deepcopy(command)
    altered["controls"]["throttle"] = 0.2
    assert "error" in client.request(altered), "Reused action identifier accepted new payload"

    baseline = observe()
    for bad_controls in [{"throttle": -1}, {"steer": 2}, {"shift": 0.5}, {"rear_brake": "all"}, {"unknown": 1}]:
        invalid = {**command, "expected_tick": 12, "action_id": "invalid", "controls": bad_controls}
        assert "error" in client.request(invalid), bad_controls
        assert observe() == baseline, "Invalid controls mutated simulation"
    assert "error" in client.request({**command, "action_id": "stale-tick"})
    assert "error" in client.request({**command, "action_id": "stale-episode", "episode_id": "unrelated"})
    assert observe() == baseline, "Stale request mutated simulation"

    checkpoint_beta = client.request({"op": "reset", "policy_id": "checkpoint-beta"})
    assert checkpoint_beta["episode_id"] != episode_id
    assert checkpoint_beta["policy_id"] == "checkpoint-beta"
    assert "error" in client.request(command), "Old episode was accepted after reset"
    episode_id = checkpoint_beta["episode_id"]
    repeated = client.request({**command, "episode_id": episode_id})
    assert repeated["state"] == advanced["state"], "Reset and same actions changed physical state"
    assert repeated["track"] == advanced["track"], "Reset and same actions changed track state"
    assert all(t["policy_id"] == "checkpoint-beta" for t in repeated["transitions"])
    episode_id = client.request({"op": "reset", "policy_id": "checkpoint-coast"})["episode_id"]
    coast = client.request({**command, "episode_id": episode_id, "controls": {"throttle": 0.0}})
    assert coast["state"] != repeated["state"], "Different controls did not affect physical state"
    return {"ok": True, "checks": ["observation_freeze", "12_tick_advance", "duplicate_idempotency",
            "conflicting_duplicate", "invalid_controls", "stale_tick", "stale_episode",
            "deterministic_reset", "checkpoint_attribution", "action_changes_state"],
            "throttle_speed_m_s": repeated["state"]["speed"], "coast_speed_m_s": coast["state"]["speed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("godot4"))
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if not args.godot:
        parser.error("Provide --godot or install Godot on PATH")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="thunderhill-agent-check-") as directory:
        logfile = Path(directory) / "godot.log"
        env = dict(os.environ, XDG_DATA_HOME=str(Path(directory) / "data"))
        with logfile.open("wb") as output:
            process = subprocess.Popen([args.godot, "--headless", "--path", str(ROOT / "godot"),
                                        "--", f"--agent-port={port}"], stdout=output,
                                       stderr=subprocess.STDOUT, env=env)
            connection = None
            try:
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Game exited {process.returncode}\n{logfile.read_text()}")
                    try:
                        connection = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                        break
                    except OSError:
                        time.sleep(0.1)
                if connection is None:
                    raise TimeoutError(f"Game did not start\n{logfile.read_text()}")
                connection.settimeout(15.0)
                result = exercise(Client(connection))
                recordings = list((Path(directory) / "data").rglob("*.jsonl"))
                policies = set()
                transition_count = 0
                replay_recording = None
                for recording in recordings:
                    for line in recording.read_text().splitlines():
                        record = json.loads(line)
                        if record.get("type") == "transition":
                            policies.add(record["policy_id"])
                            transition_count += 1
                            if record["policy_id"] == "checkpoint-alpha":
                                replay_recording = recording
                assert policies == {"checkpoint-alpha", "checkpoint-beta", "checkpoint-coast"}, policies
                assert transition_count == 36, f"Unexpected recorded transitions, duplicate may have executed: {transition_count}"
                result["checks"].append("recorded_checkpoint_transitions")
                result["recorded_transitions"] = transition_count
                assert replay_recording is not None, "No recording available for playback check"
                playback = subprocess.run(
                    [args.godot, "--headless", "--path", str(ROOT / "godot"),
                     "--script", "tests/test_replay.gd", "--", str(replay_recording)],
                    capture_output=True, text=True, timeout=30, env=env,
                )
                assert playback.returncode == 0, playback.stdout + playback.stderr
                assert "ERROR:" not in playback.stderr, playback.stderr
                replay_reports = [line.removeprefix("REPLAY_CHECK ") for line in playback.stdout.splitlines()
                                  if line.startswith("REPLAY_CHECK ")]
                assert len(replay_reports) == 1, playback.stdout + playback.stderr
                replay_result = json.loads(replay_reports[0])
                assert replay_result == {"ok": True, "transitions": 12, "failures": 0}, replay_result
                result["checks"].append("authoritative_state_replay")
                result["replayed_transitions"] = replay_result["transitions"]
                print(json.dumps(result, indent=2))
            except Exception:
                print(logfile.read_text())
                raise
            finally:
                if connection:
                    connection.close()
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
