# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Verify two real telemetry workers have independent state and recordings.

This is a short process isolation check, not a throughput or camera benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

from check_agent import ROOT, Client


@contextmanager
def worker(godot, directory, timeout):
    directory.mkdir()
    data = directory / "data"
    log = directory / "godot.log"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    connection = None
    client = None
    with log.open("wb") as output:
        process = subprocess.Popen(
            [
                godot,
                "--headless",
                "--path",
                str(ROOT / "godot"),
                "--",
                f"--agent-port={port}",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, XDG_DATA_HOME=str(data)),
        )
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Worker exited {process.returncode}: {log.read_text()}"
                    )
                try:
                    connection = socket.create_connection(
                        ("127.0.0.1", port), timeout=0.5
                    )
                    break
                except OSError:
                    time.sleep(0.1)
            if connection is None:
                raise TimeoutError(f"Worker did not start: {log.read_text()}")
            connection.settimeout(15)
            client = Client(connection)
            yield client, data
        finally:
            if client:
                client.stream.close()
            if connection:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def advance(client, observation, throttle):
    return client.request(
        {
            "op": "advance",
            "episode_id": observation["episode_id"],
            "expected_tick": observation["tick"],
            "action_id": "same-action-id",
            "controls": {"throttle": throttle},
        }
    )


def observe(client, episode):
    return client.request({"op": "observe", "episode_id": episode["episode_id"]})


def verify(workers):
    (a, data_a), (b, data_b) = workers
    initial_a = a.request({"op": "reset", "policy_id": "isolation-a"})
    initial_b = b.request({"op": "reset", "policy_id": "isolation-b"})
    assert initial_a["episode_id"] != initial_b["episode_id"], (
        "Cross worker episode collision"
    )
    assert initial_a["state"] == initial_b["state"], (
        "Workers disagree on initial physics"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(advance, a, initial_a, 0.6)
        future_b = pool.submit(advance, b, initial_b, 0.0)
        result_a, result_b = future_a.result(), future_b.result()
    assert result_a["tick"] == result_b["tick"] == 12
    assert result_a["state"] != result_b["state"], (
        "Independent actions had identical states"
    )
    for result, policy in ((result_a, "isolation-a"), (result_b, "isolation-b")):
        assert len(result["transitions"]) == 12
        assert all(row["policy_id"] == policy for row in result["transitions"])
    before_b = observe(b, initial_b)
    reset_a = a.request({"op": "reset", "policy_id": "isolation-a-reset"})
    assert observe(b, initial_b) == before_b, "Reset A changed B"
    advance(a, reset_a, 0.6)
    assert observe(b, initial_b) == before_b, "Advance A changed B"
    before_a = observe(a, reset_a)
    assert "error" in b.request({"op": "observe", "episode_id": reset_a["episode_id"]})
    b.request({"op": "reset", "policy_id": "isolation-b-reset"})
    assert observe(a, reset_a) == before_a, "Reset B changed A"
    # Reset flushes the preceding episode before inspecting real files.
    a.request({"op": "reset", "policy_id": "isolation-a-finished"})
    counts = []
    for data, expected in (
        (data_a, {"isolation-a", "isolation-a-reset"}),
        (data_b, {"isolation-b"}),
    ):
        rows = [
            json.loads(line)
            for path in data.rglob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        transitions = [row for row in rows if row.get("type") == "transition"]
        assert {row["policy_id"] for row in transitions} == expected
        assert len(transitions) == 12 * len(expected), (
            "Cross worker or duplicate recording"
        )
        counts.append(len(transitions))
    return {
        "ok": True,
        "workers": 2,
        "recorded_transitions": counts,
        "checks": [
            "unique_episode_ids",
            "equal_reset_state",
            "simultaneous_different_actions",
            "shared_action_id_isolation",
            "bidirectional_reset_isolation",
            "advance_isolation",
            "foreign_episode_rejection",
            "recording_isolation",
        ],
        "scope": "Short headless telemetry check with separate process user data directories",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--godot", default=shutil.which("godot") or shutil.which("godot4")
    )
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if not args.godot:
        parser.error("Provide --godot or install Godot on PATH")
    with (
        tempfile.TemporaryDirectory(prefix="thunderhill-parallel-") as directory,
        ExitStack() as stack,
    ):
        workers = [
            stack.enter_context(
                worker(args.godot, Path(directory) / str(i), args.timeout)
            )
            for i in range(2)
        ]
        print(json.dumps(verify(workers), indent=2))


if __name__ == "__main__":
    main()
