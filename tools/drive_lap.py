# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Privileged path following QA driver. This is not an LLM or training baseline."""
from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from check_agent import Client

ROOT = Path(__file__).resolve().parents[1]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def audit_recordings(output: Path) -> dict:
    gates = []
    excursions = 0
    ticks = 0
    driven_episodes = 0
    ordered_ticks = True
    for path in (output / "userdata").rglob("*.jsonl"):
        previous_tick = 0
        episode_ticks = 0
        for line in path.open():
            row = json.loads(line)
            if row.get("type") != "transition":
                continue
            ordered_ticks = ordered_ticks and row["previous_tick"] == previous_tick and row["tick"] == previous_tick + 1
            previous_tick = row["tick"]
            episode_ticks += 1
            excursions += not row["track"]["on_track"]
            gates.extend(event["gate"] for event in row["events"] if event["type"] == "gate")
        ticks += episode_ticks
        driven_episodes += episode_ticks > 0
    return {"recorded_ticks": ticks, "recorded_offtrack_ticks": excursions,
            "ordered_ticks": ordered_ticks, "driven_episodes": driven_episodes, "recorded_gates": gates}


class Driver:
    def __init__(self, max_speed: float):
        geometry = json.loads((ROOT / "godot/data/track.json").read_text())
        self.samples = geometry["samples"]
        self.stations = [sample["s"] for sample in self.samples]
        self.length = geometry["length_m"]
        self.max_speed = max_speed

    def point(self, station: float) -> list[float]:
        station %= self.length
        index = max(0, bisect.bisect_right(self.stations, station) - 1)
        following = (index + 1) % len(self.samples)
        end = self.stations[following] if following else self.length
        fraction = (station - self.stations[index]) / (end - self.stations[index])
        return [a + fraction * (b - a) for a, b in zip(self.samples[index]["p"], self.samples[following]["p"])]

    def controls(self, observation: dict) -> tuple[dict, dict]:
        state = observation["state"]
        speed = state["speed"]
        station = observation["track"]["progress"] * self.length
        lookahead = 7.0 + speed * 1.3
        target = self.point(station + lookahead)
        dx, dz = target[0] - state["position"][0], target[2] - state["position"][2]
        target_heading = math.atan2(dx, -dz)
        alpha = (target_heading - state["heading"] + math.pi) % (2 * math.pi) - math.pi
        curvature = 2 * math.sin(alpha) / max(math.hypot(dx, dz), 1.0)
        requested_lean = math.atan(speed * speed * curvature / 9.81)
        steering = clamp(requested_lean / 0.88, -0.75, 0.75)
        # Preview braking uses privileged centerline curvature, never model observations.
        target_speed = self.max_speed
        for ahead in range(0, 121, 6):
            index = max(0, bisect.bisect_right(self.stations, (station + ahead) % self.length) - 1)
            curve = abs(self.samples[index]["curvature"])
            corner_speed = math.sqrt(2.2 / max(curve, 0.001))
            approach_speed = math.sqrt(corner_speed * corner_speed + 2 * 1.8 * ahead)
            target_speed = min(target_speed, approach_speed)
        target_speed = max(4.5, target_speed)
        error = target_speed - speed
        throttle = clamp(0.10 + 0.14 * error, 0.0, 0.65)
        front_brake = clamp(-error * 0.15, 0.0, 0.65)
        if front_brake > 0:
            throttle = 0.0
        return {"throttle": throttle, "front_brake": front_brake, "rear_brake": front_brake * 0.18,
                "steer": steering, "assist_enabled": True, "auto_shift": True}, {
                "target_speed": target_speed, "lookahead": lookahead, "curvature": curvature,
                "target_lean": requested_lean,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("godot4"))
    parser.add_argument("--max-speed", type=float, default=18.0)
    parser.add_argument("--max-actions", type=int, default=8000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.godot:
        parser.error("Provide --godot")
    output = args.output or ROOT / "artifacts/qa" / datetime.now(timezone.utc).strftime("lap-%Y%m%dT%H%M%SZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ, XDG_DATA_HOME=str(output / "userdata"))
    start = time.monotonic()
    with (output / "godot.log").open("wb") as log:
        process = subprocess.Popen([args.godot, "--headless", "--path", str(ROOT / "godot"),
                                    "--", f"--agent-port={port}"], stdout=log, stderr=subprocess.STDOUT, env=env)
        connection = None
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError((output / "godot.log").read_text())
                try:
                    connection = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                    break
                except OSError:
                    time.sleep(0.1)
            if connection is None:
                raise TimeoutError("Game did not start")
            connection.settimeout(30)
            client = Client(connection)
            observation = client.request({"op": "reset", "policy_id": "privileged-path-qa-v1"})
            driver = Driver(args.max_speed)
            max_lateral = 0.0
            actions = 0
            reason = "action_budget"
            with (output / "driver.jsonl").open("w") as trace:
                for action_index in range(args.max_actions):
                    controls, diagnostics = driver.controls(observation)
                    request = {"op": "advance", "episode_id": observation["episode_id"],
                               "expected_tick": observation["tick"], "action_id": str(action_index), "controls": controls}
                    response = client.request(request)
                    if "error" in response:
                        raise RuntimeError(response)
                    observation = {key: value for key, value in response.items() if key != "transitions"}
                    trace.write(json.dumps({"request": request, "driver": diagnostics, "observation": observation}) + "\n")
                    actions += 1
                    max_lateral = max(max_lateral, abs(observation["track"]["lateral_m"]))
                    if action_index % 100 == 0:
                        print(json.dumps({"actions": actions, "progress": observation["track"]["progress"],
                                          "speed": observation["state"]["speed"], "lateral": observation["track"]["lateral_m"]}), flush=True)
                    if observation["terminated"]:
                        reason = "completed" if observation["track"]["completed_laps"] else "crashed"
                        break
                    if not observation["track"]["lap_valid"]:
                        reason = "track_limits"
                        break
            success = observation["track"]["completed_laps"] == 1 and observation["track"]["lap_valid"] and not observation["state"]["crashed"]
            audit = audit_recordings(output)
            record_valid = (audit["recorded_offtrack_ticks"] == 0 and audit["ordered_ticks"]
                            and audit["driven_episodes"] == 1 and audit["recorded_gates"] == [*range(1, 32), 0])
            if success and not record_valid:
                success = False
                reason = "recording_verification_failed"
            summary = {"success": success, "reason": reason, "driver": "privileged-path-qa-v1",
                       "not_training": True, "actions": actions, "sim_time_s": observation["sim_time"],
                       "wall_time_s": time.monotonic() - start, "max_lateral_m": max_lateral,
                       "max_target_speed_m_s": args.max_speed, "recording_audit": audit,
                       "final_observation": observation}
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"output": str(output), **{k: v for k, v in summary.items() if k != "final_observation"}}, indent=2), flush=True)
            if not success:
                raise SystemExit(1)
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
