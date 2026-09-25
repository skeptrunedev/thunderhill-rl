# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Qualify rolling-start sensor warmups in the real headless Godot simulator.

For each start scene this resets the game at the scene's station and speed,
runs the same 1.5 s warmup as training/alpagym_bridge.py (WarmupSpeedHold,
twelve physics ticks per 0.1 s control), asks the game to hand off model
control, and reports the per-tick measured speed band and track state. It is
a simulator initial-condition check: no model, reward or training data.

Run: python3 tools/check_rolling_start.py --godot /path/to/godot --max-speed 30 \
         --randomized-starts 8 --seed 0 [--coast]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]

from agent_harness import ThunderhillEnv  # noqa: E402
from check_parallel import worker  # noqa: E402
from lap_policy import RoadTelemetry  # noqa: E402
from training.episode_config import (  # noqa: E402
    WARMUP_SECONDS,
    WarmupSpeedHold,
    parse_scene_id,
    randomized_starts,
    scene_id,
)


class _Trace:
    def write(self, _):
        pass

    def flush(self):
        pass


def warmup(client, road, scene, *, coast):
    start = parse_scene_id(scene)
    env = ThunderhillEnv(client, 0, _Trace(), lambda: 0, road_telemetry=road)
    env.reset(
        initial_speed_m_s=start["initial_speed_m_s"], station=start["start_station_m"]
    )
    view = json.loads(env.observe())
    hold = WarmupSpeedHold(start["initial_speed_m_s"])
    first = env._observation["state"]
    ticks = [dict(speed=first["speed"], lateral=env._observation["track"]["lateral_m"],
                  gear=first["gear"], rpm=first["rpm"], accel=0.0, on_track=True, lap_valid=True)]
    for _ in range(round(WARMUP_SECONDS * 10)):
        if coast:  # the previous neutral_coasting_sensor_warmup, for comparison
            controls = dict(throttle=0.0, steer=0.0, front_brake=0.0, rear_brake=0.0)
        else:
            controls, _ = hold.controls(env._observation["state"]["speed"])
        view = json.loads(env.control_bike(view["observation_token"], **controls))
        for transition in env._observation["transitions"]:
            state, track = transition["state"], transition["track"]
            ticks.append(dict(
                speed=state["speed"], lateral=track["lateral_m"], gear=state["gear"],
                rpm=state["rpm"], accel=state["longitudinal_acceleration"],
                on_track=track["on_track"], lap_valid=track["lap_valid"],
            ))
        if view["done"]:
            break
    handoff = client.request(dict(
        op="begin_model_control", episode_id=env._observation["episode_id"],
        expected_tick=env._observation["tick"],
    ))
    target = start["initial_speed_m_s"]
    speeds = [t["speed"] for t in ticks]
    heading_error = None
    if "error" not in handoff:
        station = handoff["model_control_start"]["station_m"]
        a, b = road.point(station), road.point(station + 1.0)
        track_heading = math.atan2(b[0] - a[0], -(b[2] - a[2]))
        heading_error = (handoff["state"]["heading"] - track_heading + math.pi) % (
            2 * math.pi
        ) - math.pi
    return dict(
        scene_id=scene,
        start_station_m=start["start_station_m"],
        target_m_s=target,
        physics_ticks=len(ticks) - 1,
        speed_min=round(min(speeds), 3),
        speed_max=round(max(speeds), 3),
        max_abs_error_m_s=round(max(abs(s - target) for s in speeds), 3),
        final_speed=round(speeds[-1], 3),
        mean_accel_m_s2=round((speeds[-1] - speeds[0]) / WARMUP_SECONDS, 3),
        gears=sorted({t["gear"] for t in ticks}),
        max_abs_lateral_m=round(max(abs(t["lateral"]) for t in ticks), 3),
        always_on_track=all(t["on_track"] for t in ticks),
        lap_valid=all(t["lap_valid"] for t in ticks),
        handoff_accepted="error" not in handoff,
        handoff_error=handoff.get("error"),
        handoff_heading_error_rad=None if heading_error is None else round(heading_error, 4),
        speed_trace_0p1s=[round(s, 3) for s in speeds[::12]],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--max-speed", type=float, default=30.0)
    parser.add_argument("--randomized-starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-line", action="store_true",
                        help="Also check the historical start-line scene at --max-speed")
    parser.add_argument("--coast", action="store_true",
                        help="Replay the previous zero-throttle coasting warmup instead")
    args = parser.parse_args()
    road = RoadTelemetry()
    scenes = [s["scene_id"] for s in randomized_starts(
        road, count=args.randomized_starts, seed=args.seed, max_speed=args.max_speed
    )] if args.randomized_starts else []
    if args.start_line:
        scenes.insert(0, scene_id(args.max_speed))
    results = []
    with tempfile.TemporaryDirectory() as directory:
        with worker(args.godot, Path(directory) / "godot", 90) as (client, _):
            for scene in scenes:
                results.append(warmup(client, road, scene, coast=args.coast))
                print(json.dumps(results[-1]), flush=True)
    worst = max(r["max_abs_error_m_s"] for r in results)
    print(json.dumps(dict(
        mode="coast" if args.coast else "speed_hold",
        scenes=len(results),
        worst_abs_speed_error_m_s=worst,
        all_on_track=all(r["always_on_track"] and r["lap_valid"] for r in results),
        all_handoffs_accepted=all(r["handoff_accepted"] for r in results),
        max_abs_lateral_m=max(r["max_abs_lateral_m"] for r in results),
    )))


if __name__ == "__main__":
    main()
