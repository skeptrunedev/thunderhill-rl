# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "gymnasium>=1.1"]
# ///
"""Ride Thunderhill with a Kev decision model: the SAC evaluation, driven by typed questions.

Every 0.1 s control step renders the observation as kev_dataset's telemetry state and
asks a Kev server (/v1/systemone) the three training questions, word for word. The
action is the probability-weighted steer and pedal level (kev_dataset.expected_level),
or the most likely level with --decode argmax. The simulator waits for each action,
so model latency slows the evaluation but never changes the ride.

Rides start from the same fixed rolling starts as the SAC evaluator
(sac_env.evaluation_starts), with Godot recordings kept, and write:

  eval.jsonl       one row per ride, the sac_async eval row format (lap time, termination,
                   recording path), so the apex and video tools read it unchanged
  decisions.jsonl  every step: station, action, the answers' probabilities, the
                   off_track_soon probability and the model latency

  uv run --extra serve python -m kev.serve --run CHECKPOINT --port 8019   # in the kev repo
  uv run training/kev_drive.py --godot GODOT --run-name kev-rider-v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev_dataset import PEDAL_LEVELS, QUESTIONS, STEER_LEVELS, expected_level, render_state  # noqa: E402
from sac_env import ROOT, ThunderhillSACEnv, evaluation_starts  # noqa: E402


class KevRider:
    def __init__(self, url: str, decode: str, pedal_gain: float):
        self.url, self.decode, self.pedal_gain = url.rstrip("/") + "/v1/systemone", decode, pedal_gain

    def ask(self, obs) -> dict:
        body = json.dumps({"model": "kev-latest", "state": render_state(obs), "questions": QUESTIONS}).encode()
        request = urllib.request.Request(self.url, body, {"content-type": "application/json"})
        began = time.perf_counter()
        with urllib.request.urlopen(request, timeout=60) as response:
            answer = json.load(response)
        answer["client_ms"] = (time.perf_counter() - began) * 1000.0
        return answer

    def level(self, probabilities: dict, levels: dict) -> float:
        if self.decode == "argmax":
            return levels[max(probabilities, key=probabilities.get)]
        return expected_level(probabilities, levels)


def ride(env: ThunderhillSACEnv, rider: KevRider, start: dict, log) -> dict:
    obs, _ = env.reset(options={"start": start})
    while True:
        response = rider.ask(obs)
        answers = response["answers"]
        steer = rider.level(answers["steer"]["probabilities"], STEER_LEVELS)
        pedal = rider.level(answers["pedal"]["probabilities"], PEDAL_LEVELS)
        # sac_env.controls scales the pedal by pedal_gain; send the raw value that lands on `pedal`.
        action = np.array([steer, pedal / rider.pedal_gain], dtype=np.float32)
        next_obs, _, terminated, truncated, info = env.step(action)
        log({"episode_id": info["episode_id"], "step": env._step_count,
             "station_m": round(float(env._observation["track"]["progress"]) * env.track.length, 1),
             "steer": round(steer, 4), "pedal": round(pedal, 4),
             "steer_probabilities": answers["steer"]["probabilities"],
             "pedal_probabilities": answers["pedal"]["probabilities"],
             "off_track_soon": answers["off_track_soon"]["noul"],
             "model_ms": response.get("latency_ms"), "client_ms": round(response["client_ms"], 1)})
        obs = next_obs
        if terminated or truncated:
            return info["episode_summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--godot", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/kev")
    parser.add_argument("--kev-url", default="http://127.0.0.1:8019")
    parser.add_argument("--decode", choices=("mean", "argmax"), default="mean")
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--horizon", type=float, default=240.0)
    parser.add_argument("--pedal-gain", type=float, default=1.25)
    args = parser.parse_args()
    run_dir = args.runs_dir / args.run_name
    directory = run_dir / "eval"
    directory.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1) + "\n")
    rider = KevRider(args.kev_url, args.decode, args.pedal_gain)
    starts = evaluation_starts(args.starts)
    envs = [ThunderhillSACEnv(godot=args.godot, data_dir=directory, horizon_s=args.horizon,
                              reward_line="progress", pedal_gain=args.pedal_gain, record_godot=True,
                              policy_id=f"{args.run_name}-eval") for _ in starts]
    decisions = (run_dir / "decisions.jsonl").open("a")

    def log(row):
        decisions.write(json.dumps(row) + "\n")

    began = time.monotonic()
    try:
        with ThreadPoolExecutor(len(envs)) as pool:
            rows = list(pool.map(lambda pair: ride(pair[0], rider, pair[1], log), zip(envs, starts)))
    finally:
        decisions.close()
        for env in envs:
            env.close()
    with (directory / "eval.jsonl").open("a") as stream:
        for row in rows:
            recordings = list(directory.glob(f"**/{row['episode_id']}.jsonl"))
            row.update(policy_step=0, checkpoint=args.kev_url,
                       recording=str(recordings[0].relative_to(run_dir)) if recordings else None)
            stream.write(json.dumps(row) + "\n")
    laps = sorted(row["lap_time_s"] for row in rows if row["termination"] == "lap_completed")
    print(f"KEV EVAL {args.run_name}: {len(laps)}/{len(rows)} laps {laps} s, "
          f"terminations {[row['termination'] for row in rows]}, {time.monotonic() - began:.0f} s")


if __name__ == "__main__":
    main()
