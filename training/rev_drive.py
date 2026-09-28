# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "gymnasium>=1.1", "torch==2.8.0"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Ride Thunderhill with Rev, our decision model: the SAC evaluation, driven by typed questions.

Every 0.1 s control step renders the observation as rev_dataset's telemetry state and
asks a Rev server (kev.serve) (/v1/systemone) the three training questions, word for word. The
action is the probability-weighted steer and pedal level (rev_dataset.expected_level),
or the most likely level with --decode argmax. The simulator waits for each action,
so model latency slows the evaluation but never changes the ride.

Rides start from the same fixed rolling starts as the SAC evaluator
(sac_env.evaluation_starts), with Godot recordings kept, and write:

  eval.jsonl       one row per ride, the sac_async eval row format (lap time, termination,
                   recording path), so the apex and video tools read it unchanged
  decisions.jsonl  every step: station, action, the answers' probabilities, the
                   off_track_soon probability and the model latency

With --collect N, rides instead gather DAgger states: --workers envs ride N episodes each
from random rolling starts (--horizon 60), the SAC --teacher taking the step with
probability --beta, and every visited observation is saved with its off-track outcome to
collect-*.npz for rev_dataset.py --dagger.

  uv run --extra serve python -m kev.serve --run runs/rev/rev-0.8b-v1/checkpoint --port 8019   # in the kev repo
  uv run training/rev_drive.py --godot GODOT --run-name rev-0.8b-v1-ride
  uv run training/rev_drive.py --godot GODOT --run-name collect-r1 --collect 3 --workers 32 \
      --horizon 60 --beta 0.5 --teacher runs/sac/sac-v7-lr1e4-1/checkpoints/step_24000071.pt
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
from rev_dataset import PEDAL_LEVELS, QUESTIONS, STEER_LEVELS, expected_level, render_state  # noqa: E402
from rev_dataset import OFF_TRACK_HORIZON_STEPS, OFF_TRACK_TERMINATIONS  # noqa: E402
from sac_env import ROOT, ThunderhillSACEnv, evaluation_starts  # noqa: E402


class RevRider:
    def __init__(self, url: str, decode: str, pedal_gain: float):
        self.url, self.decode, self.pedal_gain = url.rstrip("/") + "/v1/systemone", decode, pedal_gain

    def ask(self, obs) -> dict:
        # kev.serve answers to its fixed alias whatever checkpoint it loaded; the weights are Rev's.
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


def ride(env: ThunderhillSACEnv, rider: RevRider, start: dict | None, log, teacher=None, beta=0.0,
         rng=None):
    """One episode. With a teacher, it rides the step with probability beta (Rev is still
    asked, so its answers are logged). Returns (summary, visited observations)."""
    obs, _ = env.reset(options={"start": start} if start else None)
    visited = []
    while True:
        visited.append(obs)
        response = rider.ask(obs)
        answers = response["answers"]
        steer = rider.level(answers["steer"]["probabilities"], STEER_LEVELS)
        pedal = rider.level(answers["pedal"]["probabilities"], PEDAL_LEVELS)
        by_teacher = teacher is not None and rng.random() < beta
        if by_teacher:
            steer, pedal = (float(v) for v in teacher.controls(obs))
        # sac_env.controls scales the pedal by pedal_gain; send the raw value that lands on `pedal`.
        action = np.array([steer, pedal / rider.pedal_gain], dtype=np.float32)
        next_obs, _, terminated, truncated, info = env.step(action)
        log({"episode_id": info["episode_id"], "step": env._step_count,
             "station_m": round(float(env._observation["track"]["progress"]) * env.track.length, 1),
             "steer": round(steer, 4), "pedal": round(pedal, 4), "teacher": by_teacher,
             "steer_probabilities": answers["steer"]["probabilities"],
             "pedal_probabilities": answers["pedal"]["probabilities"],
             "off_track_soon": answers["off_track_soon"]["noul"],
             "model_ms": response.get("latency_ms"), "client_ms": round(response["client_ms"], 1)})
        obs = next_obs
        if terminated or truncated:
            return info["episode_summary"], np.stack(visited)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--godot", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/rev")
    parser.add_argument("--rev-url", default="http://127.0.0.1:8019")
    parser.add_argument("--decode", choices=("mean", "argmax"), default="mean")
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--horizon", type=float, default=240.0)
    parser.add_argument("--pedal-gain", type=float, default=1.25)
    parser.add_argument("--collect", type=int, default=0, help="DAgger: episodes per worker from random starts")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--teacher", type=Path, help="SAC checkpoint that rides a --beta share of collect steps")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = args.runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1) + "\n")
    rider = RevRider(args.rev_url, args.decode, args.pedal_gain)
    if args.collect:
        collect(args, run_dir, rider)
        return
    directory = run_dir / "eval"
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
            rows = [row for row, _ in pool.map(lambda pair: ride(pair[0], rider, pair[1], log),
                                               zip(envs, starts))]
    finally:
        decisions.close()
        for env in envs:
            env.close()
    with (directory / "eval.jsonl").open("a") as stream:
        for row in rows:
            recordings = list(directory.glob(f"**/{row['episode_id']}.jsonl"))
            row.update(policy_step=0, checkpoint=args.rev_url,
                       recording=str(recordings[0].relative_to(run_dir)) if recordings else None)
            stream.write(json.dumps(row) + "\n")
    laps = sorted(row["lap_time_s"] for row in rows if row["termination"] == "lap_completed")
    print(f"REV EVAL {args.run_name}: {len(laps)}/{len(rows)} laps {laps} s, "
          f"terminations {[row['termination'] for row in rows]}, {time.monotonic() - began:.0f} s")


def collect(args, run_dir: Path, rider: RevRider):
    from sac_teacher import SacTeacher
    teacher = SacTeacher(args.teacher, args.pedal_gain) if args.teacher else None
    if args.beta > 0 and teacher is None:
        raise SystemExit("--beta needs --teacher")
    decisions = (run_dir / "decisions.jsonl").open("a")

    def worker(index):
        env = ThunderhillSACEnv(godot=args.godot, data_dir=run_dir / "collect", horizon_s=args.horizon,
                                reward_line="progress", pedal_gain=args.pedal_gain, seed=args.seed * 1000 + index,
                                policy_id=f"{args.run_name}-collect")
        rng = np.random.default_rng(args.seed * 1000 + index)
        obs, episode, off, summaries = [], [], [], []
        try:
            for k in range(args.collect):
                summary, visited = ride(env, rider, None, lambda row: decisions.write(json.dumps(row) + "\n"),
                                        teacher, args.beta, rng)
                crashed = np.zeros(len(visited), dtype=bool)
                if summary["termination"] in OFF_TRACK_TERMINATIONS:
                    crashed[-OFF_TRACK_HORIZON_STEPS:] = True
                obs.append(visited)
                episode += [k] * len(visited)
                off.append(crashed)
                summaries.append(summary)
        finally:
            env.close()
        if obs:
            np.savez_compressed(run_dir / f"collect-{index:03d}.npz", obs=np.concatenate(obs).astype(np.float32),
                                episode=np.array(episode), off_track=np.concatenate(off))
        return summaries

    began = time.monotonic()
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            summaries = [s for rows in pool.map(worker, range(args.workers)) for s in rows]
    finally:
        decisions.close()
    with (run_dir / "episodes.jsonl").open("w") as stream:
        for row in summaries:
            stream.write(json.dumps(row) + "\n")
    steps = sum(row["steps"] for row in summaries)
    ends = {}
    for row in summaries:
        ends[row["termination"]] = ends.get(row["termination"], 0) + 1
    print(f"REV COLLECT {args.run_name}: {len(summaries)} episodes, {steps} states at beta {args.beta}, "
          f"ends {ends}, {time.monotonic() - began:.0f} s")


if __name__ == "__main__":
    main()
