# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.8.0",
#   "stable-baselines3==2.7.0",
#   "gymnasium>=1.1",
#   "numpy",
#   "wandb",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""SAC with a small MLP on low-dimensional telemetry, trained on local headless
Godot workers (see sac_env.py for the observation, action, reward and resets).

  uv run training/sac_train.py --godot GODOT --run-name sac-pilot --workers 32
  uv run training/sac_train.py --godot GODOT --run-name sac-pilot --resume

Everything for a run lives in --runs-dir/RUN_NAME:
  checkpoints/step_N.zip, replay_buffer.pkl   policy and optimizer state, resume
  rollouts/shard-NNNNN.npz                    every training transition (obs,
                                              action, reward, terminated,
                                              truncated, env, episode, policy step)
  rollouts/episodes.jsonl                     one row per training episode
  eval/eval.jsonl + eval/worker-*/            deterministic evaluations and their
                                              Godot recordings (render_run_video.py)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv

from sac_env import (
    MAX_INITIAL_SPEED_M_S,
    MIN_START_SPEED_M_S,
    ROOT,
    ThunderhillSACEnv,
    Track,
)
from lap_policy import RoadTelemetry

TERMINATIONS = ("offroad", "collision", "fall", "stall", "horizon", "lap_completed",
                "worker_restart", "crash")


def make_env(args, index, data_dir):
    def factory():
        return ThunderhillSACEnv(godot=args.godot, data_dir=data_dir, horizon_s=args.horizon,
                                 seed=args.seed * 1000 + index, policy_id=args.run_name)
    return factory


class WandbWriter(KVWriter):
    """Forward SB3's own training scalars (losses, entropy coefficient) to W&B."""

    def __init__(self, wandb):
        self.wandb = wandb

    def write(self, key_values, key_excluded, step=0):
        values = {k: v for k, v in key_values.items()
                  if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)}
        self.wandb.log({**values, "global_step": step})

    def close(self):
        pass


class RolloutRecorder(BaseCallback):
    """Records every training transition to npz shards and every episode to JSONL,
    logs rollout aggregates, checkpoints and runs periodic evaluation."""

    def __init__(self, args, run_dir, evaluator, wandb):
        super().__init__()
        self.args, self.run_dir, self.evaluator, self.wandb = args, run_dir, evaluator, wandb
        self.rollouts = run_dir / "rollouts"
        self.rollouts.mkdir(parents=True, exist_ok=True)
        self.shard_index = len(list(self.rollouts.glob("shard-*.npz")))
        self.episodes_file = (self.rollouts / "episodes.jsonl").open("a")
        self.rows = {k: [] for k in ("obs", "action", "reward", "terminated", "truncated",
                                     "env", "episode", "policy_step")}
        self.recent = []
        self.best_lap = math.inf
        self.laps = 0

    def _on_training_start(self):
        count = self.training_env.num_envs
        # Episode numbers are unique within the run, continuing across resumes.
        first = 1 + max([-1] + [int(np.load(path)["episode"].max())
                                for path in self.rollouts.glob("shard-*.npz")])
        self.episode = np.arange(first, first + count)
        self.next_episode = first + count
        self.pending = 0
        self.last_log = self.last_checkpoint = self.last_eval = self.num_timesteps
        self.last_log_time = time.monotonic()

    def _on_step(self):
        step = self.num_timesteps
        infos, dones = self.locals["infos"], self.locals["dones"]
        truncated = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        rows = self.rows
        rows["obs"].append(self.model._last_obs.astype(np.float32))
        rows["action"].append(np.asarray(self.locals["actions"], dtype=np.float32))
        rows["reward"].append(np.asarray(self.locals["rewards"], dtype=np.float32))
        rows["terminated"].append(dones & ~truncated)
        rows["truncated"].append(dones & truncated)
        rows["env"].append(np.arange(len(dones), dtype=np.int16))
        rows["episode"].append(self.episode.copy())
        rows["policy_step"].append(np.full(len(dones), step, dtype=np.int64))
        for index, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            summary = dict(info["episode_summary"], env=index,
                           episode=int(self.episode[index]),
                           policy_step=step, time=time.time())
            self.episode[index] = self.next_episode
            self.next_episode += 1
            self.episodes_file.write(json.dumps(summary) + "\n")
            self.recent.append(summary)
            if summary["lap_time_s"] is not None:
                self.laps += summary["laps"]
                self.best_lap = min(self.best_lap, summary["lap_time_s"])
        self.pending += len(dones)
        if self.pending >= self.args.shard_steps:
            self._flush()
        if step - self.last_log >= self.args.log_every:
            self._log(step)
        if step - self.last_checkpoint >= self.args.checkpoint_every:
            self._checkpoint(step)
        if self.evaluator and step - self.last_eval >= self.args.eval_every:
            self._checkpoint(step)
            self.last_eval = step
            self.evaluator.run(self.model, step, self.wandb)
            self.last_log_time = time.monotonic()  # keep eval time out of steps/s
        return True

    def _flush(self):
        if not self.rows["reward"]:
            return
        arrays = {k: np.concatenate(v) for k, v in self.rows.items()}
        path = self.rollouts / f"shard-{self.shard_index:05d}.npz"
        np.savez_compressed(path, **arrays)
        self.shard_index += 1
        self.rows = {k: [] for k in self.rows}
        self.pending = 0
        self.episodes_file.flush()

    def _log(self, step):
        now = time.monotonic()
        values = {"global_step": step,
                  "time/steps_per_second": (step - self.last_log) / (now - self.last_log_time),
                  "rollout/laps_total": self.laps}
        if math.isfinite(self.best_lap):
            values["rollout/best_lap_time_s"] = self.best_lap
        recent = self.recent
        if recent:
            progress = [row["centered_progress_m"] for row in recent]
            values.update({
                "rollout/episodes": len(recent),
                "rollout/ep_return_mean": float(np.mean([row["return"] for row in recent])),
                "rollout/centered_progress_mean_m": float(np.mean(progress)),
                "rollout/centered_progress_max_m": float(np.max(progress)),
                "rollout/legal_progress_max_m": float(max(r["legal_progress_m"] for r in recent)),
                "rollout/ep_len_mean_s": float(np.mean([row["steps"] for row in recent])) / 10,
            })
            for reason in TERMINATIONS:
                values[f"termination/{reason}"] = sum(
                    row["termination"] == reason for row in recent) / len(recent)
        if self.wandb:
            self.wandb.log(values)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in values.items()}), flush=True)
        self.recent, self.last_log, self.last_log_time = [], step, now

    def _checkpoint(self, step):
        directory = self.run_dir / "checkpoints"
        directory.mkdir(exist_ok=True)
        self.model.save(directory / f"step_{step}.zip")
        self.last_checkpoint = step
        self.last_checkpoint_path = directory / f"step_{step}.zip"

    def _on_training_end(self):
        self._flush()
        self._checkpoint(self.num_timesteps)
        self.model.save_replay_buffer(self.run_dir / "replay_buffer.pkl")
        self.episodes_file.close()


class Evaluator:
    """Deterministic policy from fixed rolling starts spread around the lap, with
    Godot recordings kept for rendering."""

    def __init__(self, args, run_dir):
        track = Track(RoadTelemetry())
        starts = []
        for k in range(args.eval_starts):
            station = k * track.length / args.eval_starts
            index = int(np.searchsorted(track._s, station, side="right") - 1)
            speed = float(np.clip(0.8 * track.speed_limits[index], MIN_START_SPEED_M_S,
                                  MAX_INITIAL_SPEED_M_S))
            starts.append(dict(station=round(station, 2), speed=round(speed, 2)))
        self.starts = starts
        self.directory = run_dir / "eval"
        self.envs = [ThunderhillSACEnv(godot=args.godot, data_dir=self.directory,
                                       horizon_s=args.eval_horizon, record_godot=True,
                                       policy_id=f"{args.run_name}-eval")
                     for _ in starts]
        self.log = (self.directory / "eval.jsonl")

    def _episode(self, model, env, start):
        obs, info = env.reset(options={"start": start})
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return info["episode_summary"]

    def run(self, model, step, wandb):
        began = time.monotonic()
        with ThreadPoolExecutor(len(self.envs)) as pool:
            results = list(pool.map(lambda pair: self._episode(model, *pair),
                                    zip(self.envs, self.starts)))
        with self.log.open("a") as stream:
            for row in results:
                recordings = list(self.directory.glob(f"**/{row['episode_id']}.jsonl"))
                stream.write(json.dumps(dict(
                    row, policy_step=step, checkpoint=f"checkpoints/step_{step}.zip",
                    recording=str(recordings[0]) if recordings else None)) + "\n")
        laps = [r["lap_time_s"] for r in results if r["lap_time_s"] is not None]
        values = {
            "global_step": step,
            "eval/centered_progress_mean_m": float(np.mean([r["centered_progress_m"] for r in results])),
            "eval/legal_progress_max_m": float(max(r["legal_progress_m"] for r in results)),
            "eval/return_mean": float(np.mean([r["return"] for r in results])),
            "eval/ep_len_mean_s": float(np.mean([r["steps"] for r in results])) / 10,
            "eval/laps_completed": len(laps),
            "eval/wall_seconds": time.monotonic() - began,
        }
        if laps:
            values["eval/best_lap_time_s"] = min(laps)
        if wandb:
            wandb.log(values)
        print(json.dumps({"eval": values, "episodes": [
            {k: r[k] for k in ("termination", "steps", "centered_progress_m", "lap_time_s")}
            for r in results]}), flush=True)

    def close(self):
        for env in self.envs:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--godot", default=os.environ.get("GODOT"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "sac")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the run's latest checkpoint and replay buffer")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--horizon", type=float, default=60.0, help="training episode seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=8,
                        help="gradient steps per vectorized step (one transition per worker)")
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--net-arch", default="256,256,256")
    parser.add_argument("--log-every", type=int, default=20_000)
    parser.add_argument("--shard-steps", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--eval-starts", type=int, default=4)
    parser.add_argument("--eval-horizon", type=float, default=240.0)
    parser.add_argument("--wandb-project", default="thunderhill-rl")
    parser.add_argument("--wandb-entity", default="skeptrune-org")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if not args.godot:
        parser.error("Provide --godot or set GODOT")

    run_dir = args.runs_dir / args.run_name
    if run_dir.exists() and not args.resume:
        parser.error(f"{run_dir} exists; pass --resume or choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    wandb = None
    if not args.no_wandb:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group="sac",
                   name=args.run_name, id=args.run_name.replace("/", "-"), resume="allow",
                   config=vars(args) | {"run_dir": str(run_dir)}, dir=str(run_dir))
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    env = SubprocVecEnv([make_env(args, i, run_dir / "workers") for i in range(args.workers)],
                        start_method="forkserver")
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.zip"),
                         key=lambda p: int(p.stem.split("_")[1]))
    if args.resume and checkpoints:
        model = SAC.load(checkpoints[-1], env=env, device="cuda")
        if (run_dir / "replay_buffer.pkl").exists():
            model.load_replay_buffer(run_dir / "replay_buffer.pkl")
        print(f"Resumed {checkpoints[-1]} at {model.num_timesteps} steps", flush=True)
    else:
        model = SAC(
            "MlpPolicy", env, learning_rate=args.learning_rate, buffer_size=args.buffer_size,
            learning_starts=args.learning_starts, batch_size=args.batch_size, tau=0.005,
            gamma=args.gamma, train_freq=1, gradient_steps=args.gradient_steps,
            ent_coef="auto", seed=args.seed, device="cuda",
            policy_kwargs=dict(net_arch=[int(x) for x in args.net_arch.split(",")]),
        )
    model.set_logger(Logger(None, [WandbWriter(wandb)] if wandb else []))
    evaluator = Evaluator(args, run_dir) if args.eval_every > 0 else None
    try:
        model.learn(total_timesteps=args.total_steps, log_interval=20,
                    callback=RolloutRecorder(args, run_dir, evaluator, wandb),
                    reset_num_timesteps=not args.resume)
    finally:
        env.close()
        if evaluator:
            evaluator.close()
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
