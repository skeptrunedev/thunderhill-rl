# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.8.0",
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
"""Asynchronous actor-learner SAC on local headless Godot workers (the
environment is sac_env.py; the GPU learner is sac_learner.py).

  uv run training/sac_async.py --godot GODOT --run-name sac-async --workers 48
  uv run training/sac_async.py --godot GODOT --run-name sac-async --workers 48 --resume
  uv run training/sac_async.py --godot GODOT --run-name sac-multi --workers 28 --tracks all

On the 56-core / RTX 2080 Ti box: 48 workers step ~1,900 env steps/s (CPU bound:
each headless Godot uses ~0.9 core while stepping) and the learner sustains ~545
updates/s at batch 1024, so the default --utd 0.25 keeps both sides busy.

Each actor process owns one Godot worker and steps it continuously with a NumPy
copy of the policy, which it refreshes whenever the learner publishes new
weights (every --sync-every updates). Actors write transitions into per-actor
shared-memory rings; the learner process drains the rings into a GPU replay
buffer and runs CUDA-graph SAC updates at --utd updates per environment step,
never more. Actors pause when they run more than --max-lead environment steps
ahead of that ratio, so the update-to-data ratio stays controlled when the
learner is the slower side. An evaluator process runs the deterministic policy
from fixed starts with Godot recordings, without stopping training.

--tracks (all, or a comma list) trains one policy on several circuits: the
observation is track-relative, so the same network can drive any of them. Actor
i starts on track i mod T and, with --track-dwell, moves on to the next track
every that many seconds (a Godot restart of a few seconds, staggered across
actors), so every track gets the same share of data even when T does not divide
--workers. Rolling starts stay random along each track. The evaluator rides
--eval-starts fixed starts on every track, --eval-workers Godots at a time.

SIGINT/SIGTERM to the trainer (for a systemd unit: systemctl --user kill -s INT
--kill-whom=main UNIT) stops the actors, then saves unflushed rollouts, a final
checkpoint and the replay buffer; --resume continues from them.

Everything for a run lives in --runs-dir/RUN_NAME:
  checkpoints/step_N.pt, replay_buffer.npz    learner state (networks, optimizers,
                                              counters) and replay, for --resume
  rollouts/shard-NNNNN.npz                    every transition, unless --no-rollout-shards
                                              (~7 GB/hour at 1,900 steps/s): (obs, action, reward,
                                              next_obs, terminated, truncated, env,
                                              episode, policy_step = env step,
                                              policy_version = learner update of
                                              the acting weights, learner_update)
  rollouts/episodes.jsonl                     one row per training episode
  tracks.json                                 with --tracks: the track list that the
                                              shard column `track` indexes (episode
                                              rows and eval rows name the track)
  eval/eval.jsonl + eval/worker-*/            deterministic evaluations and their
                                              Godot recordings (render_run_video.py)
"""

from __future__ import annotations

import os

# Every process here does its CPU math on single rows; BLAS thread pools would
# only contend with the Godot workers.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from sac_env import (  # noqa: E402
    DEFAULT_TRACK,
    MAX_INITIAL_SPEED_M_S,
    MIN_START_SPEED_M_S,
    OBSERVATION_SIZE,
    ROOT,
    RoadTelemetry,
    ThunderhillSACEnv,
    Track,
    evaluation_starts,
)
from lap_policy import held_out  # noqa: E402

ACTION_SIZE = 2
TERMINATIONS = ("offroad", "collision", "fall", "stall", "horizon", "lap_completed",
                "worker_restart", "crash")

# One transition row in the shared rings (float32 columns, then int64 columns).
COLUMNS = dict(obs=OBSERVATION_SIZE, action=ACTION_SIZE, reward=1, next_obs=OBSERVATION_SIZE,
               terminated=1, truncated=1, restart=1)
OFFSETS, _offset = {}, 0
for _name, _width in COLUMNS.items():
    OFFSETS[_name] = slice(_offset, _offset + _width)
    _offset += _width
ROW_WIDTH = _offset
META = ("episode", "policy_version", "track")  # track: index into args.tracks


# Policy on the CPU ------------------------------------------------------------
def actor_shapes(hidden):
    """Parameter shapes in sac_learner.Actor.parameters() order."""
    sizes = [OBSERVATION_SIZE, *hidden]
    shapes = [shape for i, o in zip(sizes[:-1], sizes[1:]) for shape in ((o, i), (o,))]
    return shapes + [(ACTION_SIZE, sizes[-1]), (ACTION_SIZE,)] * 2


class NumpyPolicy:
    """The learner's actor evaluated with NumPy for one observation at a time."""

    def __init__(self, hidden):
        self.shapes = actor_shapes(hidden)
        self.version = -1
        self.layers = None

    def load(self, flat, version):
        arrays, start = [], 0
        for shape in self.shapes:
            size = math.prod(shape)
            arrays.append(flat[start:start + size].reshape(shape).copy())
            start += size
        self.layers = [(arrays[k].T.copy(), arrays[k + 1]) for k in range(0, len(arrays), 2)]
        self.version = version

    def act(self, obs, rng=None):
        x = obs
        for weight, bias in self.layers[:-2]:
            x = np.maximum(x @ weight + bias, 0.0)
        mean = x @ self.layers[-2][0] + self.layers[-2][1]
        if rng is None:
            return np.tanh(mean)
        log_std = np.clip(x @ self.layers[-1][0] + self.layers[-1][1], -20.0, 2.0)
        return np.tanh(mean + np.exp(log_std) * rng.standard_normal(ACTION_SIZE))


class Shared:
    """Shared memory between the learner and its actors.

    rows/meta[actor, slot] is a ring per actor with a single writer (the actor
    advances written[actor] after filling a slot) and a single reader (the
    learner advances consumed[actor]). Parameters are published under a
    sequence lock: odd while being written."""

    def __init__(self, context, actors, ring, parameter_count):
        self.actors, self.ring = actors, ring
        self._rows = context.RawArray("f", actors * ring * ROW_WIDTH)
        self._meta = context.RawArray("q", actors * ring * len(META))
        self._written = context.RawArray("q", actors)
        self._consumed = context.RawArray("q", actors)
        self._parameters = context.RawArray("f", parameter_count)
        self._control = context.RawArray("q", 3)  # sequence, version, allowed env steps
        self.episode_counter = context.Value("q", 0)
        self.stop = context.Event()
        self.summaries = context.Queue()

    def __getstate__(self):
        state = dict(self.__dict__)
        for name in ("rows", "meta", "written", "consumed", "parameters", "control"):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.views()

    def views(self):
        self.rows = np.frombuffer(self._rows, np.float32).reshape(self.actors, self.ring, ROW_WIDTH)
        self.meta = np.frombuffer(self._meta, np.int64).reshape(self.actors, self.ring, len(META))
        self.written = np.frombuffer(self._written, np.int64)
        self.consumed = np.frombuffer(self._consumed, np.int64)
        self.parameters = np.frombuffer(self._parameters, np.float32)
        self.control = np.frombuffer(self._control, np.int64)
        return self

    def publish(self, flat, version):
        self.control[0] += 1
        self.parameters[:] = flat
        self.control[1] = version
        self.control[0] += 1

    def refresh(self, policy: NumpyPolicy):
        """Load newly published weights, if any; version 0 means none yet."""
        while self.control[1] != policy.version:
            sequence = self.control[0]
            if sequence % 2:
                time.sleep(0.0005)
                continue
            version, flat = int(self.control[1]), self.parameters.copy()
            if self.control[0] == sequence:
                if version > 0:
                    policy.load(flat, version)
                policy.version = version

    def next_episode(self):
        with self.episode_counter.get_lock():
            self.episode_counter.value += 1
            return self.episode_counter.value - 1


# Actor processes ---------------------------------------------------------------
def _child_setup(niceness):
    # The trainer handles Ctrl-C and stops children through `stop`; SIGTERM (a
    # direct kill) unwinds so the Godot worker is closed.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if niceness:
        os.nice(niceness)
    return os.getppid()


def run_actor(index, args, shared: Shared):
    parent = _child_setup(args.actor_nice)
    tracks = args.tracks or [args.track]
    slot, rotation = index % len(tracks), 0

    def make_env():
        return ThunderhillSACEnv(godot=args.godot, data_dir=args.run_dir / "workers",
                                 horizon_s=args.horizon,
                                 seed=args.seed * 1000 + index + 1_000_000 * rotation,
                                 reward_line=args.reward_line, pedal_gain=args.pedal_gain,
                                 apex_bonus_m=args.apex_bonus_m,
                                 throttle_bonus_m=args.throttle_bonus_m,
                                 vehicle=args.vehicle, track=tracks[slot],
                                 policy_id=args.run_name)

    env = make_env()
    # Actors change track at staggered times, so only a few restart at once.
    rotate = bool(args.tracks) and args.track_dwell > 0 and len(tracks) > 1
    switch_at = time.monotonic() + args.track_dwell * (1 + index / args.workers)
    rng = np.random.default_rng([args.seed, index, int(time.time())])
    policy = NumpyPolicy(args.hidden)
    rows, meta = shared.rows[index], shared.meta[index]
    # --focus-fraction: this actor's recent failure stations per track; a share of its
    # episodes restart --focus-lead metres before one of them (or a distance drawn
    # uniformly from [--focus-lead-min, --focus-lead]), at 80% of the safe speed there.
    failures = {track: collections.deque(maxlen=64) for track in tracks}
    # --focus-file: stations another process wants practised too (e.g. missed apexes),
    # {"stations": [...]}, re-read whenever the file changes.
    focus_file, focus_stamp, focus_stations = args.focus_file, None, []

    def reset():
        nonlocal focus_stamp, focus_stations
        if focus_file is not None:
            try:
                stamp = focus_file.stat().st_mtime_ns
                if stamp != focus_stamp:
                    focus_stations = [float(x) for x in json.loads(focus_file.read_text())["stations"]]
                    focus_stamp = stamp
            except (FileNotFoundError, ValueError, KeyError):
                pass  # absent or mid-write: keep the last stations read
        # Failure spots and --focus-file stations are chosen equally often.
        pools = [pool for pool in (list(failures[tracks[slot]]), focus_stations) if pool]
        if pools and args.focus_fraction > 0 and rng.random() < args.focus_fraction:
            track = env.track
            spots = pools[int(rng.integers(len(pools)))]
            lead = rng.uniform(min(args.focus_lead_min, args.focus_lead), args.focus_lead)
            station = (spots[int(rng.integers(len(spots)))] - lead) % track.length
            limit = track.speed_limits[min(int(np.searchsorted(track._s, station, side="right") - 1),
                                           len(track.speed_limits) - 1)]
            speed = float(np.clip(0.8 * limit, MIN_START_SPEED_M_S, MAX_INITIAL_SPEED_M_S))
            return env.reset(options={"start": dict(station=round(float(station), 2),
                                                    speed=round(speed, 2))})
        return env.reset()

    try:
        obs, _ = reset()
        episode = shared.next_episode()
        while not shared.stop.is_set() and os.getppid() == parent:
            # Hold while ahead of the learner's update-to-data ratio or the ring is full.
            if (shared.written.sum() >= shared.control[2]
                    or shared.written[index] - shared.consumed[index] >= shared.ring):
                time.sleep(0.002)
                continue
            shared.refresh(policy)
            if policy.version > 0:
                action = policy.act(obs, rng).astype(np.float32)
            else:
                action = rng.uniform(-1.0, 1.0, ACTION_SIZE).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            row = rows[shared.written[index] % shared.ring]
            row[OFFSETS["obs"]] = obs
            row[OFFSETS["action"]] = action
            row[OFFSETS["reward"]] = reward
            row[OFFSETS["next_obs"]] = next_obs
            row[OFFSETS["terminated"]] = terminated
            row[OFFSETS["truncated"]] = truncated
            row[OFFSETS["restart"]] = info.get("worker_restart", False)
            meta[shared.written[index] % shared.ring] = (episode, max(policy.version, 0), slot)
            shared.written[index] += 1  # publishes the slot (x86 stores are ordered)
            if terminated or truncated:
                summary = dict(info["episode_summary"], env=index, episode=episode,
                               policy_version=max(policy.version, 0))
                if args.tracks:
                    summary["track"] = tracks[slot]
                shared.summaries.put(summary)
                if summary["termination"] in ("offroad", "fall", "collision"):
                    failures[tracks[slot]].append(
                        (summary["start_station_m"] + summary["legal_progress_m"]) % env.track.length)
                if rotate and time.monotonic() >= switch_at:
                    env.close()
                    slot, rotation = (slot + 1) % len(tracks), rotation + 1
                    env = make_env()
                    switch_at += args.track_dwell
                obs, _ = reset()
                episode = shared.next_episode()
            else:
                obs = next_obs
    finally:
        env.close()


# Evaluation process ------------------------------------------------------------
def run_evaluator(args, shared: Shared, requests, results):
    """Deterministic policy from fixed rolling starts spread around the lap, with
    Godot recordings kept for rendering."""
    parent = _child_setup(0)
    directory = args.run_dir / "eval"
    starts = evaluation_starts(args.eval_starts, Track(RoadTelemetry(track=args.track)))
    envs = [ThunderhillSACEnv(godot=args.godot, data_dir=directory, horizon_s=args.eval_horizon,
                              reward_line=args.reward_line, pedal_gain=args.pedal_gain,
                              vehicle=args.vehicle, track=args.track, record_godot=True,
                              policy_id=f"{args.run_name}-eval")
            for _ in starts]

    def episode(policy, env, start):
        obs, info = env.reset(options={"start": start})
        while not shared.stop.is_set():
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            if terminated or truncated:
                return info["episode_summary"]
        return None

    try:
        with ThreadPoolExecutor(len(envs)) as pool:
            while not shared.stop.is_set() and os.getppid() == parent:
                try:
                    step, update, flat, checkpoint = requests.get(timeout=1.0)
                except queue.Empty:
                    continue
                began = time.monotonic()
                policy = NumpyPolicy(args.hidden)
                policy.load(flat, update)
                rows = list(pool.map(lambda pair: episode(policy, *pair), zip(envs, starts)))
                if None in rows:
                    return
                with (directory / "eval.jsonl").open("a") as stream:
                    for row in rows:
                        recordings = list(directory.glob(f"**/{row['episode_id']}.jsonl"))
                        row.update(policy_step=step, learner_update=update, checkpoint=checkpoint,
                                   recording=(str(recordings[0].relative_to(args.run_dir))
                                              if recordings else None))
                        stream.write(json.dumps(row) + "\n")
                results.put((step, update, rows, time.monotonic() - began))
    finally:
        for env in envs:
            env.close()


def run_track_evaluator(args, shared: Shared, requests, results):
    """With --tracks: the deterministic policy from --eval-starts fixed rolling
    starts on every track, --eval-workers Godots at a time. Each thread starts
    one Godot per track, rides that track's starts, then closes it, so at most
    --eval-workers Godots run however many tracks there are."""
    parent = _child_setup(0)
    directory = args.run_dir / "eval"
    ridden = args.tracks + args.eval_only_tracks
    starts = {track: evaluation_starts(args.eval_starts, Track(RoadTelemetry(track=track)))
              for track in ridden}

    def ride(policy, track):
        env = ThunderhillSACEnv(godot=args.godot, data_dir=directory,
                                horizon_s=args.eval_horizon, reward_line=args.reward_line,
                                pedal_gain=args.pedal_gain, vehicle=args.vehicle, track=track,
                                record_godot=True, policy_id=f"{args.run_name}-eval")
        rows = []
        try:
            for start in starts[track]:
                obs, info = env.reset(options={"start": start})
                # The worker directory of the Godot that rode (and recorded) it.
                worker_dir = directory / f"worker-{env.tag}-{env._restarts - 1}"
                while not shared.stop.is_set():
                    obs, _, terminated, truncated, info = env.step(policy.act(obs))
                    if terminated or truncated:
                        break
                else:
                    return None
                recordings = list(worker_dir.glob(
                    f"**/{info['episode_summary']['episode_id']}.jsonl"))
                rows.append(dict(info["episode_summary"], track=track,
                                 held_out=track in args.eval_only_tracks, recording=(
                    str(recordings[0].relative_to(args.run_dir)) if recordings else None)))
        finally:
            env.close()
        return rows

    with ThreadPoolExecutor(args.eval_workers) as pool:
        while not shared.stop.is_set() and os.getppid() == parent:
            try:
                step, update, flat, checkpoint = requests.get(timeout=1.0)
            except queue.Empty:
                continue
            began = time.monotonic()
            policy = NumpyPolicy(args.hidden)
            policy.load(flat, update)
            per_track = list(pool.map(lambda track: ride(policy, track), ridden))
            if None in per_track:
                return
            rows = [row for track_rows in per_track for row in track_rows]
            with (directory / "eval.jsonl").open("a") as stream:
                for row in rows:
                    row.update(policy_step=step, learner_update=update, checkpoint=checkpoint)
                    stream.write(json.dumps(row) + "\n")
            results.put((step, update, rows, time.monotonic() - began))


# Learner-side recording --------------------------------------------------------
class Recorder:
    """Every transition to npz shards, every episode to JSONL, and the rollout
    aggregates for logging."""

    SHARD_FIELDS = ("obs", "action", "reward", "next_obs", "terminated", "truncated", "env",
                    "episode", "policy_step", "policy_version", "learner_update")

    def __init__(self, args, run_dir):
        self.args = args
        # With --tracks, a shard column `track` indexes run_dir/tracks.json.
        self.fields = self.SHARD_FIELDS + (("track",) if args.tracks else ())
        self.shards = not args.no_rollout_shards
        self.directory = run_dir / "rollouts"
        self.directory.mkdir(parents=True, exist_ok=True)
        shards = sorted(self.directory.glob("shard-*.npz"))
        self.shard_index = len(shards)
        # Episode numbers are unique within the run, continuing across resumes.
        episodes = [int(np.load(path)["episode"].max()) for path in shards]
        if not self.shards and (self.directory / "episodes.jsonl").exists():
            with (self.directory / "episodes.jsonl").open() as stream:
                episodes += [json.loads(line)["episode"] for line in stream]
        self.first_episode = 1 + max([-1] + episodes)
        self.episodes = (self.directory / "episodes.jsonl").open("a")
        self.pending = {k: [] for k in self.fields}
        self.pending_rows = 0
        self.writer, self.write = ThreadPoolExecutor(1), None
        self.recent = []
        self.laps = 0
        self.best_lap = math.inf

    def add_rows(self, rows):
        for k in self.fields:
            self.pending[k].append(rows[k])
        self.pending_rows += len(rows["reward"])
        if self.pending_rows >= self.args.shard_steps:
            self.flush()

    def add_episode(self, summary):
        self.episodes.write(json.dumps(summary) + "\n")
        self.recent.append(summary)
        if summary["lap_time_s"] is not None:
            self.laps += summary["laps"]
            if summary["lap_time_s"] < self.best_lap:
                self.best_lap = summary["lap_time_s"]
                print(f"NEW BEST TRAINING LAP {self.best_lap:.2f} s at env step "
                      f"{summary['policy_step']}", flush=True)

    def flush(self):
        if self.pending_rows:
            arrays = {k: np.concatenate(v) for k, v in self.pending.items()}
            # Compressing a shard takes seconds; zlib releases the GIL, so a
            # writer thread keeps the learner updating meanwhile.
            if self.write is not None:
                self.write.result()  # one shard in flight; surfaces write errors
            self.write = self.writer.submit(np.savez_compressed,
                                            self.directory / f"shard-{self.shard_index:05d}.npz",
                                            **arrays)
            self.shard_index += 1
            self.pending = {k: [] for k in self.fields}
            self.pending_rows = 0
        self.episodes.flush()

    def rollout_values(self):
        values = {"rollout/laps_total": self.laps}
        if math.isfinite(self.best_lap):
            values["rollout/best_lap_time_s"] = self.best_lap
        recent, self.recent = self.recent, []
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
            if self.args.tracks:
                values.update(self.track_values(recent))
        return values

    def track_values(self, recent):
        """Per-track episode count, failure shares and progress: a collapse shows
        first as one track's offroad share climbing."""
        values = {}
        for track in self.args.tracks:
            rows = [row for row in recent if row["track"] == track]
            if not rows:
                continue
            prefix = f"track/{track}/"
            values[prefix + "episodes"] = len(rows)
            values[prefix + "failed"] = sum(
                row["termination"] in ("offroad", "collision", "fall", "stall", "crash")
                for row in rows) / len(rows)
            values[prefix + "offroad"] = sum(
                row["termination"] == "offroad" for row in rows) / len(rows)
            values[prefix + "legal_progress_mean_m"] = float(
                np.mean([row["legal_progress_m"] for row in rows]))
        return values

    def close(self):
        self.flush()
        if self.write is not None:
            self.write.result()
        self.writer.shutdown()
        self.episodes.close()


# Learner -------------------------------------------------------------------------
class Trainer:
    def __init__(self, args, run_dir, wandb):
        import torch

        from sac_learner import SAC

        self.torch, self.args, self.run_dir, self.wandb = torch, args, run_dir, wandb
        self.sac = SAC(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
                       hidden=args.hidden, buffer_size=args.buffer_size,
                       batch_size=args.batch_size, learning_rate=args.learning_rate,
                       gamma=args.gamma, tau=args.tau, compile=not args.no_compile,
                       cuda_graph=not args.no_cuda_graph)
        self.env_steps = 0
        # Learner updates made before this run's env steps (--init-from); the
        # update-to-data accounting counts only this run's updates.
        self.update_offset = 0
        self.recorder = Recorder(args, run_dir)
        checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
        if args.resume and checkpoints:
            # Before any update: a captured graph would keep the replaced tensors.
            state = torch.load(checkpoints[-1], map_location="cuda", weights_only=False)
            self.sac.load_state_dict(state["sac"])
            self.env_steps = state["env_steps"]
            self.update_offset = state.get("update_offset", 0)
            if args.resume_weights_from:
                # Networks, optimizers and temperature from an earlier checkpoint of
                # the run (its best), keeping this resume's counters and replay.
                updates = self.sac.updates
                self.sac.load_state_dict(torch.load(args.resume_weights_from, map_location="cuda",
                                                    weights_only=False)["sac"])
                self.sac.updates = updates
                print(f"Weights from {args.resume_weights_from}", flush=True)
            if (run_dir / "replay_buffer.npz").exists():
                self.sac.buffer.load(run_dir / "replay_buffer.npz")
            print(f"Resumed {checkpoints[-1].name}: {self.env_steps} env steps, "
                  f"{self.sac.updates} updates, {self.sac.buffer.count} replay rows", flush=True)
        elif args.init_from:
            # Networks, optimizers and entropy temperature from another run's
            # checkpoint; an empty replay buffer, since its rewards may differ.
            state = torch.load(args.init_from, map_location="cuda", weights_only=False)
            self.sac.load_state_dict(state["sac"])
            self.update_offset = self.sac.updates
            print(f"Initialized from {args.init_from} ({self.update_offset} updates)", flush=True)

        # The actors' written counts start from zero in this process.
        self.resumed_steps = self.env_steps
        context = get_context("spawn")
        parameters = sum(math.prod(shape) for shape in actor_shapes(args.hidden))
        self.shared = Shared(context, args.workers, args.ring, parameters).views()
        self.shared.episode_counter.value = self.recorder.first_episode
        self.shared.control[2] = self._allowed_env_steps()
        if self.sac.updates:
            self.shared.publish(self.sac.actor.flat_parameters().cpu().numpy(), self.sac.updates)
        self.actors = [context.Process(target=run_actor, args=(i, args, self.shared),
                                       name=f"actor-{i}", daemon=True)
                       for i in range(args.workers)]
        self.eval_requests, self.eval_results = context.Queue(), context.Queue()
        self.evaluator = None
        if args.eval_every > 0:
            self.evaluator = context.Process(
                target=run_track_evaluator if args.tracks else run_evaluator,
                name="evaluator", daemon=True,
                args=(args, self.shared, self.eval_requests, self.eval_results))
        self.pending_publish = None
        self.eval_busy = False
        self.best_eval_lap = math.inf
        # Best evaluation lap per track so far, across resumes (--tracks).
        self.track_best = {}
        if args.tracks and (run_dir / "eval" / "eval.jsonl").exists():
            with (run_dir / "eval" / "eval.jsonl").open() as stream:
                for row in map(json.loads, stream):
                    if row["lap_time_s"] is not None:
                        self.track_best[row["track"]] = min(
                            self.track_best.get(row["track"], math.inf), row["lap_time_s"])
        self.staleness = []

    def _allowed_env_steps(self):
        args = self.args
        if args.max_lead < 0:
            return np.iinfo(np.int64).max
        # Env steps the learner's updates so far match at --utd, plus the lead,
        # less those of the run before a --resume (not in shared.written).
        own = self.sac.updates - self.update_offset
        return int(args.learning_starts + own / args.utd) + args.max_lead - self.resumed_steps

    def ingest(self):
        """Drain every actor ring into the replay buffer and the recorder."""
        shared, parts = self.shared, []
        for actor in range(shared.actors):
            written, consumed = int(shared.written[actor]), int(shared.consumed[actor])
            if written == consumed:
                continue
            slots = np.arange(consumed, written) % shared.ring
            parts.append((actor, shared.rows[actor, slots], shared.meta[actor, slots]))
            shared.consumed[actor] = written
        if not parts:
            return 0
        rows = np.concatenate([p[1] for p in parts])
        meta = np.concatenate([p[2] for p in parts])
        n = len(rows)
        column = lambda name: rows[:, OFFSETS[name]]  # noqa: E731
        valid = column("restart")[:, 0] == 0  # a worker restart is not an MDP transition
        self.sac.buffer.add({
            "obs": column("obs")[valid], "action": column("action")[valid],
            "reward": column("reward")[valid, 0], "next_obs": column("next_obs")[valid],
            "terminated": column("terminated")[valid, 0]})
        versions = meta[:, 1]
        if self.recorder.shards:
            self.record(parts, rows, meta, n)
        acted = versions > 0
        if acted.any():
            self.staleness.append(self.sac.updates - versions[acted])
        self.env_steps += n
        return n

    def record(self, parts, rows, meta, n):
        """Every ingested transition, restarts included, to the rollout shards."""
        column = lambda name: rows[:, OFFSETS[name]]  # noqa: E731
        record = {
            "obs": column("obs").copy(), "action": column("action").copy(),
            "reward": column("reward")[:, 0].copy(), "next_obs": column("next_obs").copy(),
            "terminated": column("terminated")[:, 0] > 0, "truncated": column("truncated")[:, 0] > 0,
            "env": np.concatenate([np.full(len(p[1]), p[0], np.int16) for p in parts]),
            "episode": meta[:, 0].copy(),
            "policy_step": self.env_steps + np.arange(n, dtype=np.int64),
            "policy_version": meta[:, 1].copy(),
            "learner_update": np.full(n, self.sac.updates, np.int64)}
        if self.args.tracks:
            record["track"] = meta[:, 2].astype(np.int16)
        self.recorder.add_rows(record)

    def drain_summaries(self):
        while True:
            try:
                summary = self.shared.summaries.get_nowait()
            except queue.Empty:
                return
            summary.update(policy_step=self.env_steps, learner_update=self.sac.updates,
                           time=time.time())
            self.recorder.add_episode(summary)

    def publish(self):
        """Hand the actor's weights to the actors without stalling the GPU queue."""
        if self.pending_publish is None:
            self.pending_publish = self.sac.snapshot_actor()
            self.last_publish = self.sac.updates
        host, event, version = self.pending_publish
        if event.query():
            self.shared.publish(host.numpy(), version)
            self.pending_publish = None

    def train(self):
        """Updates owed at --utd, in chunks that keep the loop responsive."""
        args, sac = self.args, self.sac
        if self.env_steps < args.learning_starts or sac.buffer.count < args.batch_size:
            return 0
        owed = int(args.utd * (self.env_steps - args.learning_starts)) - (
            sac.updates - self.update_offset)
        if owed <= 0:
            return 0
        if sac.updates == self.update_offset:
            print(f"Learning starts at {self.env_steps} env steps (compiling and capturing "
                  "the update)", flush=True)
        count = min(owed, args.update_chunk)
        sac.update(count)
        return count

    def checkpoint(self):
        directory = self.run_dir / "checkpoints"
        directory.mkdir(exist_ok=True)
        path = directory / f"step_{self.env_steps}.pt"
        self.torch.save(dict(sac=self.sac.state_dict(), env_steps=self.env_steps,
                             update_offset=self.update_offset, args=vars(self.args)), path)
        return path

    def request_eval(self):
        path = self.checkpoint()
        self.eval_requests.put((self.env_steps, self.sac.updates,
                                self.sac.actor.flat_parameters().cpu().numpy(),
                                str(path.relative_to(self.run_dir))))
        self.eval_busy = True

    def collect_eval(self):
        try:
            step, update, rows, seconds = self.eval_results.get_nowait()
        except queue.Empty:
            return
        self.eval_busy = False
        if self.args.tracks:
            self.collect_track_eval(step, update, rows, seconds)
            return
        laps = [r["lap_time_s"] for r in rows if r["lap_time_s"] is not None]
        values = {
            "global_step": step,
            "eval/learner_update": update,
            "eval/centered_progress_mean_m": float(np.mean([r["centered_progress_m"] for r in rows])),
            "eval/legal_progress_max_m": float(max(r["legal_progress_m"] for r in rows)),
            "eval/return_mean": float(np.mean([r["return"] for r in rows])),
            "eval/ep_len_mean_s": float(np.mean([r["steps"] for r in rows])) / 10,
            "eval/laps_completed": len(laps),
            "eval/wall_seconds": seconds,
        }
        if laps:
            values["eval/best_lap_time_s"] = min(laps)
            values["eval/mean_lap_time_s"] = float(np.mean(laps))
            self.best_eval_lap = min(self.best_eval_lap, min(laps))
        if math.isfinite(self.best_eval_lap):
            values["eval/best_lap_time_ever_s"] = self.best_eval_lap
        if self.wandb:
            self.wandb.log(values)
        lap_text = ", ".join(f"{t:.2f}" for t in sorted(laps)) or "none"
        print(f"EVAL env step {step} update {update}: {len(laps)}/{len(rows)} laps "
              f"[{lap_text}] s, best eval lap ever {self.best_eval_lap:.2f} s, "
              f"elapsed {time.monotonic() - self.started:.0f} s", flush=True)
        print(json.dumps({"eval": values, "episodes": [
            {k: r[k] for k in ("termination", "steps", "centered_progress_m", "lap_time_s")}
            for r in rows]}), flush=True)

    def collect_track_eval(self, step, update, rows, seconds):
        """Per-track laps, lap times and terminations; aggregates are the laps
        completed out of all evaluation episodes and the median over tracks of
        this evaluation's best lap relative to that track's best so far. Held-out
        tracks (--eval-only-tracks, never trained on) are reported on their own."""
        held_out = [r for r in rows if r.get("held_out")]
        rows = [r for r in rows if not r.get("held_out")]
        values = {"global_step": step, "eval/learner_update": update,
                  "eval/wall_seconds": seconds, "eval/episodes": len(rows),
                  "eval/laps_completed": sum(r["lap_time_s"] is not None for r in rows)}
        relative, lines = [], []
        for track in self.args.tracks:
            track_rows = [r for r in rows if r["track"] == track]
            laps = [r["lap_time_s"] for r in track_rows if r["lap_time_s"] is not None]
            prefix = f"eval_track/{track}/"
            values[prefix + "laps"] = len(laps)
            values[prefix + "legal_progress_mean_m"] = float(
                np.mean([r["legal_progress_m"] for r in track_rows]))
            if laps:
                self.track_best[track] = min(self.track_best.get(track, math.inf), min(laps))
                relative.append(min(laps) / self.track_best[track])
                values[prefix + "best_lap_time_s"] = min(laps)
            if track in self.track_best:
                values[prefix + "best_lap_time_ever_s"] = self.track_best[track]
            terminations = ",".join(r["termination"] for r in track_rows)
            lines.append(f"  {track:<16} {len(laps)}/{len(track_rows)} laps  "
                         f"{min(laps) if laps else math.nan:7.2f} s  (best "
                         f"{self.track_best.get(track, math.nan):7.2f})  {terminations}")
        values["eval/tracks_with_lap"] = len(relative)
        if relative:
            values["eval/median_relative_lap"] = float(np.median(relative))
        for reason in TERMINATIONS:
            values[f"eval/termination/{reason}"] = sum(
                r["termination"] == reason for r in rows) / len(rows)
        if self.wandb:
            self.wandb.log(values)
        print(f"EVAL env step {step} update {update}: {values['eval/laps_completed']}/{len(rows)} "
              f"laps on {len(relative)}/{len(self.args.tracks)} tracks, median relative lap "
              f"{values.get('eval/median_relative_lap', math.nan):.3f}, {seconds:.0f} s, "
              f"elapsed {time.monotonic() - self.started:.0f} s", flush=True)
        print("\n".join(lines), flush=True)
        if held_out:
            lines = []
            for track in self.args.eval_only_tracks:
                track_rows = [r for r in held_out if r["track"] == track]
                laps = [r["lap_time_s"] for r in track_rows if r["lap_time_s"] is not None]
                values[f"eval_heldout/{track}/laps"] = len(laps)
                values[f"eval_heldout/{track}/legal_progress_mean_m"] = float(
                    np.mean([r["legal_progress_m"] for r in track_rows]))
                if laps:
                    values[f"eval_heldout/{track}/best_lap_time_s"] = min(laps)
                lines.append(f"  {track:<16} {len(laps)}/{len(track_rows)} laps  "
                             f"{min(laps) if laps else math.nan:7.2f} s  progress "
                             f"{[round(r['legal_progress_m']) for r in track_rows]}  "
                             f"{','.join(r['termination'] for r in track_rows)}")
            if self.wandb:
                self.wandb.log({k: v for k, v in values.items() if k.startswith("eval_heldout/")}
                               | {"global_step": step})
            print(f"HELD-OUT EVAL env step {step}: "
                  f"{sum(r['lap_time_s'] is not None for r in held_out)}/{len(held_out)} laps on "
                  f"tracks never trained on", flush=True)
            print("\n".join(lines), flush=True)

    def log(self):
        now = time.monotonic()
        seconds = now - self.last_log_time
        steps = self.env_steps - self.last_log_steps
        updates = self.sac.updates - self.last_log_updates
        staleness = np.concatenate(self.staleness) if self.staleness else np.zeros(0)
        values = {
            "global_step": self.env_steps,
            "time/steps_per_second": steps / seconds,
            "time/updates_per_second": updates / seconds,
            "time/utd": updates / max(steps, 1),
            "time/utd_total": (self.sac.updates - self.update_offset)
            / max(self.env_steps - self.args.learning_starts, 1),
            "time/elapsed_s": now - self.started,
            "time/updates": self.sac.updates,
            "time/actors_held_frac": self.held_loops / max(self.loops, 1),
            "replay/size": self.sac.buffer.count,
            **self.recorder.rollout_values(),
            **self.sac.metric_values(),
        }
        if len(staleness):
            values["time/actor_staleness_mean"] = float(staleness.mean())
            values["time/actor_staleness_max"] = float(staleness.max())
        if self.wandb:
            self.wandb.log(values)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in values.items()}), flush=True)
        self.last_log_time, self.last_log_steps = now, self.env_steps
        self.last_log_updates, self.staleness = self.sac.updates, []
        self.held_loops = self.loops = 0

    def run(self, stop_requested):
        args = self.args
        for process in self.actors + ([self.evaluator] if self.evaluator else []):
            process.start()
        self.started = self.last_log_time = time.monotonic()
        self.last_log_steps, self.last_log_updates = self.env_steps, self.sac.updates
        self.last_checkpoint = self.last_eval = self.env_steps
        self.last_publish = self.sac.updates
        self.held_loops = self.loops = 0
        if self.evaluator and self.sac.updates and self.env_steps == 0:
            # --init-from: the starting weights under this run's evaluation, a
            # baseline (and a candidate) for what the run then learns.
            self.request_eval()
        last_health_check, failure = time.monotonic(), None
        try:
            while not stop_requested() and self.env_steps < args.total_steps:
                ingested = self.ingest()
                self.drain_summaries()
                trained = self.train()
                if self.sac.updates and (self.pending_publish is not None
                                         or self.sac.updates - self.last_publish >= args.sync_every
                                         or self.shared.control[1] == 0):
                    self.publish()
                self.shared.control[2] = self._allowed_env_steps()
                self.loops += 1
                self.held_loops += int(self.shared.written.sum() >= self.shared.control[2])
                if self.evaluator:
                    self.collect_eval()
                    if (not self.eval_busy and self.sac.updates
                            and self.env_steps - self.last_eval >= args.eval_every):
                        self.last_eval = self.env_steps
                        self.request_eval()
                if self.env_steps - self.last_log_steps >= args.log_every:
                    self.log()
                if self.env_steps - self.last_checkpoint >= args.checkpoint_every:
                    self.last_checkpoint = self.env_steps
                    self.checkpoint()
                if time.monotonic() - last_health_check > 1.0:
                    last_health_check = time.monotonic()
                    dead = [f"{p.name} (exit {p.exitcode})"
                            for p in self.actors + [self.evaluator] if p and not p.is_alive()]
                    if dead:
                        failure = f"child processes exited: {dead}"
                        break
                if not ingested and not trained:
                    time.sleep(0.001)
        finally:
            self.shutdown()
        if failure:
            raise RuntimeError(failure)

    def shutdown(self):
        """Stop the actors, then keep everything they produced."""
        self.shared.stop.set()
        for process in self.actors + ([self.evaluator] if self.evaluator else []):
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        self.ingest()
        self.drain_summaries()
        self.recorder.close()
        path = self.checkpoint()
        self.sac.buffer.save(self.run_dir / "replay_buffer.npz")
        print(f"Saved {path.name} and the replay buffer ({self.sac.buffer.count} rows) at "
              f"{self.env_steps} env steps, {self.sac.updates} updates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--godot", default=os.environ.get("GODOT"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "sac")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the run's latest checkpoint and replay buffer")
    parser.add_argument("--resume-weights-from", type=Path,
                        help="with --resume: take the networks, optimizers and entropy "
                        "temperature from this checkpoint (e.g. the run's best) instead of "
                        "the latest, keeping the latest's step counters and the replay buffer")
    parser.add_argument("--apex-bonus-m", type=float, default=0.0,
                        help="opt-in shaping: metres of progress credited once per required "
                        "apex (tools/apex_report.py), scaled by the inside reach achieved "
                        "there, full at the judge's 0.75")
    parser.add_argument("--throttle-bonus-m", type=float, default=0.0,
                        help="opt-in shaping: metres of progress per step at full applied "
                        "throttle while upright (|lean| < 0.2) and off the front brake")
    parser.add_argument("--init-from", type=Path,
                        help="start a new run from another run's checkpoint networks, "
                        "with an empty replay buffer")
    parser.add_argument("--reward-line", choices=("progress", "centered"), default="progress",
                        help="progress: centerline progress, so the policy may use the track "
                        "width; centered: reward v6 weighting toward the centerline")
    parser.add_argument("--pedal-gain", type=float, default=1.25,
                        help="scale on the throttle/brake action before clipping, so full "
                        "throttle and full brake are reachable without saturating tanh")
    parser.add_argument("--vehicle", choices=("motorcycle", "car"), default="motorcycle",
                        help="simulated vehicle: the motorcycle (motorcycle.gd) or the MX-5 Cup "
                        "car (car.gd, docs/car-reference.md)")
    parser.add_argument("--track", default="thunderhill-east",
                        help="circuit: thunderhill-east or a MotoGP circuit built by "
                        "tools/build_circuit.py (godot/tracks/<id>)")
    parser.add_argument("--tracks",
                        help="train on several circuits at once: 'all' (thunderhill-east and "
                        "every circuit with a track.json) or a comma list; overrides --track")
    parser.add_argument("--eval-only-tracks", default="",
                        help="with --tracks: comma list of circuits the evaluator also rides but "
                        "no actor trains on, to measure generalization to unseen tracks")
    parser.add_argument("--focus-fraction", type=float, default=0.0,
                        help="share of an actor's episodes that restart before one of its recent "
                        "failures on that track (offroad, fall, collision)")
    parser.add_argument("--focus-file", type=Path,
                        help="JSON {\"stations\": [...]} of extra focus stations (single track), "
                        "re-read when it changes; picked as often as the failure spots")
    parser.add_argument("--focus-lead-min", type=float, default=math.inf,
                        help="draw each focused start's lead uniformly from [this, --focus-lead]; "
                        "default: always --focus-lead")
    parser.add_argument("--focus-lead", type=float, default=150.0,
                        help="metres before a failure station that a focused episode starts")
    parser.add_argument("--track-dwell", type=float, default=900.0,
                        help="with --tracks: seconds an actor stays on a track before moving to "
                        "the next one (a Godot restart); 0 keeps each actor on one track")
    parser.add_argument("--workers", type=int, default=48, help="actor processes (one Godot each)")
    parser.add_argument("--actor-nice", type=int, default=5,
                        help="niceness added to actors and their Godot workers, so the "
                        "learner's single thread is never starved")
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--horizon", type=float, default=60.0, help="training episode seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--utd", type=float, default=0.25,
                        help="gradient updates per environment step (the learner never exceeds it)")
    parser.add_argument("--max-lead", type=int, default=20_000,
                        help="environment steps actors may run ahead of --utd before pausing; "
                        "negative disables the pause")
    parser.add_argument("--update-chunk", type=int, default=16,
                        help="updates queued per learner loop iteration")
    parser.add_argument("--sync-every", type=int, default=20,
                        help="publish the actor's weights every N updates")
    parser.add_argument("--ring", type=int, default=4096, help="transitions per actor ring")
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--net-arch", default="256,256,256")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--log-every", type=int, default=20_000)
    parser.add_argument("--shard-steps", type=int, default=200_000)
    parser.add_argument("--no-rollout-shards", action="store_true",
                        help="skip the per-transition rollout shards; episodes.jsonl, "
                        "evaluations and checkpoints are still written")
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=500_000,
                        help="env steps between evaluations (~4 min at 1,900 steps/s; each "
                        "keeps ~165 MB of Godot recordings)")
    parser.add_argument("--eval-starts", type=int, default=4,
                        help="fixed evaluation starts (per track with --tracks)")
    parser.add_argument("--eval-workers", type=int, default=4,
                        help="with --tracks: Godots the evaluator runs at once")
    parser.add_argument("--eval-horizon", type=float, default=240.0)
    parser.add_argument("--wandb-project", default="thunderhill-rl")
    parser.add_argument("--wandb-entity", default="skeptrune-org")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if not args.godot:
        parser.error("Provide --godot or set GODOT")
    args.hidden = [int(x) for x in args.net_arch.split(",")]
    args.run_dir = run_dir = (args.runs_dir / args.run_name).resolve()
    if run_dir.exists() and not args.resume:
        parser.error(f"{run_dir} exists; pass --resume or choose another --run-name")
    if args.tracks:
        circuits = ROOT / "godot" / "tracks"
        known = sorted(path.parent.name for path in circuits.glob("*/track.json"))
        # "all" means every training circuit: held-out test tracks never enter training.
        trainable = [t for t in known if not held_out(t)]
        args.tracks = ([DEFAULT_TRACK, *trainable] if args.tracks == "all"
                       else [t.strip() for t in args.tracks.split(",") if t.strip()])
        for track in args.tracks:
            if track != DEFAULT_TRACK and track not in known:
                parser.error(f"Unknown track {track!r}; available: {[DEFAULT_TRACK, *known]}")
            if track in known and held_out(track):
                parser.error(f"{track} is a held-out test track; it is never trained on")
            if track != DEFAULT_TRACK and not (circuits / track / "generated" / "imagery.json").exists():
                parser.error(f"Circuit {track} is not built: uv run tools/build_circuit.py {track}")
    args.eval_only_tracks = [t.strip() for t in args.eval_only_tracks.split(",") if t.strip()]
    if args.eval_only_tracks:
        if not args.tracks:
            parser.error("--eval-only-tracks needs --tracks")
        for track in args.eval_only_tracks:
            if track in args.tracks:
                parser.error(f"{track} is a training track; --eval-only-tracks must be held out")
            if not (ROOT / "godot" / "tracks" / track / "generated" / "imagery.json").exists():
                parser.error(f"Circuit {track} is not built: uv run tools/build_circuit.py {track}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.tracks:
        # Shards store a track index; the list it indexes must not change on --resume.
        listing = run_dir / "tracks.json"
        if listing.exists() and json.loads(listing.read_text()) != args.tracks:
            parser.error(f"{listing} lists other tracks than --tracks")
        listing.write_text(json.dumps(args.tracks, indent=1) + "\n")

    wandb = None
    if not args.no_wandb:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group="sac",
                   name=args.run_name, id=args.run_name.replace("/", "-"), resume="allow",
                   config=vars(args) | {"run_dir": str(run_dir), "trainer": "sac_async"},
                   dir=str(run_dir))
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    try:
        Trainer(args, run_dir, wandb).run(lambda: bool(stop))
    finally:
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
