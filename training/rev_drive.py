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
"""Ride any circuit with Rev, our decision model: the SAC evaluation, driven by typed questions.

Every 0.1 s control step renders the observation as rev_dataset's telemetry state, headed
by the circuit's name, and asks a Rev server (kev.serve, /v1/systemone) the training
questions word for word: steer and pedal on every step, off_track_soon on a sampled
--off-track-fraction of them (riding needs only the first two; the third is logged as a
diagnostic). The action is the probability-weighted steer and pedal level
(rev_dataset.expected_level), or the most likely level with --decode argmax. The
simulator waits for each action, so model latency slows the evaluation but never
changes the ride.

Evaluation rides --starts fixed rolling starts (sac_env.evaluation_starts) on every
--tracks circuit and --heldout-starts on every --heldout-tracks test track, --workers
at a time, with Godot recordings kept, and writes:

  eval/eval.jsonl  one row per ride, the sac_async eval row format plus track and held_out
                   (lap time, termination, recording path), so the apex and video tools read it
  decisions.jsonl  every step: track, station, action, the answers' probabilities, the
                   off_track_soon probability when asked, and the model latency
  summary.json     per circuit: laps, lap times, progress, terminations

With --collect N, rides instead gather DAgger states: --workers x N episodes from random
rolling starts (--horizon 60), split across the --tracks circuits in proportion to
--track-weights, a --focus-fraction of each circuit's episodes starting before one of
its failures in the --focus-from rides. The circuit's SAC teacher (--teachers registry,
sac_teacher.Teachers) takes each step with probability --beta, and every visited
observation is saved with its circuit and off-track outcome to collect-*.npz for
rev_dataset.py --dagger. Held-out test tracks are never collected on.

Every process that rides takes one of REV_GODOT_SLOTS (16) machine-wide Godot slots for
each env (lock files, released when the env closes or the process dies), so Rev's
evaluations and collections together never run more Godot workers than that.

  uv run --extra serve python training/rev_serve.py --run runs/rev/rev-0.8b-r8/checkpoint   # in the kev repo
  uv run training/rev_drive.py --godot GODOT --run-name rev-0.8b-r8-eval --tracks all
  uv run training/rev_drive.py --godot GODOT --run-name collect-m1 --collect 2 --workers 16 --tracks all \\
      --horizon 60 --beta 0.5 --teachers runs/sac/specialists.json --focus-from runs/rev/rev-0.8b-r8-eval
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_dataset import PEDAL_LEVELS, QUESTIONS, STEER_LEVELS, expected_level, render_state  # noqa: E402
from rev_dataset import OFF_TRACK_HORIZON_STEPS, OFF_TRACK_TERMINATIONS  # noqa: E402
from sac_env import (  # noqa: E402
    MAX_INITIAL_SPEED_M_S,
    MIN_START_SPEED_M_S,
    ROOT,
    ThunderhillSACEnv,
    Track,
    evaluation_starts,
)
from lap_policy import DEFAULT_TRACK, RoadTelemetry, held_out  # noqa: E402

FOCUS_LEAD_M = 150.0  # focus rides start this far before a failure, at 80% of the safe speed there
HELD_OUT_TRACKS = "portimao,laguna-seca"
RIDE_QUESTIONS = {qid: QUESTIONS[qid] for qid in ("steer", "pedal")}
SLOTS = Path.home() / ".cache/thunderhill-rev/godot-slots"
REQUEST_TIMEOUT_S = 15.0


def circuits(spec: str) -> list[str]:
    """'all' = Thunderhill and every built training circuit; otherwise a comma list."""
    known = sorted(path.parent.name for path in (ROOT / "godot/tracks").glob("*/track.json"))
    if spec == "all":
        return [DEFAULT_TRACK, *(t for t in known if not held_out(t))]
    tracks = [t.strip() for t in spec.split(",") if t.strip()]
    if unknown := [t for t in tracks if t != DEFAULT_TRACK and t not in known]:
        raise SystemExit(f"unknown circuits {unknown}; available: {[DEFAULT_TRACK, *known]}")
    return tracks


class GodotSlots:
    """A machine-wide cap on the Godot workers Rev's rides run: one lock file per slot,
    held while an env is open (the kernel drops it if the process dies)."""

    def __init__(self, count: int = int(os.environ.get("REV_GODOT_SLOTS", "16"))):
        SLOTS.mkdir(parents=True, exist_ok=True)
        self.paths = [SLOTS / f"slot-{i:02d}.lock" for i in range(count)]

    def free(self) -> int:
        """Slots no process holds right now."""
        count = 0
        for path in self.paths:
            with path.open("w") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                fcntl.flock(stream, fcntl.LOCK_UN)
                count += 1
        return count

    @contextlib.contextmanager
    def slot(self):
        while True:
            for path in self.paths:
                stream = path.open("w")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                return
            time.sleep(1.0)


def crash_starts(directories, track: str) -> list[dict]:
    """Rolling starts FOCUS_LEAD_M before every offroad or fall on this circuit in these rides
    (rev_drive runs: eval/eval.jsonl or a collection's episodes.jsonl; rows without a track
    rode Thunderhill)."""
    geometry = Track(RoadTelemetry(track=track))
    starts = []
    for directory in directories:
        for name in ("eval/eval.jsonl", "episodes.jsonl"):
            path = Path(directory) / name
            if not path.exists():
                continue
            for line in path.open():
                row = json.loads(line)
                if row.get("track", DEFAULT_TRACK) != track or row["termination"] not in OFF_TRACK_TERMINATIONS:
                    continue
                station = (row["start_station_m"] + row["legal_progress_m"] - FOCUS_LEAD_M) % geometry.length
                index = int(np.searchsorted(geometry._s, station, side="right") - 1)
                speed = float(np.clip(0.8 * geometry.speed_limits[index], MIN_START_SPEED_M_S, MAX_INITIAL_SPEED_M_S))
                starts.append(dict(station=round(float(station), 2), speed=round(speed, 2)))
    return starts


class RevRider:
    """Asks a Rev server over one kept-alive HTTP(S) connection per thread: a fresh TLS
    handshake per control step cost more than the model (660 ms against 8 ms on Modal).

    Modal's web ingress drops about 1 request in 1,500 before it reaches the container (a
    probe build logged the id of every request kev.serve received: the hung ones never
    arrived, at any point of a connection's life), and the client then waits out its whole
    timeout. Answers take at most ~2 s even beside a busy SAC learner, so a request gets
    REQUEST_TIMEOUT_S and is then sent again on a fresh connection: a state's answer does
    not depend on how many times it is asked."""

    def __init__(self, url: str, decode: str, pedal_gain: float, off_track_fraction: float, named: bool):
        self.url = urllib.parse.urlsplit(url.rstrip("/") + "/v1/systemone")
        self.decode, self.pedal_gain, self.named = decode, pedal_gain, named
        self.off_track_fraction = off_track_fraction
        key = os.environ.get("REV_API_KEY")
        self.headers = {"content-type": "application/json", **({"authorization": f"Bearer {key}"} if key else {})}
        self.local = threading.local()

    def _connection(self, fresh=False):
        if fresh or getattr(self.local, "connection", None) is None:
            kind = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
            self.local.connection = kind(self.url.netloc, timeout=REQUEST_TIMEOUT_S)
        return self.local.connection

    def ask(self, obs, circuit: str, rng) -> dict:
        questions = QUESTIONS if rng.random() < self.off_track_fraction else RIDE_QUESTIONS
        # kev.serve answers to its fixed alias whatever checkpoint it loaded; the weights are Rev's.
        body = json.dumps({"model": "kev-latest", "state": render_state(obs, circuit if self.named else None),
                           "questions": questions}).encode()
        began = time.perf_counter()
        for attempt in range(5):
            try:
                connection = self._connection(fresh=attempt > 0)
                connection.request("POST", self.url.path, body, self.headers)
                response = connection.getresponse()
                payload = response.read()
                if response.status != 200:
                    raise RuntimeError(f"Rev server returned {response.status}: {payload[:200]!r}")
                answer = json.loads(payload)
                break
            except (OSError, http.client.HTTPException, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(attempt)
        answer["client_ms"] = (time.perf_counter() - began) * 1000.0
        return answer

    def level(self, probabilities: dict, levels: dict) -> float:
        if self.decode == "argmax":
            return levels[max(probabilities, key=probabilities.get)]
        return expected_level(probabilities, levels)


def ride(env: ThunderhillSACEnv, rider: RevRider, start: dict | None, log, rng, teacher=None, beta=0.0):
    """One episode. With a teacher, it rides the step with probability beta (Rev is still
    asked, so its answers are logged). Returns (summary, visited observations)."""
    obs, _ = env.reset(options={"start": start} if start else None)
    visited = []
    while True:
        visited.append(obs)
        response = rider.ask(obs, env.track_id, rng)
        answers = response["answers"]
        steer = rider.level(answers["steer"]["probabilities"], STEER_LEVELS)
        pedal = rider.level(answers["pedal"]["probabilities"], PEDAL_LEVELS)
        by_teacher = teacher is not None and rng.random() < beta
        if by_teacher:
            steer, pedal = (float(v) for v in teacher.controls(obs))
        # sac_env.controls scales the pedal by pedal_gain; send the raw value that lands on `pedal`.
        action = np.array([steer, pedal / rider.pedal_gain], dtype=np.float32)
        next_obs, _, terminated, truncated, info = env.step(action)
        # A worker restart truncates the episode with only its summary (sac_env.step).
        episode_id = info.get("episode_id") or info.get("episode_summary", {}).get("episode_id")
        log({"track": env.track_id, "episode_id": episode_id, "step": env._step_count,
             "station_m": round(float(env._observation["track"]["progress"]) * env.track.length, 1),
             "steer": round(steer, 4), "pedal": round(pedal, 4), "teacher": by_teacher,
             "steer_probabilities": answers["steer"]["probabilities"],
             "pedal_probabilities": answers["pedal"]["probabilities"],
             "off_track_soon": answers["off_track_soon"]["noul"] if "off_track_soon" in answers else None,
             "model_ms": response.get("latency_ms"), "client_ms": round(response["client_ms"], 1)})
        obs = next_obs
        if terminated or truncated:
            return info["episode_summary"], np.stack(visited)


def per_track(rows) -> dict:
    """track -> laps, lap times, progress and terminations of its rides."""
    out = {}
    for row in rows:
        entry = out.setdefault(row["track"], {"held_out": row["held_out"], "rides": 0, "laps": 0, "lap_times": [],
                                              "progress_m": [], "terminations": []})
        entry["rides"] += 1
        entry["progress_m"].append(round(row["legal_progress_m"]))
        entry["terminations"].append(row["termination"])
        if row["termination"] == "lap_completed":
            entry["laps"] += 1
            entry["lap_times"].append(round(row["lap_time_s"], 3))
    return dict(sorted(out.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--godot", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/rev")
    parser.add_argument("--rev-url", default="http://127.0.0.1:8019")
    parser.add_argument("--decode", choices=("mean", "argmax"), default="mean")
    parser.add_argument("--tracks", default=DEFAULT_TRACK, help="'all' or a comma list of training circuits")
    parser.add_argument("--heldout-tracks", default="", help=f"test tracks evaluated, never collected on "
                                                             f"(e.g. {HELD_OUT_TRACKS})")
    parser.add_argument("--starts", type=int, default=4, help="evaluation starts per training circuit")
    parser.add_argument("--heldout-starts", type=int, default=2, help="evaluation starts per held-out track")
    parser.add_argument("--no-circuit", action="store_true",
                        help="leave the circuit's name out of every state (generalization check)")
    parser.add_argument("--off-track-fraction", type=float, default=0.1,
                        help="share of steps that also ask off_track_soon")
    parser.add_argument("--horizon", type=float, default=240.0)
    parser.add_argument("--pedal-gain", type=float, default=1.25)
    parser.add_argument("--collect", type=int, default=0, help="DAgger: episodes per worker from random starts")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--track-weights", type=json.loads, default={},
                        help='JSON {track: weight}: each circuit\'s share of --collect episodes (default equal)')
    parser.add_argument("--teachers", type=Path, help="sac_specialists registry: the circuit's teacher rides "
                                                      "a --beta share of collect steps")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--focus-from", type=Path, nargs="*", default=[],
                        help="rev_drive runs whose offroad and fall spots --collect rides revisit")
    parser.add_argument("--focus-fraction", type=float, default=0.5,
                        help="share of --collect episodes that start before a --focus-from failure")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.tracks = circuits(args.tracks)
    args.heldout_tracks = circuits(args.heldout_tracks) if args.heldout_tracks else []
    if bad := [t for t in args.tracks if held_out(t)]:
        parser.error(f"{bad} are held-out test tracks: pass them with --heldout-tracks (evaluation only)")
    if args.collect and args.heldout_tracks:
        parser.error("held-out test tracks are never collected on")
    run_dir = args.runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1) + "\n")
    rider = RevRider(args.rev_url, args.decode, args.pedal_gain, args.off_track_fraction, not args.no_circuit)
    if args.collect:
        collect(args, run_dir, rider)
        return
    evaluate(args, run_dir, rider)


def evaluate(args, run_dir: Path, rider: RevRider):
    directory = run_dir / "eval"
    jobs = [(track, start, False) for track in args.tracks
            for start in evaluation_starts(args.starts, Track(RoadTelemetry(track=track)))]
    jobs += [(track, start, True) for track in args.heldout_tracks
             for start in evaluation_starts(args.heldout_starts, Track(RoadTelemetry(track=track)))]
    decisions, lock, slots = (run_dir / "decisions.jsonl").open("a"), threading.Lock(), GodotSlots()

    def log(row):
        with lock:
            decisions.write(json.dumps(row) + "\n")

    def job(index):
        track, start, test = jobs[index]
        with slots.slot():
            env = ThunderhillSACEnv(godot=args.godot, data_dir=directory, horizon_s=args.horizon, track=track,
                                    reward_line="progress", pedal_gain=args.pedal_gain, record_godot=True,
                                    policy_id=f"{args.run_name}-eval")
            try:
                row, _ = ride(env, rider, start, log, np.random.default_rng(args.seed * 1000 + index))
            finally:
                env.close()
        return dict(row, track=track, held_out=test)

    began = time.monotonic()
    try:
        with ThreadPoolExecutor(min(args.workers, len(jobs))) as pool:
            rows = list(pool.map(job, range(len(jobs))))
    finally:
        decisions.close()
    with (directory / "eval.jsonl").open("a") as stream:
        for row in rows:
            recordings = list(directory.glob(f"**/{row['episode_id']}.jsonl"))
            row.update(policy_step=0, checkpoint=args.rev_url,
                       recording=str(recordings[0].relative_to(run_dir)) if recordings else None)
            stream.write(json.dumps(row) + "\n")
    summary = per_track(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    steps = sum(row["steps"] for row in rows)
    elapsed = time.monotonic() - began
    for track, entry in summary.items():
        print(f"  {track:18s}{' (held out)' if entry['held_out'] else '':12s} {entry['laps']}/{entry['rides']} laps "
              f"{entry['lap_times']} progress {entry['progress_m']} {entry['terminations']}")
    trained = [r for r in rows if not r["held_out"]]
    print(f"REV EVAL {args.run_name}: {sum(r['termination'] == 'lap_completed' for r in trained)}/{len(trained)} laps "
          f"on {len(args.tracks)} circuits, held out "
          f"{sum(r['termination'] == 'lap_completed' for r in rows if r['held_out'])}/{len(rows) - len(trained)}; "
          f"{steps} steps in {elapsed:.0f} s ({steps / elapsed:.1f} steps/s)")


def allocate(total: int, weights: dict) -> dict:
    """total episodes split across circuits in proportion to their weights (largest remainders)."""
    shares = {track: total * w / sum(weights.values()) for track, w in weights.items()}
    counts = {track: int(share) for track, share in shares.items()}
    for track in sorted(shares, key=lambda t: counts[t] - shares[t])[: total - sum(counts.values())]:
        counts[track] += 1
    return counts


def collect(args, run_dir: Path, rider: RevRider):
    from sac_teacher import Teachers
    teachers = Teachers(args.teachers, pedal_gain=args.pedal_gain) if args.teachers else None
    if args.beta > 0 and teachers is None:
        raise SystemExit("--beta needs --teachers")
    weights = {track: float(args.track_weights.get(track, 1.0)) for track in args.tracks}
    episodes = allocate(args.collect * args.workers, weights)
    # One job per env: a circuit's episodes in runs of at most --collect, so a Godot rides several.
    jobs = [(track, min(args.collect, n - k)) for track, n in episodes.items() for k in range(0, n, args.collect)]
    focus = {track: crash_starts(args.focus_from, track) for track in args.tracks}
    print(f"{sum(episodes.values())} episodes on {sum(n > 0 for n in episodes.values())} circuits "
          f"{json.dumps({t: n for t, n in episodes.items() if n})}; failure spots to revisit "
          f"{json.dumps({t: len(s) for t, s in focus.items() if s})} from {len(args.focus_from)} runs", flush=True)
    if teachers:
        for track in args.tracks:  # loaded here, not by the worker threads
            teachers[track]
        print(f"teachers {json.dumps(teachers.describe(args.tracks))}", flush=True)
    decisions, lock, slots = (run_dir / "decisions.jsonl").open("a"), threading.Lock(), GodotSlots()

    def log(row):
        with lock:
            decisions.write(json.dumps(row) + "\n")

    def job(index):
        track, count = jobs[index]
        rng = np.random.default_rng(args.seed * 1000 + index)
        teacher = teachers[track] if teachers else None
        obs, episode, off, summaries = [], [], [], []
        with slots.slot():
            env = ThunderhillSACEnv(godot=args.godot, data_dir=run_dir / "collect", horizon_s=args.horizon,
                                    track=track, reward_line="progress", pedal_gain=args.pedal_gain,
                                    seed=args.seed * 1000 + index, policy_id=f"{args.run_name}-collect")
            try:
                for k in range(count):
                    spots = focus[track]
                    start = spots[int(rng.integers(len(spots)))] if spots and rng.random() < args.focus_fraction else None
                    summary, visited = ride(env, rider, start, log, rng, teacher, args.beta)
                    crashed = np.zeros(len(visited), dtype=bool)
                    if summary["termination"] in OFF_TRACK_TERMINATIONS:
                        crashed[-OFF_TRACK_HORIZON_STEPS:] = True
                    obs.append(visited)
                    episode += [k] * len(visited)
                    off.append(crashed)
                    summaries.append(dict(summary, track=track, focus=start is not None))
            finally:
                env.close()
        if obs:
            states = np.concatenate(obs).astype(np.float32)
            np.savez_compressed(run_dir / f"collect-{index:03d}.npz", obs=states, episode=np.array(episode),
                                off_track=np.concatenate(off), track=np.full(len(states), track))
        return summaries

    began = time.monotonic()
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            summaries = [s for rows in pool.map(job, range(len(jobs))) for s in rows]
    finally:
        decisions.close()
    with (run_dir / "episodes.jsonl").open("w") as stream:
        for row in summaries:
            stream.write(json.dumps(row) + "\n")
    steps = sum(row["steps"] for row in summaries)
    ends = {}
    for row in summaries:
        ends[row["termination"]] = ends.get(row["termination"], 0) + 1
    elapsed = time.monotonic() - began
    print(f"REV COLLECT {args.run_name}: {len(summaries)} episodes, {steps} states at beta {args.beta}, "
          f"ends {ends}, {elapsed:.0f} s ({steps / elapsed:.1f} steps/s)")


if __name__ == "__main__":
    main()
