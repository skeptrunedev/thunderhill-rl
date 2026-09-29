# /// script
# requires-python = ">=3.12,<3.13"
# ///
"""One specialist SAC policy per training circuit, trained in turn with
sac_async.py: the teachers that Rev distils.

  uv run --script training/sac_specialists.py --godot GODOT

Each specialist warm-starts from the general multi-track checkpoint (--init-from)
with an empty replay buffer and trains on its one circuit (--track) with focus
practice before recent failures and before the apexes its latest evaluation
missed (--focus-file):

  learn   --learning-rate (1e-4) until an evaluation laps from every start
  refine  --refine-learning-rate (3e-5) on the same run (--resume keeps the
          replay buffer) until the circuit meets the bar

There is no step budget. A circuit is done when its best evaluation (most laps
out of --eval-starts fixed rolling starts, then most apexes, then the fastest
lap, then the furthest mean progress) meets the bar the first Thunderhill
specialist set:

  reliability  every start laps
  apexes       on the best lap, every corner of at least 30 degrees is apexed
               (tools/apex_report.py)
  full gas     the median throttle while upright and off the brakes is at least
               --min-throttle
  speed        the best lap improved by less than --plateau-fraction over the
               last --plateau-evals evaluations

A circuit whose best evaluation has not improved for --stuck-steps env steps is
marked stuck, left out of the registry and reported; the next circuit starts.

The first evaluation, at env step 0, rides the general policy itself, so every
circuit has a baseline under the same protocol. After each circuit the registry
(--registry, runs/sac/specialists.json) gains

  track -> {checkpoint, best_lap_s, mean_lap_s, laps, eval_starts, apexes_hit,
            apexes_total, upright_median_throttle, steps, general, run}

where general is the step-0 evaluation of the starting policy. --reuse adds
already-trained teachers (thunderhill-east) without training them.

Resumable: finished and stuck circuits are skipped and a half-done one resumes
from its latest checkpoint and replay buffer, in its phase (runs/sac/
specialists/specialist-TRACK/specialist.json). SIGINT/SIGTERM stops the running
trainer cleanly. Rollout shards are skipped (episodes, evaluations and
checkpoints are kept); after a circuit its replay buffer is deleted and the
Godot recordings of every evaluation but the best are gzip-compressed. Below
--min-free-gb free disk the trainer is stopped and this exits 0, so a
Restart=on-failure unit stays down.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from apex_report import report as apex_report  # noqa: E402
from lap_policy import DEFAULT_TRACK, ROOT, held_out  # noqa: E402

# The general policy's weakest circuits first: no laps, then unreliable laps.
FIRST = ("balaton-park", "red-bull-ring", "goiania", "cota", "assen", "aragon", "valencia")
GENERAL = ROOT / "runs/sac/sac-multitrack-1"
APEX_FIELDS = ("apexes_hit", "apexes_total", "missed_apex_stations_m", "upright_median_throttle")


class Stopped(Exception):
    """A signal or the disk floor stopped the trainer; nothing is finished."""


def recording_path(run_dir: Path, recording: str) -> Path:
    path = run_dir / recording
    return path if path.exists() else path.with_name(path.name + ".gz")


def evaluations(run_dir: Path, starts: int, track: str, only: str | None = None) -> list[dict]:
    """Completed evaluations of a run (or of its checkpoint `only`), in order: one
    per checkpoint, with the apex report of its best lap (cached in
    run_dir/apex.jsonl, except for `only`)."""
    path = run_dir / "eval" / "eval.jsonl"
    if not path.exists():
        return []
    groups: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines(keepends=True):
        if line.endswith("\n"):  # the evaluator may be mid-write
            row = json.loads(line)
            groups.setdefault(row["checkpoint"], []).append(row)
    cache_path = run_dir / "apex.jsonl"
    cache = ({row["checkpoint"]: row for row in map(json.loads, cache_path.open())}
             if cache_path.exists() else {})
    result = []
    for checkpoint, rows in groups.items():
        if len(rows) != starts or only not in (None, checkpoint):
            continue
        laps = [r for r in rows if r["lap_time_s"] is not None]
        evaluation = dict(checkpoint=checkpoint, steps=rows[0]["policy_step"], laps=len(laps),
                          best_lap_s=min(r["lap_time_s"] for r in laps) if laps else None,
                          mean_lap_s=sum(r["lap_time_s"] for r in laps) / len(laps) if laps else None,
                          progress_m=sum(r["legal_progress_m"] for r in rows) / len(rows),
                          **dict.fromkeys(APEX_FIELDS), rows=rows)
        best_lap = min(laps, key=lambda r: r["lap_time_s"]) if laps else None
        if best_lap and best_lap.get("recording"):
            if checkpoint not in cache:
                full = apex_report(track, recording_path(run_dir, best_lap["recording"]))
                cache[checkpoint] = dict({k: full[k] for k in APEX_FIELDS}, checkpoint=checkpoint)
                if only is None:
                    with cache_path.open("a") as stream:
                        stream.write(json.dumps(cache[checkpoint]) + "\n")
            evaluation.update({k: cache[checkpoint][k] for k in APEX_FIELDS})
        result.append(evaluation)
    return result


def rank(evaluation):
    """Most laps, then most apexes, then the fastest lap; between lapless
    evaluations, the furthest mean legal progress."""
    return (evaluation["laps"], evaluation["apexes_hit"] or 0,
            -(evaluation["best_lap_s"] or math.inf), evaluation["progress_m"])


def improved(new, old, min_delta):
    """More laps or apexes, a best lap min_delta seconds faster, or (lapless)
    20 m more mean progress."""
    if (new["laps"], new["apexes_hit"] or 0) != (old["laps"], old["apexes_hit"] or 0):
        return (new["laps"], new["apexes_hit"] or 0) > (old["laps"], old["apexes_hit"] or 0)
    if new["best_lap_s"] is not None:
        return old["best_lap_s"] is None or new["best_lap_s"] < old["best_lap_s"] - min_delta
    return old["best_lap_s"] is None and new["progress_m"] > old["progress_m"] + 20.0


def last_improvement(evals: list[dict], min_delta: float) -> int:
    """Env step of the last evaluation that improved on the best before it."""
    best, step = None, 0
    for e in evals:
        if best is None or improved(e, best, min_delta):
            step = e["steps"]
        if best is None or rank(e) > rank(best):
            best = e
    return step


def summary(evaluation: dict | None) -> dict | None:
    if evaluation is None:
        return None
    return {k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in evaluation.items() if k != "rows"}


def describe(e: dict) -> str:
    return (f"{e['laps']} laps, best {e['best_lap_s'] and round(e['best_lap_s'], 2)} s, apexes "
            f"{e['apexes_hit']}/{e['apexes_total']} (missed {e['missed_apex_stations_m']}), "
            f"throttle {e['upright_median_throttle']}, progress {e['progress_m']:.0f} m "
            f"at {e['steps']}")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


class Specialists:
    def __init__(self, args):
        self.args = args
        self.child: subprocess.Popen | None = None
        self.stop_requested = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.on_signal)

    def on_signal(self, *_):
        self.stop_requested = True
        if self.child and self.child.poll() is None:
            self.child.send_signal(signal.SIGINT)

    # Registry ----------------------------------------------------------------
    def registry(self) -> dict:
        path = self.args.registry
        return json.loads(path.read_text()) if path.exists() else {}

    def publish(self, track: str, entry: dict):
        registry = self.registry()
        registry[track] = entry
        tmp = self.args.registry.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(sorted(registry.items())), indent=1) + "\n")
        tmp.replace(self.args.registry)
        print(f"REGISTRY {track}: {json.dumps(entry)}", flush=True)

    def publish_reused(self):
        """--reuse TRACK=CHECKPOINT: a teacher trained elsewhere, measured by its
        own run's evaluation of that checkpoint, republished at every start."""
        for track, checkpoint in self.args.reuse.items():
            run_dir = checkpoint.parent.parent
            relative = str(checkpoint.relative_to(run_dir))
            rows = [json.loads(line) for line in (run_dir / "eval/eval.jsonl").open()]
            rows = [r for r in rows if r["checkpoint"] == relative]
            if not rows:
                raise SystemExit(f"--reuse {track}: {checkpoint} has no evaluation")
            evaluation, = evaluations(run_dir, len(rows), track, only=relative)
            self.publish(track, dict(summary(evaluation), checkpoint=str(checkpoint.relative_to(ROOT)),
                                     eval_starts=len(rows), meets_bar=self.meets_bar(evaluation),
                                     general=None, run=str(run_dir.relative_to(ROOT)), reused=True))

    # The bar -------------------------------------------------------------------
    def meets_bar(self, e: dict, starts: int | None = None) -> bool:
        """Every start laps, every required apex hit, full throttle when upright."""
        return (e["laps"] == (starts or len(e["rows"])) and bool(e["apexes_total"])
                and e["apexes_hit"] == e["apexes_total"]
                and (e["upright_median_throttle"] or 0.0) >= self.args.min_throttle)

    def lap_plateaued(self, evals: list[dict]) -> bool:
        """The best lap so far improved by less than --plateau-fraction over the
        last --plateau-evals evaluations."""
        best, running = math.inf, []
        for e in evals:
            best = min(best, e["best_lap_s"] or math.inf)
            running.append(best)
        k = self.args.plateau_evals
        return (len(running) > k and math.isfinite(running[-1 - k])
                and running[-1 - k] - running[-1] < self.args.plateau_fraction * running[-1 - k])

    # One circuit ---------------------------------------------------------------
    def run_dir(self, track):
        return self.args.runs_dir / f"specialist-{track}"

    def train(self, track: str) -> bool:
        """Train one circuit to the bar; False when it is (or was) stuck."""
        args, run_dir = self.args, self.run_dir(track)
        state_path = run_dir / "specialist.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(phase="learn")
        while state["phase"] in ("learn", "refine"):
            self.check_disk()
            learn = state["phase"] == "learn"
            outcome = self.run_trainer(
                track, args.learning_rate if learn else args.refine_learning_rate, learn)
            state.update({"phase": {"reliable": "refine"}.get(outcome, outcome),
                          f"{state['phase']}_end": outcome,
                          f"{state['phase']}_end_step": self.latest_steps(run_dir)})
            run_dir.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=1) + "\n")
        if state["phase"] == "stuck":
            print(f"STUCK {track} (skipped): {state_path}", flush=True)
            return False
        self.finish(track)
        return True

    def latest_steps(self, run_dir: Path) -> int:
        steps = [int(p.stem.split("_")[1]) for p in (run_dir / "checkpoints").glob("step_*.pt")]
        return max(steps, default=0)

    def outcome(self, evals: list[dict], learn: bool) -> str | None:
        """reliable (learn phase: an evaluation lapped from every start), done
        (the bar is met), stuck, or None to keep training."""
        args = self.args
        if not evals:
            return None
        best = max(evals, key=rank)
        if learn and best["laps"] == args.eval_starts:
            return "reliable"
        if not learn and self.meets_bar(best, args.eval_starts) and self.lap_plateaued(evals):
            return "done"
        if evals[-1]["steps"] - last_improvement(evals, args.min_delta) >= args.stuck_steps:
            return "stuck"
        return None

    def run_trainer(self, track, learning_rate, learn) -> str:
        """sac_async.py on one circuit until outcome() decides; after every
        evaluation the apexes it missed go to the actors' --focus-file."""
        args, run_dir = self.args, self.run_dir(track)
        decided = self.outcome(evaluations(run_dir, args.eval_starts, track), learn)
        if decided:
            return decided
        focus = run_dir / "focus.json"
        command = [
            args.uv, "run", "--script", str(ROOT / "training/sac_async.py"),
            "--godot", args.godot, "--runs-dir", str(args.runs_dir),
            "--run-name", run_dir.name, "--resume", "--init-from", str(args.init_from),
            "--track", track, "--reward-line", "progress", "--pedal-gain", "1.25",
            "--workers", str(args.workers), "--buffer-size", str(args.buffer_size),
            "--learning-rate", str(learning_rate), "--total-steps", str(10**12),
            "--focus-fraction", str(args.focus_fraction), "--focus-file", str(focus),
            "--eval-every", str(args.eval_every), "--eval-starts", str(args.eval_starts),
            "--eval-horizon", str(args.eval_horizon), "--no-rollout-shards",
            *(["--no-wandb"] if args.no_wandb else []),
        ]
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"TRAIN {track}: {'learn' if learn else 'refine'} at lr {learning_rate} from "
              f"{self.latest_steps(run_dir)} env steps (log {run_dir / 'train.log'})", flush=True)
        with (run_dir / "train.log").open("a") as log:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        decided, reported = None, None
        while self.child.poll() is None:
            time.sleep(15)
            if self.stop_requested:  # on_signal already stopped the trainer
                break
            if free_gb(run_dir) < args.min_free_gb:
                self.child.send_signal(signal.SIGINT)
                break
            evals = evaluations(run_dir, args.eval_starts, track)
            if not evals or evals[-1]["checkpoint"] == reported:
                continue
            reported, latest = evals[-1]["checkpoint"], evals[-1]
            tmp = focus.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(stations=latest["missed_apex_stations_m"] or [],
                                           checkpoint=latest["checkpoint"])) + "\n")
            tmp.replace(focus)
            print(f"EVAL {track}: {describe(latest)}; best so far "
                  f"{describe(max(evals, key=rank))}", flush=True)
            decided = self.outcome(evals, learn)
            if decided:
                self.child.send_signal(signal.SIGINT)
                break
        code = self.child.wait()
        self.child = None
        if code != 0:
            raise SystemExit(f"sac_async.py on {track} exited {code}; see {run_dir / 'train.log'}")
        if decided is None:
            self.check_disk()
            raise Stopped(f"stopped during {track} at {self.latest_steps(run_dir)} env steps")
        print(f"TRAIN {track}: {decided} at {self.latest_steps(run_dir)} env steps", flush=True)
        return decided

    def finish(self, track):
        """Publish the best evaluated checkpoint, then free the circuit's disk."""
        run_dir = self.run_dir(track)
        evals = evaluations(run_dir, self.args.eval_starts, track)
        best = max(evals, key=rank)
        general = next((e for e in evals if e["steps"] == 0), None)
        entry = dict(summary(best), checkpoint=str((run_dir / best["checkpoint"]).relative_to(ROOT)),
                     eval_starts=self.args.eval_starts, meets_bar=True,
                     trained_steps=self.latest_steps(run_dir), general=summary(general),
                     run=str(run_dir.relative_to(ROOT)),
                     finished=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"))
        self.publish(track, entry)
        # Recordings of the best evaluation stay renderable; the rest are kept gzipped.
        keep = {run_dir / r["recording"] for r in best["rows"] if r.get("recording")}
        recordings = [p for p in (run_dir / "eval").glob("worker-*/data/**/runs/*/*.jsonl")
                      if p not in keep]
        for start in range(0, len(recordings), 200):
            subprocess.run(["gzip", "-f", *map(str, recordings[start:start + 200])], check=True)
        (run_dir / "replay_buffer.npz").unlink(missing_ok=True)
        print(f"DONE {track}: gzipped {len(recordings)} recordings, deleted the replay buffer; "
              f"{free_gb(run_dir):.1f} GB free", flush=True)

    def check_disk(self):
        free = free_gb(self.args.runs_dir)
        if free < self.args.min_free_gb:
            print(f"DISK FLOOR: {free:.1f} GB free < --min-free-gb {self.args.min_free_gb}; "
                  "stopping", flush=True)
            raise Stopped("disk floor")

    def run(self):
        args = self.args
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        self.publish_reused()
        stuck = []
        for track in args.tracks:
            if track in self.registry():
                print(f"SKIP {track}: in {args.registry.relative_to(ROOT)}", flush=True)
                continue
            if not self.train(track):
                stuck.append(track)
        print(f"ALL SPECIALISTS DONE; stuck: {stuck or 'none'}", flush=True)


def training_order(listing: Path) -> list[str]:
    """FIRST, then the general run's other circuits in its order; never a held-out one."""
    tracks = json.loads(listing.read_text())
    order = [t for t in FIRST if t in tracks] + [t for t in tracks if t not in FIRST]
    return [t for t in order if not held_out(t)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--uv", default=shutil.which("uv") or "uv")
    parser.add_argument("--init-from", type=Path,
                        default=GENERAL / "checkpoints/step_40672729.pt")
    parser.add_argument("--tracks", help="comma list in training order; default: "
                        f"{', '.join(FIRST)}, then the rest of {GENERAL.name}/tracks.json")
    parser.add_argument("--reuse", default=f"{DEFAULT_TRACK}=runs/sac/sac-v7-lr1e4-1/checkpoints/"
                        "step_24000071.pt", help="comma list of TRACK=CHECKPOINT teachers "
                        "published as they are, not trained")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/sac/specialists")
    parser.add_argument("--registry", type=Path, default=ROOT / "runs/sac/specialists.json")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--buffer-size", type=int, default=6_000_000)
    parser.add_argument("--focus-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--refine-learning-rate", type=float, default=3e-5)
    parser.add_argument("--min-delta", type=float, default=0.2,
                        help="seconds a best lap must improve by to count as progress")
    parser.add_argument("--stuck-steps", type=int, default=10_000_000,
                        help="env steps without progress after which a circuit is stuck")
    parser.add_argument("--plateau-evals", type=int, default=3)
    parser.add_argument("--plateau-fraction", type=float, default=0.003,
                        help="the best lap must improve by less than this over --plateau-evals")
    parser.add_argument("--min-throttle", type=float, default=0.95,
                        help="median throttle while upright and off the brakes")
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--eval-starts", type=int, default=6)
    parser.add_argument("--eval-horizon", type=float, default=300.0)
    parser.add_argument("--min-free-gb", type=float, default=30.0,
                        help="stop below this much free disk (the final replay-buffer save "
                        "writes ~3 GB more)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.init_from = args.init_from.resolve()
    args.runs_dir, args.registry = args.runs_dir.resolve(), args.registry.resolve()
    args.reuse = {track: (ROOT / path).resolve() for track, path in
                  (item.split("=", 1) for item in args.reuse.split(",") if item)}
    args.tracks = ([t.strip() for t in args.tracks.split(",") if t.strip()] if args.tracks
                   else training_order(GENERAL / "tracks.json"))
    for track in args.tracks:
        if held_out(track):
            parser.error(f"{track} is a held-out test track; it is never trained on")
    args.tracks = [t for t in args.tracks if t not in args.reuse]
    print(f"Specialists in order: {args.tracks}", flush=True)
    try:
        Specialists(args).run()
    except Stopped as stopped:
        print(f"STOPPED: {stopped}", flush=True)


if __name__ == "__main__":
    main()
