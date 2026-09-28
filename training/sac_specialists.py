# /// script
# requires-python = ">=3.12,<3.13"
# ///
"""One specialist SAC policy per training circuit, trained in turn with
sac_async.py: the teachers that Rev distils.

  uv run --script training/sac_specialists.py --godot GODOT

Each specialist warm-starts from the general multi-track checkpoint (--init-from)
with an empty replay buffer and trains on its one circuit (--track) with focus
practice before recent failures:

  phase 1  --learning-rate (1e-4) until --phase1-steps or a plateau
  phase 2  --refine-learning-rate (3e-5) on the same run (--resume keeps the
           replay buffer) until --max-steps or a plateau

A plateau is --patience env steps without the best evaluation improving (more
laps, or a best lap --min-delta faster), once every start laps and at least
--min-steps have run. The first evaluation, at env step 0, rides the general
policy itself, so every circuit has a baseline under the same protocol; it
also remains a candidate, so a specialist is never worse than its start.

The kept checkpoint is the best evaluated one (most laps out of --eval-starts
fixed rolling starts, then the fastest lap), not the last. After each circuit
the registry (--registry, runs/sac/specialists.json) gains

  track -> {checkpoint, best_lap_s, mean_lap_s, laps, eval_starts, steps, general, run}

where general is the step-0 evaluation of the starting policy. --reuse adds
already-trained teachers (thunderhill-east) without training them.

Resumable: finished circuits are skipped and a half-done one resumes from its
latest checkpoint and replay buffer, in its phase (runs/sac/specialists/
specialist-TRACK/specialist.json). SIGINT/SIGTERM stops the running trainer
cleanly. Rollout shards are skipped (episodes, evaluations and checkpoints are
kept); after a circuit its replay buffer is deleted and the Godot recordings of
every evaluation but the best are gzip-compressed. Below --min-free-gb free disk
the trainer is stopped and this exits 0, so a Restart=on-failure unit stays down.
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
from lap_policy import DEFAULT_TRACK, ROOT, held_out  # noqa: E402

# The general policy's weakest circuits first: no laps, then unreliable laps.
FIRST = ("balaton-park", "red-bull-ring", "goiania", "cota", "assen", "aragon", "valencia")
GENERAL = ROOT / "runs/sac/sac-multitrack-1"


class Stopped(Exception):
    """A signal or the disk floor stopped the trainer; nothing is finished."""


def evaluations(run_dir: Path, starts: int) -> list[dict]:
    """Completed evaluations of a run, in order: one per checkpoint."""
    path = run_dir / "eval" / "eval.jsonl"
    if not path.exists():
        return []
    groups: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines(keepends=True):
        if line.endswith("\n"):  # the evaluator may be mid-write
            row = json.loads(line)
            groups.setdefault(row["checkpoint"], []).append(row)
    result = []
    for checkpoint, rows in groups.items():
        if len(rows) != starts:
            continue
        laps = [r["lap_time_s"] for r in rows if r["lap_time_s"] is not None]
        result.append(dict(checkpoint=checkpoint, steps=rows[0]["policy_step"], laps=len(laps),
                           best_lap_s=min(laps) if laps else None,
                           mean_lap_s=sum(laps) / len(laps) if laps else None, rows=rows))
    return result


def rank(evaluation):
    """Most laps, then the fastest lap."""
    return evaluation["laps"], -(evaluation["best_lap_s"] or math.inf)


def last_improvement(evals: list[dict], min_delta: float) -> int:
    """Env step of the last evaluation that beat the best before it by more laps
    or a best lap at least min_delta faster."""
    best, step = None, 0
    for e in evals:
        if best is None or e["laps"] > best["laps"] or (
                e["laps"] == best["laps"] and e["best_lap_s"] is not None
                and e["best_lap_s"] < best["best_lap_s"] - min_delta):
            best, step = e, e["steps"]
        elif rank(e) > rank(best):
            best = e
    return step


def summary(evaluation: dict | None) -> dict | None:
    if evaluation is None:
        return None
    return {k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in evaluation.items() if k != "rows"}


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
        own run's evaluation of that checkpoint."""
        registry = self.registry()
        for track, checkpoint in self.args.reuse.items():
            if track in registry:
                continue
            run_dir = checkpoint.parent.parent
            relative = str(checkpoint.relative_to(run_dir))
            rows = [json.loads(line) for line in (run_dir / "eval/eval.jsonl").open()]
            rows = [r for r in rows if r["checkpoint"] == relative]
            if not rows:
                raise SystemExit(f"--reuse {track}: {checkpoint} has no evaluation")
            evaluation = next(e for e in evaluations(run_dir, len(rows))
                              if e["checkpoint"] == relative)
            self.publish(track, dict(summary(evaluation), checkpoint=str(checkpoint.relative_to(ROOT)),
                                     eval_starts=len(rows), general=None,
                                     run=str(run_dir.relative_to(ROOT)), reused=True))

    # One circuit ---------------------------------------------------------------
    def run_dir(self, track):
        return self.args.runs_dir / f"specialist-{track}"

    def train(self, track: str):
        args, run_dir = self.args, self.run_dir(track)
        state_path = run_dir / "specialist.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(phase=1)
        while state["phase"] != "done":
            self.check_disk()
            if state["phase"] == 1:
                reason = self.run_trainer(track, args.learning_rate, args.phase1_steps, since=0)
                state = dict(phase=2, phase2_from=self.latest_steps(run_dir), phase1_end=reason)
            else:
                reason = self.run_trainer(track, args.refine_learning_rate, args.max_steps,
                                          since=state["phase2_from"])
                state.update(phase="done", phase2_end=reason)
            run_dir.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=1) + "\n")
        self.finish(track)

    def latest_steps(self, run_dir: Path) -> int:
        steps = [int(p.stem.split("_")[1]) for p in (run_dir / "checkpoints").glob("step_*.pt")]
        return max(steps, default=0)

    def run_trainer(self, track, learning_rate, total_steps, since) -> str:
        """sac_async.py on one circuit until total_steps ("budget") or a plateau
        counted from env step `since` ("plateau")."""
        args, run_dir = self.args, self.run_dir(track)
        if self.latest_steps(run_dir) >= total_steps:
            return "budget"
        command = [
            args.uv, "run", "--script", str(ROOT / "training/sac_async.py"),
            "--godot", args.godot, "--runs-dir", str(args.runs_dir),
            "--run-name", run_dir.name, "--resume", "--init-from", str(args.init_from),
            "--track", track, "--reward-line", "progress", "--pedal-gain", "1.25",
            "--workers", str(args.workers), "--buffer-size", str(args.buffer_size),
            "--learning-rate", str(learning_rate), "--total-steps", str(total_steps),
            "--focus-fraction", str(args.focus_fraction),
            "--eval-every", str(args.eval_every), "--eval-starts", str(args.eval_starts),
            "--eval-horizon", str(args.eval_horizon), "--no-rollout-shards",
            *(["--no-wandb"] if args.no_wandb else []),
        ]
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"TRAIN {track}: lr {learning_rate}, to {total_steps} env steps, from "
              f"{self.latest_steps(run_dir)} (log {run_dir / 'train.log'})", flush=True)
        with (run_dir / "train.log").open("a") as log:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        reason, reported = "budget", 0
        while self.child.poll() is None:
            time.sleep(15)
            if self.stop_requested:  # on_signal already stopped the trainer
                reason = "stopped"
                break
            if free_gb(run_dir) < args.min_free_gb:
                self.child.send_signal(signal.SIGINT)
                reason = "stopped"
                break
            evals = evaluations(run_dir, args.eval_starts)
            if len(evals) > reported:
                reported = len(evals)
                best = max(evals, key=rank)
                print(f"EVAL {track} step {evals[-1]['steps']}: {evals[-1]['laps']}/"
                      f"{args.eval_starts} laps, best {evals[-1]['best_lap_s']}; best so far "
                      f"{best['laps']} laps {best['best_lap_s']} at {best['steps']}", flush=True)
            if self.plateaued(evals, since):
                self.child.send_signal(signal.SIGINT)
                reason = "plateau"
                break
        code = self.child.wait()
        self.child = None
        if code != 0:
            raise SystemExit(f"sac_async.py on {track} exited {code}; see {run_dir / 'train.log'}")
        if reason == "stopped" or self.stop_requested:
            self.check_disk()
            raise Stopped(f"stopped during {track} at {self.latest_steps(run_dir)} env steps")
        print(f"TRAIN {track}: {reason} at {self.latest_steps(run_dir)} env steps", flush=True)
        return reason

    def plateaued(self, evals, since) -> bool:
        args = self.args
        if not evals or evals[-1]["steps"] < max(args.min_steps, since + args.patience):
            return False
        if max(evals, key=rank)["laps"] < args.eval_starts:
            return False  # not yet reliable: use the whole budget
        return evals[-1]["steps"] - max(last_improvement(evals, args.min_delta), since) \
            >= args.patience

    def finish(self, track):
        """Publish the best evaluated checkpoint, then free the circuit's disk."""
        run_dir = self.run_dir(track)
        evals = evaluations(run_dir, self.args.eval_starts)
        best = max(evals, key=rank)
        general = next((e for e in evals if e["steps"] == 0), None)
        entry = dict(summary(best), checkpoint=str((run_dir / best["checkpoint"]).relative_to(ROOT)),
                     eval_starts=self.args.eval_starts, trained_steps=self.latest_steps(run_dir),
                     general=summary(general), run=str(run_dir.relative_to(ROOT)),
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
        for track in args.tracks:
            if track in self.registry():
                print(f"SKIP {track}: in {args.registry.relative_to(ROOT)}", flush=True)
                continue
            self.train(track)
        print("ALL SPECIALISTS DONE", flush=True)


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
    parser.add_argument("--phase1-steps", type=int, default=4_000_000)
    parser.add_argument("--max-steps", type=int, default=7_000_000)
    parser.add_argument("--min-steps", type=int, default=2_000_000)
    parser.add_argument("--patience", type=int, default=1_500_000,
                        help="env steps without improvement that count as a plateau")
    parser.add_argument("--min-delta", type=float, default=0.2,
                        help="seconds a best lap must improve by to count")
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
