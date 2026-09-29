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
  speed        at least --min-total-steps in all and --min-refine-steps at the
               refine lr, and the best lap improved by less than
               --plateau-fraction over the last --plateau-steps env steps

A circuit whose best evaluation has not improved for --stuck-steps env steps is
marked stuck, left out of the registry and reported; the next circuit starts.

The first evaluation, at env step 0, rides the general policy itself, so every
circuit has a baseline under the same protocol. After each circuit the registry
(--registry, runs/sac/specialists.json) gains

  track -> {checkpoint, best_lap_s, mean_lap_s, laps, eval_starts, apexes_hit,
            apexes_total, missed_apex_stations_m, upright_median_throttle,
            full_throttle_share, meets_bar, steps, general, run}

where general is the step-0 evaluation of the starting policy. --reuse adds
already-trained teachers (thunderhill-east) without training them.

Resumable: finished and stuck circuits are skipped and a half-done one resumes
from its latest checkpoint and replay buffer, in its phase (runs/sac/
specialists/specialist-TRACK/specialist.json). SIGINT/SIGTERM, or a file named
STOP in the circuit's run dir (how a Modal lane is stopped), stops the running
trainer cleanly, saving its checkpoint and replay buffer. Rollout shards are skipped (episodes, evaluations and
checkpoints are kept); after a circuit its replay buffer is deleted and the
Godot recordings of every evaluation but the best are gzip-compressed. Below
--min-free-gb free disk the trainer is stopped and this exits 0, so a
Restart=on-failure unit stays down.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
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
from apex_report import VERSION as APEX_VERSION, report as apex_report  # noqa: E402
from lap_policy import DEFAULT_TRACK, ROOT, held_out  # noqa: E402

# Circuits the general policy already laps reliably come first (Rev needs teachers
# soon), then those it laps unreliably, then those it never lapped.
LATER = ("aragon", "assen", "valencia", "cota", "goiania", "misano", "buriram")
LAST = ("balaton-park", "red-bull-ring")
GENERAL = ROOT / "runs/sac/sac-multitrack-1"
APEX_FIELDS = ("apexes_hit", "apexes_total", "missed_apex_stations_m", "upright_median_throttle",
               "full_throttle_share", "corner_source", "auto_apexes_hit", "auto_apexes_total")


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
    # Reports from another apex_report version are redone.
    cache = ({row["checkpoint"]: row for row in map(json.loads, cache_path.open())
              if row.get("version") == APEX_VERSION} if cache_path.exists() else {})
    result = []
    for checkpoint, rows in groups.items():
        # A checkpoint evaluated twice (e.g. step_0.pt by two starts of the same run)
        # keeps its latest evaluation.
        if not rows or len(rows) % starts or only not in (None, checkpoint):
            continue
        rows = rows[-starts:]
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
                cache[checkpoint] = dict({k: full[k] for k in APEX_FIELDS}, checkpoint=checkpoint,
                                         version=APEX_VERSION)
                if only is None:
                    with cache_path.open("a") as stream:
                        stream.write(json.dumps(cache[checkpoint]) + "\n")
            evaluation.update({k: cache[checkpoint][k] for k in APEX_FIELDS})
        result.append(evaluation)
    return result


def full_bar(evaluation) -> bool:
    """Every start lapped, every required apex hit, full throttle when upright."""
    return (bool(evaluation.get("apexes_total"))
            and evaluation["laps"] == len(evaluation.get("rows") or [None] * 6)
            and evaluation.get("apexes_hit") == evaluation["apexes_total"]
            and (evaluation.get("upright_median_throttle") or 0.0) >= MIN_THROTTLE)


def rank(evaluation):
    """Evaluations meeting the whole bar first (a faster one short of it, e.g. on
    throttle, must never shadow one that meets it); then most laps, most apexes,
    the fastest lap; between lapless evaluations, the furthest mean progress."""
    return (full_bar(evaluation), evaluation["laps"], evaluation["apexes_hit"] or 0,
            -(evaluation["best_lap_s"] or math.inf), evaluation["progress_m"])


def improved(new, old, min_delta):
    """Reaching the whole bar, more laps or apexes, a best lap min_delta seconds
    faster, or (lapless) 20 m more mean progress."""
    if full_bar(new) != full_bar(old):
        return full_bar(new)
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


MIN_THROTTLE = 0.95


def meets_bar(e: dict, starts: int, min_throttle: float = MIN_THROTTLE) -> bool:
    """Every start laps, every required apex hit, full throttle when upright."""
    return (e["laps"] == starts and bool(e["apexes_total"])
            and e["apexes_hit"] == e["apexes_total"]
            and (e["upright_median_throttle"] or 0.0) >= min_throttle)


def publish_entry(registry: Path, track: str, entry: dict):
    """Set one circuit's registry entry; the local unit and the Modal sync both
    write the registry, so under an exclusive lock."""
    with registry.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(registry.read_text()) if registry.exists() else {}
        current[track] = entry
        tmp = registry.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(sorted(current.items())), indent=1) + "\n")
        tmp.replace(registry)


# A shaped lane may publish a lap up to this much slower than its fastest lap before
# shaping: those laps often skipped apexes, and a line through every apex can cost
# a few tenths.
CEILING_TOLERANCE = 0.005


def lap_ceiling(shaping: dict) -> float:
    return shaping["fastest_lap_s"] * (1 + CEILING_TOLERANCE)


def ceiling_record(shaping: dict | None) -> dict | None:
    """The shaped-lane lap ceiling as the registry records it."""
    if not shaping:
        return None
    return dict(fastest_pre_shaping_lap_s=round(shaping["fastest_lap_s"], 3),
                tolerance=CEILING_TOLERANCE, ceiling_s=round(lap_ceiling(shaping), 3))


def shaped_result_ok(evaluation: dict, shaping: dict | None) -> bool:
    """A lane under reward shaping may only publish an evaluation of the shaped run
    (after shaping started) no slower than lap_ceiling() of its fastest lap before."""
    return shaping is None or (evaluation["steps"] > shaping["at_step"]
                               and evaluation["best_lap_s"] is not None
                               and evaluation["best_lap_s"] <= lap_ceiling(shaping))


def publish_checked(registry: Path, track: str, run_dir: Path, lane_entry: dict, starts: int,
                    source: str) -> tuple[bool, dict]:
    """A lane's published result, measured again here at its checkpoint (the
    recordings of that evaluation must be in run_dir, with its specialist.json)
    and published to the registry only if it meets the bar and, for a shaped
    lane, comes from the shaped run at least as fast as before."""
    checkpoint = Path(lane_entry["checkpoint"]).relative_to(run_dir.relative_to(ROOT)).as_posix()
    evaluation, = evaluations(run_dir, starts, track, only=checkpoint)
    state_path = run_dir / "specialist.json"
    shaping = (json.loads(state_path.read_text()).get("shaping_from")
               if state_path.exists() else None)
    entry = dict(lane_entry, **summary(evaluation))
    entry.update(checkpoint=lane_entry["checkpoint"], synced_from=source,
                 lap_ceiling=ceiling_record(shaping),
                 meets_bar=meets_bar(evaluation, starts) and shaped_result_ok(evaluation, shaping))
    if entry["meets_bar"]:
        publish_entry(registry, track, entry)
    return entry["meets_bar"], entry


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
        publish_entry(self.args.registry, track, entry)
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
        return meets_bar(e, starts or len(e["rows"]), self.args.min_throttle)

    def lap_plateaued(self, evals: list[dict], refine_from: int) -> bool:
        """Refined long enough, and the best lap so far improved by less than
        --plateau-fraction over the last --plateau-steps env steps."""
        args, latest = self.args, evals[-1]["steps"]
        if latest < args.min_total_steps or latest - refine_from < args.min_refine_steps:
            return False
        then = min((e["best_lap_s"] or math.inf for e in evals
                    if e["steps"] <= latest - args.plateau_steps), default=math.inf)
        now = min(e["best_lap_s"] or math.inf for e in evals)
        return math.isfinite(then) and then - now < args.plateau_fraction * then

    # One circuit ---------------------------------------------------------------
    def run_dir(self, track):
        return self.args.runs_dir / f"specialist-{track}"

    def train(self, track: str) -> bool:
        """Train one circuit to the bar; False when it is (or was) stuck."""
        args, run_dir = self.args, self.run_dir(track)
        state_path = run_dir / "specialist.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(phase="learn")
        self.resume_weights = None
        if args.restart_from_best and "shaping_from" not in state:
            # Shaping starts from the run's fastest evaluation that lapped from every
            # start, the lap a shaped result must match; shaping adds the apexes.
            best = min((e for e in evaluations(run_dir, args.eval_starts, track)
                        if e["laps"] == args.eval_starts), key=lambda e: e["best_lap_s"])
            state.update(phase="refine", shaping_from=dict(
                checkpoint=best["checkpoint"], best_lap_s=best["best_lap_s"],
                fastest_lap_s=best["best_lap_s"],
                apexes=f"{best['apexes_hit']}/{best['apexes_total']}",
                missed=best["missed_apex_stations_m"], at_step=self.latest_steps(run_dir),
                apex_bonus_m=args.apex_bonus_m, throttle_bonus_m=args.throttle_bonus_m))
            self.resume_weights = run_dir / best["checkpoint"]
            state_path.write_text(json.dumps(state, indent=1) + "\n")
            print(f"SHAPING {track}: from {best['checkpoint']} ({describe(best)})", flush=True)
        shaping = state.get("shaping_from")
        if shaping and "fastest_lap_s" not in shaping:
            # The ceiling is the run's fastest lap before shaping among evaluations
            # lapping from every start, whatever their apexes.
            shaping["fastest_lap_s"] = min(
                e["best_lap_s"] for e in evaluations(run_dir, args.eval_starts, track)
                if e["laps"] == args.eval_starts and e["steps"] <= shaping["at_step"])
            state_path.write_text(json.dumps(state, indent=1) + "\n")
        # A changed recipe restarts the stuck clock: its evaluations are a new attempt.
        recipe = json.dumps({k: getattr(args, k) for k in (
            "apex_bonus_m", "apex_bonus_dense", "apex_bonus_stations", "apex_bonus_floor",
            "throttle_bonus_m",
            "max_start_speed", "focus_lead", "focus_lead_min", "focus_curriculum",
            "focus_stations")}, sort_keys=True)
        if state.get("recipe") != recipe:
            state.update(recipe=recipe, recipe_from=self.latest_steps(run_dir))
            if state["phase"] == "stuck":
                state["phase"] = "refine"
            run_dir.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=1) + "\n")
        self.state = state
        if state["phase"] == "done" and self.outcome(
                evaluations(run_dir, args.eval_starts, track), False,
                state.get("learn_end_step", 0)) != "done":
            print(f"REOPEN {track}: done under an earlier bar, refining on", flush=True)
            state["phase"] = "refine"
        while state["phase"] in ("learn", "refine"):
            self.check_disk()
            learn = state["phase"] == "learn"
            outcome = self.run_trainer(
                track, args.learning_rate if learn else args.refine_learning_rate, learn,
                state.get("learn_end_step", 0))
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

    def outcome(self, evals: list[dict], learn: bool, refine_from: int) -> str | None:
        """reliable (learn phase: an evaluation lapped from every start), done
        (the bar is met), stuck, or None to keep training."""
        args = self.args
        if not evals:
            return None
        best = max(evals, key=rank)
        if learn and best["laps"] == args.eval_starts:
            return "reliable"
        shaping = getattr(self, "state", {}).get("shaping_from")
        if shaping:
            # Only the shaped run counts, and it needs its own refine window.
            refine_from = max(refine_from, shaping["at_step"])
            shaped = [e for e in evals if shaped_result_ok(e, shaping)]
            best = max(shaped, key=rank) if shaped else None
        if (not learn and best and self.meets_bar(best, args.eval_starts)
                and self.fast_enough(evals, best) and self.lap_plateaued(evals, refine_from)):
            return "done"
        start = max(shaping["at_step"] if shaping else 0, self.state.get("recipe_from", 0))
        since = [e for e in evals if e["steps"] > start] if start else evals
        if since and since[-1]["steps"] - max(last_improvement(since, args.min_delta), start) \
                >= args.stuck_steps:
            return "stuck"
        return None

    def fast_enough(self, evals: list[dict], best: dict) -> bool:
        """Faster than the general policy's step-0 evaluation (where it lapped) and,
        once shaping started, no slower than the run's best lap before it."""
        general = next((e for e in evals if e["steps"] == 0), None)
        if general is None:
            return False  # no baseline measured: never publish unchecked
        if general["best_lap_s"] is not None and best["best_lap_s"] >= general["best_lap_s"]:
            return False
        shaping = getattr(self, "state", {}).get("shaping_from")
        return shaping is None or best["best_lap_s"] <= lap_ceiling(shaping)

    def run_trainer(self, track, learning_rate, learn, refine_from) -> str:
        """sac_async.py on one circuit until outcome() decides; after every
        evaluation the apexes it missed go to the actors' --focus-file."""
        args, run_dir = self.args, self.run_dir(track)
        decided = self.outcome(evaluations(run_dir, args.eval_starts, track), learn, refine_from)
        if decided:
            return decided
        if (run_dir / "checkpoints").exists() and not any(
                e["steps"] == 0 for e in evaluations(run_dir, args.eval_starts, track)):
            # A run whose step-0 evaluation was lost (stopped before it was saved)
            # measures its general-policy baseline now, the same way.
            print(f"BASELINE {track}: no step-0 evaluation; evaluating {args.init_from}", flush=True)
            subprocess.run([
                args.uv, "run", "--script", str(ROOT / "training/sac_async.py"), "--godot", args.godot,
                "--runs-dir", str(args.runs_dir), "--run-name", run_dir.name, "--resume", "--eval-only",
                "--init-from", str(args.init_from), "--track", track, "--reward-line", "progress",
                "--pedal-gain", "1.25", "--eval-starts", str(args.eval_starts),
                "--eval-horizon", str(args.eval_horizon), "--no-wandb"], cwd=ROOT, check=True)
        focus = run_dir / "focus.json"
        command = [
            args.uv, "run", "--script", str(ROOT / "training/sac_async.py"),
            "--godot", args.godot, "--runs-dir", str(args.runs_dir),
            "--run-name", run_dir.name, "--resume", "--init-from", str(args.init_from),
            "--track", track, "--reward-line", "progress", "--pedal-gain", "1.25",
            "--workers", str(args.workers), "--buffer-size", str(args.buffer_size),
            "--learning-rate", str(learning_rate), "--total-steps", str(10**12),
            "--focus-fraction", str(args.focus_fraction), "--focus-file", str(focus),
            "--focus-lead-min", str(args.focus_lead_min), "--focus-lead", str(args.focus_lead),
            *(["--focus-curriculum"] if args.focus_curriculum else []),
            "--max-start-speed", str(args.max_start_speed),
            *(["--apex-bonus-stations", args.apex_bonus_stations] if args.apex_bonus_stations else []),
            "--apex-bonus-floor", str(args.apex_bonus_floor),
            "--eval-every", str(args.eval_every), "--eval-starts", str(args.eval_starts),
            "--eval-horizon", str(args.eval_horizon), "--no-rollout-shards",
            "--apex-bonus-m", str(args.apex_bonus_m), "--throttle-bonus-m",
            str(args.throttle_bonus_m), *(["--apex-bonus-dense"] if args.apex_bonus_dense else []),
            *(["--resume-weights-from", str(self.resume_weights)] if self.resume_weights else []),
            *(["--no-wandb"] if args.no_wandb else []),
        ]
        self.resume_weights = None  # once: later restarts resume the latest
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
            if (run_dir / "STOP").exists():  # a clean stop asked for through the run dir
                (run_dir / "STOP").unlink()
                self.child.send_signal(signal.SIGINT)
                break
            if free_gb(run_dir) < args.min_free_gb:
                self.child.send_signal(signal.SIGINT)
                break
            evals = evaluations(run_dir, args.eval_starts, track)
            if not evals or evals[-1]["checkpoint"] == reported:
                continue
            reported, latest = evals[-1]["checkpoint"], evals[-1]
            tmp = focus.with_suffix(".tmp")
            stations = sorted(set(latest["missed_apex_stations_m"] or []) | set(args.focus_stations))
            tmp.write_text(json.dumps(dict(stations=stations,
                                           checkpoint=latest["checkpoint"])) + "\n")
            tmp.replace(focus)
            general = evals[0]["best_lap_s"] if evals[0]["steps"] == 0 else None
            print(f"EVAL {track}: {describe(latest)}; best so far "
                  f"{describe(max(evals, key=rank))}; general {general} s", flush=True)
            decided = self.outcome(evals, learn, refine_from)
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
        shaping = getattr(self, "state", {}).get("shaping_from")
        best = max([e for e in evals if shaped_result_ok(e, shaping)], key=rank)
        general = next((e for e in evals if e["steps"] == 0), None)
        entry = dict(summary(best), checkpoint=str((run_dir / best["checkpoint"]).relative_to(ROOT)),
                     eval_starts=self.args.eval_starts, meets_bar=True,
                     trained_steps=self.latest_steps(run_dir),
                     lap_ceiling=ceiling_record(shaping),
                     # The step-0 evaluation rides the --init-from weights unchanged.
                     general=general and dict(summary(general), checkpoint=str(
                         self.args.init_from.relative_to(ROOT)), evaluated_at="step_0.pt"),
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
            if not self.train(track):
                stuck.append(track)
        print(f"ALL SPECIALISTS DONE; stuck: {stuck or 'none'}", flush=True)


def training_order(listing: Path) -> list[str]:
    """The general run's circuits in its order, LATER and LAST moved to the end;
    never a held-out one."""
    tracks = json.loads(listing.read_text())
    order = ([t for t in tracks if t not in LATER + LAST] + [t for t in LATER if t in tracks]
             + [t for t in LAST if t in tracks])
    return [t for t in order if not held_out(t)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--uv", default=shutil.which("uv") or "uv")
    parser.add_argument("--init-from", type=Path,
                        default=GENERAL / "checkpoints/step_40672729.pt")
    parser.add_argument("--tracks", help="comma list in training order; default: "
                        f"{GENERAL.name}/tracks.json, then {', '.join(LATER + LAST)}")
    parser.add_argument("--reuse", default=f"{DEFAULT_TRACK}=runs/sac/sac-v7-lr1e4-1/checkpoints/"
                        "step_24000071.pt", help="comma list of TRACK=CHECKPOINT teachers "
                        "published as they are, not trained")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/sac/specialists")
    parser.add_argument("--registry", type=Path, default=ROOT / "runs/sac/specialists.json")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--buffer-size", type=int, default=6_000_000)
    parser.add_argument("--focus-fraction", type=float, default=0.5)
    parser.add_argument("--apex-bonus-m", type=float, default=0.0,
                        help="opt-in reward shaping (sac_async --apex-bonus-m) for circuits "
                        "short of the apex bar")
    parser.add_argument("--throttle-bonus-m", type=float, default=0.0,
                        help="opt-in reward shaping (sac_async --throttle-bonus-m)")
    parser.add_argument("--apex-bonus-dense", action="store_true",
                        help="opt-in (sac_async --apex-bonus-dense)")
    parser.add_argument("--restart-from-best", action="store_true",
                        help="once per run: resume from the weights of its fastest evaluation "
                        "that lapped from every start (replay buffer kept), and publish "
                        "nothing slower than that lap")
    parser.add_argument("--focus-stations", type=lambda text: [float(x) for x in text.split(",")],
                        default=[], help="comma list of stations always in the focus file, "
                        "besides the latest evaluation's missed apexes")
    parser.add_argument("--focus-curriculum", action="store_true",
                        help="opt-in (sac_async --focus-curriculum)")
    parser.add_argument("--apex-bonus-floor", type=float, default=0.0,
                        help="sac_async --apex-bonus-floor")
    parser.add_argument("--apex-bonus-stations", default="",
                        help="comma list (sac_async --apex-bonus-stations)")
    parser.add_argument("--max-start-speed", type=float, default=40.0,
                        help="training start speed cap (sac_async --max-start-speed)")
    parser.add_argument("--focus-lead", type=float, default=150.0,
                        help="farthest a focused start begins before the spot (sac_async)")
    parser.add_argument("--focus-lead-min", type=float, default=25.0,
                        help="focused starts begin 25-150 m before the spot: close starts at "
                        "low speed let the policy meet a corner it never survives the "
                        "approach to (red-bull-ring turn 3)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--refine-learning-rate", type=float, default=3e-5)
    parser.add_argument("--min-delta", type=float, default=0.2,
                        help="seconds a best lap must improve by to count as progress")
    parser.add_argument("--stuck-steps", type=int, default=10_000_000,
                        help="env steps without progress after which a circuit is stuck")
    parser.add_argument("--min-total-steps", type=int, default=8_000_000)
    parser.add_argument("--min-refine-steps", type=int, default=3_000_000,
                        help="env steps at --refine-learning-rate before a circuit can be done")
    parser.add_argument("--plateau-steps", type=int, default=3_000_000)
    parser.add_argument("--plateau-fraction", type=float, default=0.002,
                        help="the best lap must improve by less than this over --plateau-steps")
    parser.add_argument("--min-throttle", type=float, default=MIN_THROTTLE,
                        help="median throttle while upright and off the brakes")
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--eval-starts", type=int, default=6)
    parser.add_argument("--eval-horizon", type=float, default=300.0)
    parser.add_argument("--min-free-gb", type=float, default=30.0,
                        help="stop below this much free disk (the final replay-buffer save "
                        "writes ~3 GB more)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.init_from = args.init_from.absolute()  # not resolve(): see runs_dir below
    # absolute(), not resolve(): on Modal runs/ is a volume symlinked out of the
    # checkout, and registry paths are relative to the checkout.
    args.runs_dir, args.registry = args.runs_dir.absolute(), args.registry.absolute()
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
