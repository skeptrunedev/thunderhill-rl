# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch==2.8.0"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""SAC rollouts as labelled records for Rev, our Jev-style decision model for riding.

Rev is a Qwen base model with a LoRA and a pointer head, trained from scratch on these
records with the open-source Kev code (github.com/jaredpalmer/kev); it never starts from
Kev's weights. It answers typed questions about a state with calibrated probabilities,
in the TypeSafe System One record format. Each record here is one control step of the SAC rider (the 64-dim sac_env observation, rendered
as labelled telemetry) with three questions:

  steer           choice over 7 levels of the steer action (hard left .. hard right)
  pedal           choice over 5 levels of the applied pedal (full brake .. full throttle)
  off_track_soon  noul: did the episode end off the road (offroad or fall) within the
                  next OFF_TRACK_HORIZON_STEPS control steps

steer and pedal carry soft targets: the continuous action split between its two
nearest levels, so the probability-weighted level recovers the action exactly and
full throttle stays reachable. Positive steer turns right (it moves the bike toward
the right edge); lateral offsets and edge positions are positive to the left.

Records come from whole episodes inside consecutive shard blocks, so a label never
depends on steps outside the loaded rows. Training and held-out records come from
different runs (fit the temperature on data from outside the training distribution); the held-out run is split in half by episode into calibration and
development.

With --teacher, steer and pedal targets are that SAC checkpoint's deterministic action
(sac_teacher.py) instead of the exploration samples SAC recorded while training, and
--dagger adds the states Rev visited on its own rides (rev_drive.py --collect), labelled
by the same teacher: DAgger, so Rev also learns to recover from its own mistakes.

  uv run training/rev_dataset.py --out runs/rev/data-v1
  uv run training/rev_dataset.py --teacher runs/sac/sac-v7-lr1e4-1/checkpoints/step_24000071.pt \
      --dagger runs/rev/collect-r1 --out runs/rev/data-r1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COURSE_DISTANCES_M = (5, 10, 20, 30, 45, 60, 80, 100, 125, 150)  # sac_env.COURSE_DISTANCES_M
OFF_TRACK_HORIZON_STEPS = 10  # one second at 10 Hz
OFF_TRACK_TERMINATIONS = ("offroad", "fall")

STEER_LEVELS = {
    "hard left": -1.0, "left": -2 / 3, "slight left": -1 / 3, "straight": 0.0,
    "slight right": 1 / 3, "right": 2 / 3, "hard right": 1.0,
}
PEDAL_LEVELS = {
    "full brake": -1.0, "brake": -0.5, "coast": 0.0, "part throttle": 0.5, "full throttle": 1.0,
}
QUESTIONS = {
    "steer": {"type": "choice", "criteria": {name: None for name in STEER_LEVELS},
              "instructions": "How should the rider steer on this control step to ride the "
                              "fastest line that stays on the track?"},
    "pedal": {"type": "choice", "criteria": {name: None for name in PEDAL_LEVELS},
              "instructions": "What throttle or brake should the rider apply on this control "
                              "step to ride the fastest line that stays on the track?"},
    "off_track_soon": {"type": "noul",
                       "instructions": "Will the motorcycle leave the track or fall within "
                                       "the next second?"},
}


AHEAD = "/".join(str(d) for d in COURSE_DISTANCES_M)


def render_state(obs) -> dict:
    """The sac_env observation vector as labelled telemetry in physical units, short
    enough for the 384-token training states of the Kev trainer."""
    o = np.asarray(obs, dtype=np.float64)
    course = o[13:53].reshape(len(COURSE_DISTANCES_M), 4) * 50.0  # left fwd, left lat, right fwd, right lat
    ahead = lambda values: " ".join(str(v) for v in values)  # noqa: E731
    # Edges to 0.1 m where the next few metres decide the line, whole metres beyond.
    edge = lambda values: ahead(round(v, 1) if d <= 30 else round(v)  # noqa: E731
                                for d, v in zip(COURSE_DISTANCES_M, values))
    return {
        "speed km/h": round(o[0] * 40.0 * 3.6),
        "accel g (forward, lateral)": f"{o[1] * 10.0 / 9.81:.2f}, {o[2] * 10.0 / 9.81:.2f}",
        "lean deg, rate deg/s": f"{math.degrees(o[3]):.0f}, {math.degrees(o[4] * 2.0):.0f}",
        "front steering deg": round(math.degrees(o[5] * 0.5), 1),
        "heading vs track deg": round(math.degrees(math.atan2(o[6], o[7])), 1),
        "offset from center (+left, 1 = edge)": round(o[8], 2),
        "half width m": round(o[9] * 10.0, 1),
        "gear": round(o[10] * 6.0),
        "previous steer (+right), pedal": f"{o[11]:.2f}, {o[12]:.2f}",
        f"curvature 1/km (+left) at {AHEAD} m": ahead(round(c * 1000.0 / 50.0) for c in o[53:63]),
        f"left edge m (+left) at {AHEAD} m": edge(course[:, 1]),
        f"right edge m (+left) at {AHEAD} m": edge(course[:, 3]),
        "on track": bool(o[63] > 0.5),
    }


def soft_target(value: float, levels: dict) -> dict:
    """Split value between its two bracketing levels so the expected level equals it."""
    names, points = list(levels), np.array(list(levels.values()))
    value = float(np.clip(value, points[0], points[-1]))
    upper = int(np.clip(np.searchsorted(points, value), 1, len(points) - 1))
    lower = upper - 1
    t = (value - points[lower]) / (points[upper] - points[lower])
    target = {names[lower]: round(1.0 - t, 4), names[upper]: round(t, 4)}
    return {name: weight for name, weight in target.items() if weight > 0}


def expected_level(probabilities: dict, levels: dict) -> float:
    """The action a steer or pedal answer stands for: its probability-weighted level."""
    return float(sum(p * levels[name] for name, p in probabilities.items()))


def record(obs, steer: float, pedal: float, off_track: bool) -> dict:
    """steer and pedal as applied: pedal after sac_env.controls' gain and clip."""
    questions = {}
    for qid, value, levels in (("steer", steer, STEER_LEVELS), ("pedal", pedal, PEDAL_LEVELS)):
        target = soft_target(value, levels)
        questions[qid] = {**QUESTIONS[qid], "label": max(target, key=target.get), "target": target}
    questions["off_track_soon"] = {**QUESTIONS["off_track_soon"], "label": off_track}
    return {"state": render_state(obs), "questions": questions}


def episode_rows(run_dir: Path, shards: list[Path]):
    """(obs, action, off_track) for every step of the episodes complete inside shards."""
    terminations = {}
    for line in (run_dir / "rollouts/episodes.jsonl").open():
        row = json.loads(line)
        terminations[row["episode"]] = (row["termination"], row["steps"])
    blocks = [np.load(path) for path in shards]
    episode = np.concatenate([b["episode"] for b in blocks])
    obs = np.concatenate([b["obs"] for b in blocks])
    action = np.concatenate([b["action"] for b in blocks])
    order = np.argsort(episode, kind="stable")  # stable: each episode's rows stay in time order
    ids, starts, counts = np.unique(episode[order], return_index=True, return_counts=True)
    for eid, start, count in zip(ids, starts, counts):
        termination, steps = terminations.get(int(eid), (None, None))
        if steps != count:  # started before the block or continues after it
            continue
        rows = order[start:start + count]
        off = np.zeros(count, dtype=bool)
        if termination in OFF_TRACK_TERMINATIONS:
            off[-OFF_TRACK_HORIZON_STEPS:] = True
        yield int(eid), obs[rows], action[rows], off


def sample(run_dir: Path, n: int, positive_fraction: float, blocks: int, block_shards: int,
           rng: np.random.Generator):
    """n records from `blocks` evenly spaced blocks of consecutive shards; off-track
    steps are oversampled to positive_fraction. Returns {episode: [record inputs]}."""
    shards = sorted((run_dir / "rollouts").glob("shard-*.npz"))
    first = np.linspace(0, len(shards) - block_shards, blocks).round().astype(int)
    n_neg = n - round(n * positive_fraction)
    quotas = [n_neg // blocks + (b < n_neg % blocks) for b in range(blocks)]
    positives, picked, seen = [], [], 0
    for k, quota in zip(first, quotas):
        negatives = []
        for eid, obs, action, off in episode_rows(run_dir, shards[k:k + block_shards]):
            positives += [(eid, obs[i], action[i], True) for i in np.flatnonzero(off)]
            negatives += [(eid, obs[i], action[i], False) for i in np.flatnonzero(~off)]
        seen += len(negatives)
        picked += [negatives[i] for i in rng.choice(len(negatives), quota, replace=False)]
    n_pos = min(len(positives), n - len(picked))
    picked += [positives[i] for i in rng.choice(len(positives), n_pos, replace=False)]
    print(f"{run_dir.name}: {len(positives)} off-track and {seen} other steps "
          f"in {blocks}x{block_shards} shards; picked {n_pos} + {len(picked) - n_pos}")
    return picked


def write(path: Path, rows, pedal_gain: float, teacher=None):
    """rows are (episode, obs, recorded raw action or None, off_track); with a teacher the
    targets are its actions, otherwise the recorded ones through sac_env.controls."""
    obs = np.stack([row[1] for row in rows])
    if teacher is not None:
        controls = teacher.controls(obs)
    else:
        raw = np.stack([row[2] for row in rows])
        controls = np.stack([np.clip(raw[:, 0], -1, 1), np.clip(pedal_gain * raw[:, 1], -1, 1)], axis=1)
    with path.open("w") as stream:
        for (_, o, _, off), (steer, pedal) in zip(rows, controls):
            stream.write(json.dumps(record(o, float(steer), float(pedal), off)) + "\n")


def dagger_rows(directory):
    """(episode, obs, None, off_track) for every step Rev rode in one rev_drive --collect run."""
    rows = []
    for path in sorted(Path(directory).glob("collect-*.npz")):
        z = np.load(path)
        rows += [(f"{Path(directory).name}/{path.stem}:{e}", o, None, bool(off))
                 for e, o, off in zip(z["episode"], z["obs"], z["off_track"])]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-runs", nargs="+", default=["sac-v7-lr1e4-1", "sac-v7-pedal-2"])
    parser.add_argument("--heldout-run", default="sac-v7-lr3e5-1")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/sac")
    parser.add_argument("--train-records", type=int, default=30_000)
    parser.add_argument("--heldout-records", type=int, default=3_000)
    parser.add_argument("--positive-fraction", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-shards", type=int, default=3)
    parser.add_argument("--pedal-gain", type=float, default=1.25)
    parser.add_argument("--teacher", type=Path,
                        help="SAC checkpoint whose deterministic action labels steer and pedal")
    parser.add_argument("--dagger", type=Path, nargs="*", default=[],
                        help="rev_drive --collect directories: Rev's own states, teacher-labelled")
    parser.add_argument("--max-train-records", type=int, default=0,
                        help="keep every state of the last --dagger collection and fill the rest of "
                             "this budget with a uniform sample of the older records")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if set(args.train_runs) & {args.heldout_run}:
        parser.error("the held-out run must not be a training run")
    if args.dagger and not args.teacher:
        parser.error("--dagger states have no recorded action; pass --teacher to label them")
    teacher = None
    if args.teacher:
        from sac_teacher import SacTeacher
        teacher = SacTeacher(args.teacher, args.pedal_gain)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)

    per_run = [args.train_records // len(args.train_runs)] * len(args.train_runs)
    per_run[0] += args.train_records - sum(per_run)
    train = []
    for run, n in zip(args.train_runs, per_run):
        train += sample(args.runs_dir / run, n, args.positive_fraction, args.blocks, args.block_shards, rng)
    collections = [dagger_rows(directory) for directory in args.dagger]
    dagger = [row for rows in collections for row in rows]
    print(f"{len(dagger)} DAgger states from {len(args.dagger)} collections")
    if args.max_train_records and len(train) + len(dagger) > args.max_train_records:
        newest = collections[-1] if collections else []
        older = train + [row for rows in collections[:-1] for row in rows]
        keep = max(0, args.max_train_records - len(newest))
        train = [older[i] for i in rng.choice(len(older), min(keep, len(older)), replace=False)] + newest
        print(f"capped to {len(train)} records: all {len(newest)} newest DAgger states + {len(train) - len(newest)} older")
    else:
        train += dagger
    rng.shuffle(train)
    heldout = sample(args.runs_dir / args.heldout_run, args.heldout_records, args.positive_fraction,
                     args.blocks, args.block_shards, rng)
    # Whole episodes to one side, so calibration and development never share a trajectory.
    episodes = sorted({eid for eid, *_ in heldout})
    calibration_ids = set(rng.permutation(episodes)[: len(episodes) // 2].tolist())
    calibration = [row for row in heldout if row[0] in calibration_ids]
    development = [row for row in heldout if row[0] not in calibration_ids]

    for name, rows in (("train", train), ("calibration", calibration), ("development", development)):
        write(args.out / f"{name}.jsonl", rows, args.pedal_gain, teacher)
    summary = {
        "train_runs": args.train_runs, "heldout_run": args.heldout_run,
        "teacher": str(args.teacher) if args.teacher else None, "dagger_states": len(dagger),
        "records": {"train": len(train), "calibration": len(calibration), "development": len(development)},
        "off_track_fraction": {name: round(float(np.mean([r[3] for r in rows])), 4) for name, rows in
                               (("train", train), ("calibration", calibration), ("development", development))},
        "args": json.loads(json.dumps(vars(args), default=str)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["records"]), json.dumps(summary["off_track_fraction"]))


if __name__ == "__main__":
    main()
