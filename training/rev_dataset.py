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

Every state belongs to a circuit (a multi-track SAC run's shard column `track`, a
collection's `track` array; Thunderhill for the single-track runs), and the state names
it in its first line, `circuit: jerez`. The name lets Rev tell the circuits' teachers
apart and learn what the 150 m of course ahead cannot show (a braking zone after a long
straight); --circuit-dropout renders that share of training states without it, so Rev
also keeps a policy that reads the course alone. Held-out test tracks (lap_policy.held_out:
portimao, laguna-seca) never enter a dataset; any state from one is refused.
--exclude-circuits leaves further circuits out of every part (e.g. those whose
specialist never cleared the bar), so they stay unseen by Rev and test it too.

Records come from whole episodes inside consecutive shard blocks, so a label never
depends on steps outside the loaded rows. Training and held-out records come from
different policies: another run, or the last --heldout-shards shards of a training run,
which its training records then leave out (fit the temperature on data from outside the
training distribution). The held-out records are split in half by episode into
calibration and development.

With --teachers, steer and pedal targets are the deterministic action (sac_teacher.py) of
each state's circuit teacher (its specialist in the registry, else the general policy)
instead of the exploration samples SAC recorded while training, and --dagger adds the
states Rev visited on its own rides (rev_drive.py --collect), labelled the same way:
DAgger, so Rev also learns to recover from its own mistakes. Labels are computed when the
dataset is built, so a rebuild after a specialist lands relabels that circuit's states.

  uv run training/rev_dataset.py --out runs/rev/data-v1
  uv run training/rev_dataset.py --teachers runs/sac/specialists.json --train-runs sac-multitrack-1 \
      --heldout-run sac-multitrack-1 --dagger runs/rev/collect-m1 --out runs/rev/data-m1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lap_policy import DEFAULT_TRACK, held_out  # noqa: E402

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


def render_state(obs, circuit: str | None = None) -> dict:
    """The sac_env observation vector as labelled telemetry in physical units, short
    enough for the 384-token training states of the Kev trainer, after the circuit's name."""
    o = np.asarray(obs, dtype=np.float64)
    course = o[13:53].reshape(len(COURSE_DISTANCES_M), 4) * 50.0  # left fwd, left lat, right fwd, right lat
    ahead = lambda values: " ".join(str(v) for v in values)  # noqa: E731
    # Edges to 0.1 m where the next few metres decide the line, whole metres beyond.
    edge = lambda values: ahead(round(v, 1) if d <= 30 else round(v)  # noqa: E731
                                for d, v in zip(COURSE_DISTANCES_M, values))
    return {
        **({"circuit": circuit} if circuit else {}),
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


def record(obs, steer: float, pedal: float, off_track: bool, circuit: str | None = None) -> dict:
    """steer and pedal as applied: pedal after sac_env.controls' gain and clip."""
    questions = {}
    for qid, value, levels in (("steer", steer, STEER_LEVELS), ("pedal", pedal, PEDAL_LEVELS)):
        target = soft_target(value, levels)
        questions[qid] = {**QUESTIONS[qid], "label": max(target, key=target.get), "target": target}
    questions["off_track_soon"] = {**QUESTIONS["off_track_soon"], "label": off_track}
    return {"state": render_state(obs, circuit), "questions": questions}


def run_tracks(run_dir: Path) -> list[str]:
    """The circuits a SAC run's shard column `track` indexes; one-circuit runs have none."""
    path = run_dir / "tracks.json"
    return json.loads(path.read_text()) if path.exists() else [DEFAULT_TRACK]


def refuse_held_out(tracks):
    if bad := sorted({str(t) for t in tracks if held_out(str(t))}):
        raise SystemExit(f"states from held-out test tracks {bad}: they are never trained on")


def episode_rows(run_dir: Path, shards: list[Path]):
    """(episode, obs, action, off_track, circuit) for every step of the episodes complete inside shards."""
    terminations = {}
    for line in (run_dir / "rollouts/episodes.jsonl").open():
        row = json.loads(line)
        terminations[row["episode"]] = (row["termination"], row["steps"])
    names = run_tracks(run_dir)
    blocks = [np.load(path) for path in shards]
    episode = np.concatenate([b["episode"] for b in blocks])
    obs = np.concatenate([b["obs"] for b in blocks])
    action = np.concatenate([b["action"] for b in blocks])
    track = np.concatenate([b["track"] if "track" in b.files else np.zeros(len(b["episode"]), np.int16)
                            for b in blocks])
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
        yield int(eid), obs[rows], action[rows], off, names[int(track[rows[0]])]


def run_shards(run_dir: Path, heldout_shards: int, part: str) -> list[Path]:
    """A run's rollout shards; with heldout_shards, the last ones are the "heldout" part
    and the rest the "train" part."""
    shards = sorted((run_dir / "rollouts").glob("shard-*.npz"))
    if not heldout_shards:
        return shards
    return shards[-heldout_shards:] if part == "heldout" else shards[:-heldout_shards]


def sample(run_dir: Path, shards: list[Path], n: int, positive_fraction: float, blocks: int,
           block_shards: int, rng: np.random.Generator):
    """n records from `blocks` evenly spaced blocks of consecutive shards; off-track
    steps are oversampled to positive_fraction. Returns [(episode, obs, action, off, circuit)]."""
    blocks = min(blocks, len(shards) // block_shards)  # blocks never overlap
    first = np.linspace(0, len(shards) - block_shards, blocks).round().astype(int)
    n_neg = n - round(n * positive_fraction)
    quotas = [n_neg // blocks + (b < n_neg % blocks) for b in range(blocks)]
    positives, picked, seen = [], [], 0
    for k, quota in zip(first, quotas):
        negatives = []
        for eid, obs, action, off, circuit in episode_rows(run_dir, shards[k:k + block_shards]):
            positives += [(eid, obs[i], action[i], True, circuit) for i in np.flatnonzero(off)]
            negatives += [(eid, obs[i], action[i], False, circuit) for i in np.flatnonzero(~off)]
        seen += len(negatives)
        picked += [negatives[i] for i in rng.choice(len(negatives), quota, replace=False)]
    n_pos = min(len(positives), n - len(picked))
    picked += [positives[i] for i in rng.choice(len(positives), n_pos, replace=False)]
    print(f"{run_dir.name}: {len(positives)} off-track and {seen} other steps "
          f"in {blocks}x{block_shards} shards; picked {n_pos} + {len(picked) - n_pos}")
    return picked


def write(path: Path, rows, pedal_gain: float, teachers=None, circuit_dropout: float = 0.0, rng=None):
    """rows are (episode, obs, recorded raw action or None, off_track, circuit); with teachers
    the targets are the circuit teacher's actions, otherwise the recorded ones through
    sac_env.controls. A circuit_dropout share of the states leave out their circuit's name."""
    obs = np.stack([row[1] for row in rows])
    if teachers is not None:
        controls = teachers.controls(obs, [row[4] for row in rows])
    else:
        raw = np.stack([row[2] for row in rows])
        controls = np.stack([np.clip(raw[:, 0], -1, 1), np.clip(pedal_gain * raw[:, 1], -1, 1)], axis=1)
    named = rng.random(len(rows)) >= circuit_dropout if circuit_dropout else np.ones(len(rows), dtype=bool)
    with path.open("w") as stream:
        for (_, o, _, off, circuit), (steer, pedal), name in zip(rows, controls, named):
            stream.write(json.dumps(record(o, float(steer), float(pedal), off, circuit if name else None)) + "\n")


def dagger_rows(directory):
    """(episode, obs, None, off_track, circuit) for every step Rev rode in one rev_drive --collect
    run; collections from before multi-track riding have no `track` and rode Thunderhill."""
    rows = []
    for path in sorted(Path(directory).glob("collect-*.npz")):
        z = np.load(path)
        tracks = z["track"] if "track" in z.files else np.full(len(z["episode"]), DEFAULT_TRACK)
        rows += [(f"{Path(directory).name}/{path.stem}:{e}", o, None, bool(off), str(t))
                 for e, o, off, t in zip(z["episode"], z["obs"], z["off_track"], tracks)]
    return rows


def replay(directory: Path, n: int, train_path: Path, teachers, rng) -> int:
    """Append n random training lines of a previous dataset to train_path; refuses a dataset labelled by other
    teachers (its summary's registry description must match these teachers' for its circuits)."""
    summary = json.loads((directory / "summary.json").read_text())
    previous = summary.get("teachers")
    if teachers is None or previous is None or teachers.describe(sorted(previous)) != previous:
        raise SystemExit(f"{directory} was labelled by other teachers; it cannot be replayed with these")
    total = summary["records"]["train"]
    if not 0 < n <= total:
        raise SystemExit(f"--replay-records must be 1 through {total}")
    picked = set(rng.choice(total, n, replace=False).tolist())
    with (directory / "train.jsonl").open() as source, train_path.open("a") as out:
        for index, line in enumerate(source):
            if index in picked:
                out.write(line)
    print(f"replayed {n} of {total} training records from {directory}")
    return n


def balanced(rows, n: int, rng) -> list:
    """n rows with the circuits' shares as equal as their rows allow: circuits are filled
    smallest first, and one with fewer rows than an equal share of what is left gives
    them all."""
    by_circuit = {}
    for row in rows:
        by_circuit.setdefault(row[4], []).append(row)
    picked, budget, remaining = [], min(n, len(rows)), len(by_circuit)
    for circuit, group in sorted(by_circuit.items(), key=lambda item: len(item[1])):
        take = min(len(group), budget // remaining)
        picked += [group[i] for i in rng.choice(len(group), take, replace=False)]
        budget, remaining = budget - take, remaining - 1
    return picked


def circuit_counts(rows) -> dict:
    counts = {}
    for row in rows:
        counts[row[4]] = counts.get(row[4], 0) + 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-runs", nargs="+", default=["sac-v7-lr1e4-1", "sac-v7-pedal-2"])
    parser.add_argument("--heldout-run", default="sac-v7-lr3e5-1")
    parser.add_argument("--heldout-shards", type=int, default=0,
                        help="with a --heldout-run that is also a training run: its last shards are the "
                             "held-out records and its training records leave them out")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/sac")
    parser.add_argument("--train-records", type=int, default=30_000)
    parser.add_argument("--heldout-records", type=int, default=3_000)
    parser.add_argument("--positive-fraction", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-shards", type=int, default=3)
    parser.add_argument("--pedal-gain", type=float, default=1.25)
    parser.add_argument("--teachers", type=Path,
                        help="sac_specialists registry: each circuit's specialist labels steer and pedal "
                             "(the general --general-teacher where a circuit has none)")
    parser.add_argument("--general-teacher", type=Path, help="default: sac_teacher.GENERAL")
    parser.add_argument("--circuit-dropout", type=float, default=0.0,
                        help="share of training states rendered without their circuit's name")
    parser.add_argument("--dagger", type=Path, nargs="*", default=[],
                        help="rev_drive --collect directories: Rev's own states, teacher-labelled")
    parser.add_argument("--max-train-records", type=int, default=0,
                        help="keep every state of the last --newest --dagger collections and fill the rest "
                             "of this budget with a sample of the older records balanced across circuits")
    parser.add_argument("--circuit-floor", type=int, default=0,
                        help="keep every --dagger state and top each circuit up to this many training records "
                             "with sampled SAC states (DAgger's aggregate: no round's states are ever dropped)")
    parser.add_argument("--newest", type=int, default=1,
                        help="how many of the last --dagger collections --max-train-records keeps whole "
                             "(a collection split over several Modal containers is several directories)")
    parser.add_argument("--replay", type=Path,
                        help="a previous dataset directory: --replay-records of its train.jsonl lines join the new "
                             "training records verbatim (already labelled by the same teachers), so a DAgger round "
                             "labels only its own states; with --train-records 0 no SAC states are sampled")
    parser.add_argument("--replay-records", type=int, default=0)
    parser.add_argument("--exclude-circuits", default="",
                        help="comma list of circuits whose states never enter the dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    shared = args.heldout_run in args.train_runs
    if shared != bool(args.heldout_shards):
        parser.error("a --heldout-run that is also a training run needs --heldout-shards, and only then")
    if args.dagger and not args.teachers:
        parser.error("--dagger states have no recorded action; pass --teachers to label them")
    teachers = None
    if args.teachers:
        from sac_teacher import GENERAL, Teachers
        teachers = Teachers(args.teachers, args.general_teacher or GENERAL, args.pedal_gain)
    rng = np.random.default_rng(args.seed)

    per_run = [args.train_records // len(args.train_runs)] * len(args.train_runs)
    per_run[0] += args.train_records - sum(per_run)
    train = []
    for run, n in zip(args.train_runs, per_run):
        if not n:
            continue
        shards = run_shards(args.runs_dir / run, args.heldout_shards if run == args.heldout_run else 0, "train")
        train += sample(args.runs_dir / run, shards, n, args.positive_fraction, args.blocks, args.block_shards, rng)
    excluded = {t.strip() for t in args.exclude_circuits.split(",") if t.strip()}
    keep = lambda rows: [row for row in rows if row[4] not in excluded]  # noqa: E731
    train = keep(train)
    collections = [keep(dagger_rows(directory)) for directory in args.dagger]
    dagger = [row for rows in collections for row in rows]
    print(f"{len(dagger)} DAgger states from {len(args.dagger)} collections")
    if args.circuit_floor:
        if args.max_train_records:
            parser.error("--circuit-floor and --max-train-records are alternatives")
        have = circuit_counts(dagger)
        sac = {}
        for row in train:
            sac.setdefault(row[4], []).append(row)
        topped = []
        for circuit, rows in sorted(sac.items()):
            need = max(0, args.circuit_floor - have.get(circuit, 0))
            if need > len(rows):
                print(f"{circuit}: only {len(rows)} SAC states to top up {need}; raise --train-records")
            topped += [rows[i] for i in rng.choice(len(rows), min(need, len(rows)), replace=False)]
        train = topped + dagger
        print(f"aggregate: all {len(dagger)} DAgger states + {len(topped)} SAC states toward {args.circuit_floor} per circuit")
    elif args.max_train_records and len(train) + len(dagger) > args.max_train_records:
        split = len(collections) - min(args.newest, len(collections))
        newest = [row for rows in collections[split:] for row in rows]
        older = train + [row for rows in collections[:split] for row in rows]
        train = balanced(older, max(0, args.max_train_records - len(newest)), rng) + newest
        print(f"capped to {len(train)} records: all {len(newest)} newest DAgger states + {len(train) - len(newest)} older")
    else:
        train += dagger
    rng.shuffle(train)
    run_dir = args.runs_dir / args.heldout_run
    heldout = keep(sample(run_dir, run_shards(run_dir, args.heldout_shards, "heldout"), args.heldout_records,
                          args.positive_fraction, args.blocks, args.block_shards, rng))
    refuse_held_out({row[4] for row in train + heldout})
    # Whole episodes to one side, so calibration and development never share a trajectory.
    episodes = sorted({eid for eid, *_ in heldout})
    calibration_ids = set(rng.permutation(episodes)[: len(episodes) // 2].tolist())
    calibration = [row for row in heldout if row[0] in calibration_ids]
    development = [row for row in heldout if row[0] not in calibration_ids]

    args.out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", train), ("calibration", calibration), ("development", development)):
        # Calibration and development states always name their circuit, as rides do.
        write(args.out / f"{name}.jsonl", rows, args.pedal_gain, teachers,
              args.circuit_dropout if name == "train" else 0.0, rng)
    replayed = replay(args.replay, args.replay_records, args.out / "train.jsonl", teachers, rng) if args.replay else 0
    parts = (("train", train), ("calibration", calibration), ("development", development))
    summary = {
        "train_runs": args.train_runs, "heldout_run": args.heldout_run,
        "teachers": teachers.describe(sorted({row[4] for _, rows in parts for row in rows})) if teachers else None,
        "dagger_states": len(dagger),
        "records": {name: len(rows) + (replayed if name == "train" else 0) for name, rows in parts},
        "replayed": {"from": str(args.replay), "records": replayed} if args.replay else None,
        "circuits": {name: circuit_counts(rows) for name, rows in parts},
        "off_track_fraction": {name: round(float(np.mean([r[3] for r in rows])), 4) for name, rows in parts},
        "args": json.loads(json.dumps(vars(args), default=str)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["records"]), json.dumps(summary["off_track_fraction"]))
    print(json.dumps(summary["circuits"]["train"]))


if __name__ == "__main__":
    main()
