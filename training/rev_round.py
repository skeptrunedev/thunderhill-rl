"""One fast Rev DAgger round on Modal: collect, label, train, evaluate, compare.

  1. collect   the parent rides --containers L4s (modal_rev_eval.py::collect), half the
               episodes starting just before the offroads and falls of its own evaluation
  2. dataset   rev_dataset.py labels only those new states with the specialist teachers and
               replays --replay-records already-labelled records of the previous dataset
  3. train     modal_rev.py::train on --gpus H100s (data-parallel LoRA), from the parent
  4. evaluate  the standard 96 rides (4 starts on each of 20 training circuits and 4 held-out
               ones) split over 4 L4s; a part Modal preempts is ridden again under a fresh name
  5. compare   laps and legal progress against the parent on the same starts

Every stage is skipped when its output exists, so a round that stopped resumes where it was.

  uv run training/rev_round.py --parent rev-0.8b-s7 --name rev-0.8b-s8 --previous-data runs/rev/data-s7
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "runs/rev"
MODAL = [str(Path.home() / ".local/bin/uvx"), "--from", "modal==1.5.5", "modal", "run"]
TRAINING = ("aragon,assen,balaton-park,brno,catalunya,goiania,jerez,le-mans,lusail,mandalika,misano,"
            "motegi,mugello,phillip-island,red-bull-ring,sachsenring,sepang,silverstone,thunderhill-east,valencia")
HELD_OUT = ("buriram", "cota", "portimao", "laguna-seca")
EXCLUDED = ",".join(HELD_OUT)
DAGGER_HISTORY = ["teacher-s1", "teacher-s1-motegi", "teacher-s1-aragon"] + [f"collect-r{i}" for i in range(1, 9)] + [
    "collect-m1", "collect-s2"]


def stage(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd: list[str], log: Path) -> int:
    with log.open("a") as stream:
        return subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def eval_groups(starts: int) -> list[tuple[str, str]]:
    """The standard evaluation in parts of about 24 rides: 4 parts at 4 starts per circuit (five training
    circuits and one held-out track each), 8 parts at 8 starts (the held-out tracks go to the first four)."""
    tracks, parts = TRAINING.split(","), 4 if starts <= 4 else 8
    return [(",".join(tracks[i::parts]), HELD_OUT[i] if i < len(HELD_OUT) else "") for i in range(parts)]


def eval_prefix(model: str, starts: int) -> str:
    """4 starts per circuit are NAME-eval-*; any other count NAME-eval<starts>-*. Evenly spaced starts
    (sac_env.evaluation_starts) make the 4 a subset of the 8, so the two compare start for start."""
    return f"{model}-eval" if starts == 4 else f"{model}-eval{starts}"


def eval_parts(model: str, starts: int = 4) -> list[Path]:
    """The completed evaluation parts of a model at this many starts: this script's groups, or (4 starts)
    the older two-part split."""
    parts, prefix = [], eval_prefix(model, starts)
    patterns = (f"{prefix}-g*",) if starts != 4 else (f"{prefix}-g*", f"{prefix}-[ab]", f"{prefix}-b2")
    for pattern in patterns:
        for path in sorted(REV.glob(pattern)):
            provenance = path / "provenance.json"
            if provenance.exists() and json.loads(provenance.read_text())["status"] == "complete":
                parts.append(path)
    return parts


def collected(tag: str) -> list[str]:
    """The round's completed collection parts (a preempted part is ridden again under another index)."""
    parts = []
    for path in sorted(REV.glob(f"collect-{tag}-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1])):
        provenance = path / "provenance.json"
        if provenance.exists() and json.loads(provenance.read_text())["status"] == "complete":
            parts.append(path.name)
    return parts


def collect(args, log: Path) -> list[str]:
    parts = collected(args.tag)
    if len(parts) >= args.containers:
        stage(f"collection {args.tag} exists: {len(parts)} parts")
        return parts
    evaluated = eval_parts(args.parent, args.eval_starts) or eval_parts(args.parent)
    focus = ",".join(path.name for path in evaluated)
    if not focus:
        raise SystemExit(f"{args.parent} has no completed evaluation to focus the collection on")
    # Circuits the parent fails get more of the episodes: weight 1 + 3 x its failed share of starts.
    weights = {}
    for row in (json.loads(line) for path in evaluated for line in (path / "eval/eval.jsonl").open()):
        if not row["held_out"]:
            weights.setdefault(row["track"], []).append(row["laps"] == 0)
    weights = {track: round(1 + 3 * sum(failed) / len(failed), 2) for track, failed in weights.items()}
    missing = args.containers - len(parts)
    first = 1 + max((int(p.name.rsplit("-", 1)[-1]) for p in REV.glob(f"collect-{args.tag}-*")), default=-1)
    stage(f"collecting {missing} of {args.containers} parts on L4s from {args.parent}, focus {focus}")
    stage(f"episode weights by failure: {json.dumps({t: w for t, w in sorted(weights.items(), key=lambda x: -x[1]) if w > 1})}")
    code = run([*MODAL, "training/modal_rev_eval.py::collect", "--checkpoint", args.parent, "--name",
                f"collect-{args.tag}", "--focus-from", focus, "--containers", str(missing),
                "--episodes", str(args.episodes), "--workers", "8", "--first-index", str(first),
                "--track-weights", json.dumps(weights)], log)
    parts = collected(args.tag)
    if code or len(parts) < args.containers:
        raise SystemExit(f"collection has {len(parts)} of {args.containers} parts; see {log}")
    return parts


def all_collections() -> list[str]:
    """Every DAgger collection so far, oldest first: the history plus each round's completed parts."""
    rounds = sorted({path.name.rsplit("-", 1)[0] for path in REV.glob("collect-s*-*") if path.name.rsplit("-", 1)[-1].isdigit()},
                    key=lambda prefix: int(prefix.removeprefix("collect-s")))
    return DAGGER_HISTORY + [part for prefix in rounds for part in collected(prefix.removeprefix("collect-"))]


def dataset(args, collections: list[str], log: Path) -> Path:
    out = REV / f"data-{args.tag}"
    if (out / "summary.json").exists():
        stage(f"{out.name} exists")
        return out
    common = ["--teachers", "runs/sac/specialists.json", "--train-runs", "sac-multitrack-1",
              "--heldout-run", "sac-multitrack-1", "--heldout-shards", "12", "--circuit-dropout", "0.2",
              "--exclude-circuits", EXCLUDED, "--seed", str(args.seed)]
    if args.circuit_floor:
        everything = all_collections()
        stage(f"aggregating all {len(everything)} DAgger collections, SAC states toward {args.circuit_floor} per circuit")
        recipe = ["--train-records", str(args.sac_records), "--circuit-floor", str(args.circuit_floor),
                  "--dagger", *[f"runs/rev/{name}" for name in everything]]
    else:
        stage(f"labelling {len(collections)} collections and replaying {args.replay_records} records of {args.previous_data}")
        recipe = ["--train-records", "0", "--replay", str(args.previous_data),
                  "--replay-records", str(args.replay_records), "--dagger", *[f"runs/rev/{name}" for name in collections]]
    code = run([str(Path.home() / ".local/bin/uv"), "run", "training/rev_dataset.py", *common, *recipe,
                "--out", str(out)], log)
    if code:
        raise SystemExit(f"dataset build failed; see {log}")
    return out


def train(args, data: Path, log: Path):
    if (REV / args.name / "result.json").exists():
        stage(f"{args.name} is trained")
        return
    stage(f"training {args.name} from {args.parent} on {args.gpus} H100s")
    code = run([*MODAL, "training/modal_rev.py::train", "--data", str(data), "--name", args.name, "--size", args.size,
                "--init-from", args.parent, "--shared-prefix", "1", "--epochs", "1", "--lr", str(args.lr),
                "--batch", str(args.batch), "--accum", str(args.accum), "--gpus", str(args.gpus), "--seed", str(args.seed),
                *(["--lora", str(args.lora)] if args.lora else []),
                *(["--upload-init"] if args.upload_init else [])], log)
    if code or not (REV / args.name / "result.json").exists():
        raise SystemExit(f"training failed; see {log}")


def evaluate(args, log: Path, model: str | None = None):
    model = model or args.name
    starts = args.eval_starts

    def part(index: int, tracks: str, heldout: str):
        for attempt in range(3):  # Modal can preempt an L4; its restart is refused, so ride again fresh
            name = f"{eval_prefix(model, starts)}-g{index}" + (f"-r{attempt}" if attempt else "")
            if (REV / name / "provenance.json").exists():
                if json.loads((REV / name / "provenance.json").read_text())["status"] == "complete":
                    return name
                continue
            run([*MODAL, "training/modal_rev_eval.py::evaluate", "--checkpoint", model, "--name", name,
                 "--tracks", tracks, "--heldout-tracks", heldout, "--starts", str(starts),
                 "--heldout-starts", str(starts)], log)
            provenance = REV / name / "provenance.json"
            if provenance.exists() and json.loads(provenance.read_text())["status"] == "complete":
                return name
        raise SystemExit(f"evaluation part {index} failed three times; see {log}")

    groups = eval_groups(starts)
    if len(eval_parts(model, starts)) >= len(groups):
        stage(f"{model} is evaluated at {starts} starts")
        return
    stage(f"evaluating {model} at {starts} starts on {len(groups)} L4s")
    with ThreadPoolExecutor(len(groups)) as pool:
        names = list(pool.map(lambda group: part(*group), [(i, *g) for i, g in enumerate(groups)]))
    stage(f"evaluated: {', '.join(names)}")


def rides(model: str, starts: int = 4) -> list[dict]:
    rows = []
    for path in eval_parts(model, starts):
        rows += [json.loads(line) for line in (path / "eval/eval.jsonl").open()]
    return rows


def per_circuit(new: list[dict], old: list[dict]) -> dict:
    """Laps and mean legal progress per circuit, new against parent: where a round gained and lost."""
    table = {}
    for track in sorted({r["track"] for r in new}):
        a, b = [r for r in new if r["track"] == track], [r for r in old if r["track"] == track]
        mean = lambda rows: round(sum(r["legal_progress_m"] for r in rows) / max(1, len(rows)))  # noqa: E731
        table[track] = {"laps": f"{sum(r['laps'] > 0 for r in a)}/{len(a)}", "parent_laps": f"{sum(r['laps'] > 0 for r in b)}/{len(b)}",
                        "mean_m": mean(a), "parent_mean_m": mean(b), "held_out": a[0]["held_out"]}
    return table


def compare(args):
    def summary(rows):
        trained = [r for r in rows if not r["held_out"]]
        heldout = [r for r in rows if r["held_out"]]
        progress = sorted(r["legal_progress_m"] for r in rows)
        return {"rides": len(rows), "laps_trained": sum(r["laps"] > 0 for r in trained), "trained": len(trained),
                "laps_heldout": sum(r["laps"] > 0 for r in heldout), "heldout": len(heldout),
                "mean_m": round(sum(progress) / len(progress)), "median_m": round(progress[len(progress) // 2]),
                "falls": sum(r["termination"] == "fall" for r in rows),
                "offroad": sum(r["termination"] == "offroad" for r in rows)}

    key = lambda r: (r["track"], round(r["start_station_m"], 1), round(r["start_speed_m_s"], 2))  # noqa: E731
    reference = args.compare_to or args.parent
    new, old = rides(args.name, args.eval_starts), rides(reference, args.eval_starts)
    before = {key(r): r for r in old}
    pairs = [(r["legal_progress_m"], before[key(r)]["legal_progress_m"]) for r in new if key(r) in before]
    report = {"model": args.name, "parent": args.parent, "compared_to": reference, "new": summary(new), "parent_summary": summary(old),
              "matched_starts": len(pairs), "farther": sum(a > b for a, b in pairs),
              "shorter": sum(a < b for a, b in pairs), "starts_per_circuit": args.eval_starts,
              "per_circuit": per_circuit(new, old)}
    (REV / f"{args.name}.round.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", required=True, help="the Rev run to collect from and continue")
    parser.add_argument("--name", required=True, help="the new Rev run, e.g. rev-0.8b-s8")
    parser.add_argument("--previous-data", type=Path, help="the dataset whose records are replayed")
    parser.add_argument("--replay-records", type=int, default=750_000)
    parser.add_argument("--containers", type=int, default=10, help="collection L4s")
    parser.add_argument("--episodes", type=int, default=7, help="collection episodes per rider (8 riders per L4)")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16, help="records per GPU per micro-batch")
    parser.add_argument("--accum", type=int, default=1, help="micro-batches per step (a step is batch x accum x gpus records)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--circuit-floor", type=int, default=0,
                        help="aggregate instead of replaying: every DAgger collection so far, each circuit topped "
                             "up to this many records with SAC states (--sac-records sampled)")
    parser.add_argument("--sac-records", type=int, default=1_000_000)
    parser.add_argument("--eval-starts", type=int, default=4, help="starts per circuit; 8 also evaluates the parent")
    parser.add_argument("--data", type=Path, help="train on this existing dataset: no collection and no build "
                                                     "(an ablation on a fixed dataset)")
    parser.add_argument("--size", choices=("0.8b", "4b"), default="0.8b", help="Rev base size (the parent's)")
    parser.add_argument("--lora", type=int, default=0, help="LoRA rank (the parent must have it: rev_widen_lora.py)")
    parser.add_argument("--upload-init", action="store_true", help="upload a local parent to the Modal volume first")
    parser.add_argument("--compare-to", default="", help="the evaluated model to compare with (default the parent)")
    args = parser.parse_args()
    args.tag = args.name.rsplit("-", 1)[-1]
    log = REV / f"{args.name}.round.log"
    began = time.time()
    if args.data:
        data = args.data
        stage(f"training on the existing {data}")
    else:
        if args.eval_starts != 4:  # the parent rides the same starts, alongside the collection
            with ThreadPoolExecutor(2) as pool:
                parent = pool.submit(evaluate, args, log, args.parent)
                collections = collect(args, log)
                parent.result()
        else:
            collections = collect(args, log)
        data = dataset(args, collections, log)
    train(args, data, log)
    evaluate(args, log)
    compare(args)
    stage(f"round {args.name} took {(time.time() - began) / 3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())
