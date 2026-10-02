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


def eval_groups() -> list[tuple[str, str]]:
    """Four parts of the standard evaluation: five training circuits and one held-out track each."""
    tracks = TRAINING.split(",")
    return [(",".join(tracks[i::4]), HELD_OUT[i]) for i in range(4)]


def eval_parts(model: str) -> list[Path]:
    """The completed evaluation parts of a model: this script's groups, or the older two-part split."""
    parts = []
    for pattern in (f"{model}-eval-g*", f"{model}-eval-[ab]", f"{model}-eval-b2"):
        for path in sorted(REV.glob(pattern)):
            provenance = path / "provenance.json"
            if provenance.exists() and json.loads(provenance.read_text())["status"] == "complete":
                parts.append(path)
    return parts


def collect(args, log: Path) -> list[str]:
    names = [f"collect-{args.tag}-{i}" for i in range(args.containers)]
    if all((REV / name / "provenance.json").exists() for name in names):
        stage(f"collection {args.tag} exists")
        return names
    focus = ",".join(path.name for path in eval_parts(args.parent))
    if not focus:
        raise SystemExit(f"{args.parent} has no completed evaluation to focus the collection on")
    stage(f"collecting on {args.containers} L4s from {args.parent}, focus {focus}")
    code = run([*MODAL, "training/modal_rev_eval.py::collect", "--checkpoint", args.parent, "--name",
                f"collect-{args.tag}", "--focus-from", focus, "--containers", str(args.containers),
                "--episodes", str(args.episodes), "--workers", "8"], log)
    if code:
        raise SystemExit(f"collection failed; see {log}")
    return names


def dataset(args, collections: list[str], log: Path) -> Path:
    out = REV / f"data-{args.tag}"
    if (out / "summary.json").exists():
        stage(f"{out.name} exists")
        return out
    stage(f"labelling {len(collections)} collections and replaying {args.replay_records} records of {args.previous_data}")
    code = run([str(Path.home() / ".local/bin/uv"), "run", "training/rev_dataset.py",
                "--teachers", "runs/sac/specialists.json", "--train-runs", "sac-multitrack-1",
                "--heldout-run", "sac-multitrack-1", "--heldout-shards", "12", "--train-records", "0",
                "--circuit-dropout", "0.2", "--exclude-circuits", EXCLUDED, "--seed", str(args.seed),
                "--replay", str(args.previous_data), "--replay-records", str(args.replay_records),
                "--dagger", *[f"runs/rev/{name}" for name in collections], "--out", str(out)], log)
    if code:
        raise SystemExit(f"dataset build failed; see {log}")
    return out


def train(args, data: Path, log: Path):
    if (REV / args.name / "result.json").exists():
        stage(f"{args.name} is trained")
        return
    stage(f"training {args.name} from {args.parent} on {args.gpus} H100s")
    code = run([*MODAL, "training/modal_rev.py::train", "--data", str(data), "--name", args.name, "--size", "0.8b",
                "--init-from", args.parent, "--shared-prefix", "1", "--epochs", "1", "--lr", str(args.lr),
                "--batch", str(args.batch), "--gpus", str(args.gpus), "--seed", str(args.seed)], log)
    if code or not (REV / args.name / "result.json").exists():
        raise SystemExit(f"training failed; see {log}")


def evaluate(args, log: Path):
    def part(index: int, tracks: str, heldout: str):
        for attempt in range(3):  # Modal can preempt an L4; its restart is refused, so ride again fresh
            name = f"{args.name}-eval-g{index}" + (f"-r{attempt}" if attempt else "")
            if (REV / name / "provenance.json").exists():
                if json.loads((REV / name / "provenance.json").read_text())["status"] == "complete":
                    return name
                continue
            run([*MODAL, "training/modal_rev_eval.py::evaluate", "--checkpoint", args.name, "--name", name,
                 "--tracks", tracks, "--heldout-tracks", heldout, "--starts", "4", "--heldout-starts", "4"], log)
            provenance = REV / name / "provenance.json"
            if provenance.exists() and json.loads(provenance.read_text())["status"] == "complete":
                return name
        raise SystemExit(f"evaluation part {index} failed three times; see {log}")

    stage(f"evaluating {args.name} on 4 L4s")
    with ThreadPoolExecutor(4) as pool:
        names = list(pool.map(lambda group: part(*group), [(i, *g) for i, g in enumerate(eval_groups())]))
    stage(f"evaluated: {', '.join(names)}")


def rides(model: str) -> list[dict]:
    rows = []
    for path in eval_parts(model):
        rows += [json.loads(line) for line in (path / "eval/eval.jsonl").open()]
    return rows


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
    new, old = rides(args.name), rides(args.parent)
    before = {key(r): r for r in old}
    pairs = [(r["legal_progress_m"], before[key(r)]["legal_progress_m"]) for r in new if key(r) in before]
    report = {"model": args.name, "parent": args.parent, "new": summary(new), "parent_summary": summary(old),
              "matched_starts": len(pairs), "farther": sum(a > b for a, b in pairs),
              "shorter": sum(a < b for a, b in pairs)}
    (REV / f"{args.name}.round.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", required=True, help="the Rev run to collect from and continue")
    parser.add_argument("--name", required=True, help="the new Rev run, e.g. rev-0.8b-s8")
    parser.add_argument("--previous-data", type=Path, required=True, help="the dataset whose records are replayed")
    parser.add_argument("--replay-records", type=int, default=750_000)
    parser.add_argument("--containers", type=int, default=10, help="collection L4s")
    parser.add_argument("--episodes", type=int, default=7, help="collection episodes per rider (8 riders per L4)")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16, help="records per GPU per step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.tag = args.name.rsplit("-", 1)[-1]
    log = REV / f"{args.name}.round.log"
    began = time.time()
    collections = collect(args, log)
    data = dataset(args, collections, log)
    train(args, data, log)
    evaluate(args, log)
    compare(args)
    stage(f"round {args.name} took {(time.time() - began) / 3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())
