# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""DAgger rounds for Rev, end to end: ride, collect, relabel, retrain, repeat.

Each round, with the current Rev served locally (kev.serve from the pinned Kev checkout
that modal_rev.py trains with):
  1. evaluate: rev_drive.py rides the fixed SAC evaluation starts;
  2. collect: rev_drive.py --collect rides random starts, the SAC teacher taking a
     shrinking --betas share of the steps, and saves every visited state;
  3. data: rev_dataset.py labels the base SAC records and every collection with the
     teacher's action, all of the newest collection plus a sample of the rest;
  4. train: modal_rev.py continues from this round's Rev for one epoch.
Every round's evaluation goes to runs/rev/dagger/rounds.jsonl, and best.json names the
best Rev so far (most completed laps, then fastest lap).

  uv run training/rev_dagger.py --godot GODOT --start rev-0.8b-v1 --rounds 8
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEV = Path.home() / "git_projects/references/jev-models/kev"
TEACHER = ROOT / "runs/sac/sac-v7-lr1e4-1/checkpoints/step_24000071.pt"


def run(cmd, log: Path, cwd=ROOT):
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(str(c) for c in cmd)[:200]}", flush=True)
    with log.open("a") as stream:
        subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


class Server:
    """kev.serve for one Rev checkpoint on the local GPU, for the duration of a with block."""

    def __init__(self, checkpoint: Path, port: int, log: Path):
        self.checkpoint, self.port, self.log = checkpoint, port, log

    def __enter__(self):
        env = {**os.environ, "KEV_DTYPE": "fp32"}  # the 2080 Ti has no native bf16
        self.stream = self.log.open("a")
        self.process = subprocess.Popen(
            ["uv", "run", "--extra", "serve", "python", "-m", "kev.serve", "--run", str(self.checkpoint),
             "--port", str(self.port)], cwd=KEV, env=env, stdout=self.stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2)
                return self
            except OSError:
                if self.process.poll() is not None:
                    raise RuntimeError(f"kev.serve exited; see {self.log}")
                time.sleep(5)
        raise TimeoutError("kev.serve did not come up")

    def __exit__(self, *exc):
        os.killpg(self.process.pid, signal.SIGTERM)
        self.process.wait(timeout=60)
        self.stream.close()


def evaluation(ride_dir: Path) -> dict:
    rows = [json.loads(line) for line in (ride_dir / "eval/eval.jsonl").open()]
    laps = sorted(r["lap_time_s"] for r in rows if r["termination"] == "lap_completed")
    return {"laps": len(laps), "lap_times": laps, "best_lap_s": laps[0] if laps else None,
            "progress_m": [round(r["legal_progress_m"]) for r in rows],
            "terminations": [r["termination"] for r in rows]}


def better(a: dict, b: dict | None) -> bool:
    if b is None:
        return True
    if a["laps"] != b["laps"]:
        return a["laps"] > b["laps"]
    if a["laps"]:
        return a["best_lap_s"] < b["best_lap_s"]
    return sum(a["progress_m"]) > sum(b["progress_m"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--godot", required=True)
    parser.add_argument("--start", required=True, help="the Rev run (runs/rev/NAME) of round 0")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--betas", default="0.5,0.3,0.2,0.1,0")
    parser.add_argument("--collect-episodes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-train-records", type=int, default=60_000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/rev/dagger")
    parser.add_argument("--teacher", type=Path, default=TEACHER)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "rev_dagger.log"
    betas = [float(b) for b in args.betas.split(",")]
    best_path = args.out / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    current, collections = args.start, []
    for k in range(args.rounds + 1):
        checkpoint = ROOT / "runs/rev" / current / "checkpoint"
        beta = betas[min(k, len(betas) - 1)]
        with Server(checkpoint, args.port, log):
            ride = f"{current}-eval"
            run(["uv", "run", "training/rev_drive.py", "--godot", args.godot, "--run-name", ride,
                 "--rev-url", f"http://127.0.0.1:{args.port}"], log)
            result = {"round": k, "rev": current, **evaluation(ROOT / "runs/rev" / ride),
                      "ride": f"runs/rev/{ride}", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with (args.out / "rounds.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            if better(result, best):
                best = result
                best_path.write_text(json.dumps(best, indent=1) + "\n")
            print(f"REV ROUND {k} {current}: {result['laps']}/4 laps {result['lap_times']} "
                  f"progress {result['progress_m']} {result['terminations']} | best {best['rev']} "
                  f"{best['laps']} laps {best['best_lap_s']}", flush=True)
            if k == args.rounds:
                break
            collection = f"collect-r{k + 1}"
            run(["uv", "run", "training/rev_drive.py", "--godot", args.godot, "--run-name", collection,
                 "--rev-url", f"http://127.0.0.1:{args.port}", "--collect", args.collect_episodes,
                 "--workers", args.workers, "--horizon", 60, "--beta", beta, "--teacher", args.teacher,
                 "--seed", k + 1], log)
        collections.append(ROOT / "runs/rev" / collection)
        data = ROOT / f"runs/rev/data-r{k + 1}"
        run(["uv", "run", "training/rev_dataset.py", "--teacher", args.teacher, "--dagger", *collections,
             "--max-train-records", args.max_train_records, "--seed", k + 1, "--out", data], log)
        name = f"{args.start.rsplit('-v', 1)[0]}-r{k + 1}"
        run(["uvx", "--from", "modal==1.5.5", "modal", "run", "training/modal_rev.py::train", "--data", data,
             "--name", name, "--init-from", current, "--epochs", 1, "--lr", args.lr], log)
        current = name


if __name__ == "__main__":
    main()
