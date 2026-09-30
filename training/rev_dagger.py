# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""DAgger rounds for Rev on every circuit, end to end: ride, collect, relabel, retrain, repeat.

Each round, with the current Rev served locally (rev_serve.py: kev.serve from the pinned
Kev checkout that modal_rev.py trains with, batched through CUDA graphs under a GPU cap):
  1. evaluate: rev_drive.py rides --eval-starts fixed starts on every --tracks circuit and
     --heldout-starts on each held-out test track (evaluated, never collected on or trained on);
  2. collect: rev_drive.py --collect rides random starts spread across the circuits, weighted
     toward the ones Rev fails on (track_weights), a --focus-fraction of them starting before
     the last evaluation's and collection's failures on that circuit, the circuit's SAC
     teacher taking a shrinking --betas share of the steps;
  3. data: rev_dataset.py labels the base SAC records and every collection with each
     circuit's current teacher (its specialist once runs/sac/specialists.json has one, else
     the general policy), all of the newest collection plus a circuit-balanced sample of the
     rest; every round rebuilds its labels, so a new specialist relabels its circuit's states;
  4. train: the selected backend continues from this round's Rev for one epoch.
     --train-backend local uses rev_train.py on this GPU, without any Modal calls.
Every round's evaluation goes to --out/rounds.jsonl, per circuit next to its teacher's and
the general policy's best evaluation lap, and best.json names the best Rev so far (most
laps on the training circuits, then most progress).

Rerunning the same command resumes: each stage whose output is complete is skipped and a
half-written one is archived and redone, so a supervisor can restart it after any
failure. No new round starts after --until.

  uv run training/rev_dagger.py --godot GODOT --start rev-0.8b-r8 --tag m --prior-collections runs/rev/collect-r*
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from rev_gpu import gpu_slot

ROOT = Path(__file__).resolve().parents[1]
KEV = Path.home() / "git_projects/references/jev-models/kev"
REGISTRY = ROOT / "runs/sac/specialists.json"
GENERAL_RUN = ROOT / "runs/sac/sac-multitrack-1"
GENERAL_CHECKPOINT = "checkpoints/step_40672729.pt"


def run(cmd, log: Path, cwd=ROOT):
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(str(c) for c in cmd)[:240]}", flush=True)
    with log.open("a") as stream:
        subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


class Server:
    """rev_serve.py for one Rev checkpoint on the local GPU, for the duration of a with block."""

    def __init__(self, checkpoint: Path, port: int, log: Path, gpu_memory_gb: float, max_batch: int = 8,
                 temperature: float | None = None):
        self.checkpoint, self.port, self.log, self.gpu_memory_gb = checkpoint, port, log, gpu_memory_gb
        self.max_batch = max_batch
        self.temperature = temperature

    def __enter__(self):
        self.gpu = gpu_slot()
        self.gpu.__enter__()
        try:
            return self.start()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def start(self):
        self.began = time.time()
        self.stream = self.log.open("a")
        self.process = subprocess.Popen(
            ["uv", "run", "--extra", "serve", "python", str(ROOT / "training/rev_serve.py"),
             "--run", str(self.checkpoint), "--port", str(self.port), "--gpu-memory-gb", str(self.gpu_memory_gb),
             "--max-batch", str(self.max_batch),
             *(["--temperature", str(self.temperature)] if self.temperature is not None else [])],
            cwd=KEV, stdout=self.stream, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(180):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2) as response:
                    card = json.load(response)["models"][0]
                if Path(card["run"]).resolve() != self.checkpoint.resolve():
                    raise RuntimeError(f"port {self.port} serves {card['run']}, expected {self.checkpoint}")
                self.began = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] REV SERVER UP {self.checkpoint.parent.name}", flush=True)
                return self
            except OSError:
                if self.process.poll() is not None:
                    raise RuntimeError(f"rev_serve exited; see {self.log}")
                time.sleep(5)
        raise TimeoutError("rev_serve did not come up")

    def __exit__(self, *exc):
        try:
            process = getattr(self, "process", None)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        finally:
            if hasattr(self, "stream"):
                self.stream.close()
            self.gpu.__exit__(*exc)
        # The local GPU is shared with SAC training: it is held only while Rev rides.
        print(f"[{time.strftime('%H:%M:%S')}] REV SERVER DOWN after {time.time() - self.began:.0f} s", flush=True)


def done(path: Path, marker: str) -> bool:
    """True when a stage is complete; preserve partial artifacts before a redo."""
    if (path / marker).exists():
        return True
    if path.exists():
        archived = path.with_name(f"{path.name}.incomplete-{time.time_ns()}")
        path.rename(archived)
        print(f"preserved incomplete stage at {archived}", flush=True)
    return False


def references() -> dict:
    """track -> {teacher: its registry best lap, general: the general policy's best evaluation
    lap}: what each circuit's Rev laps are measured against."""
    general = {}
    for line in (GENERAL_RUN / "eval/eval.jsonl").open():
        row = json.loads(line)
        if row["checkpoint"] == GENERAL_CHECKPOINT and row["lap_time_s"] is not None:
            general[row["track"]] = min(general.get(row["track"], 1e9), round(row["lap_time_s"], 3))
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    tracks = sorted(set(general) | set(registry))
    return {t: {"teacher": registry.get(t, {}).get("best_lap_s", general.get(t)),
                "specialist": t in registry, "general": general.get(t)} for t in tracks}


def evaluation(ride_dir: Path) -> dict:
    summary = json.loads((ride_dir / "summary.json").read_text())
    refs = references()
    for track, entry in summary.items():
        entry.update({f"{k}_lap_s": v for k, v in refs.get(track, {}).items() if k != "specialist"},
                     specialist=refs.get(track, {}).get("specialist", False))
    trained = {t: e for t, e in summary.items() if not e["held_out"]}
    return {"laps": sum(e["laps"] for e in trained.values()), "rides": sum(e["rides"] for e in trained.values()),
            "circuits_lapped": sum(e["laps"] > 0 for e in trained.values()),
            "progress_m": sum(sum(e["progress_m"]) for e in trained.values()),
            "heldout_laps": sum(e["laps"] for e in summary.values() if e["held_out"]),
            "tracks": summary}


def better(a: dict, b: dict | None) -> bool:
    if b is None:
        return True
    if a["laps"] != b["laps"]:
        return a["laps"] > b["laps"]
    return a["progress_m"] > b["progress_m"]


def track_weights(ride_dir: Path, collection: Path | None) -> dict:
    """Each circuit's share of the next collection: 0.5, plus its failed share of the
    evaluation's rides, plus its offroad-or-fall share of the last collection's episodes."""
    summary = json.loads((ride_dir / "summary.json").read_text())
    weights = {t: 0.5 + sum(end != "lap_completed" for end in e["terminations"]) / e["rides"]
               for t, e in summary.items() if not e["held_out"]}
    if collection is not None and (collection / "episodes.jsonl").exists():
        episodes = [json.loads(line) for line in (collection / "episodes.jsonl").open()]
        for track in weights:
            ends = [e["termination"] for e in episodes if e.get("track", "thunderhill-east") == track]
            if ends:
                weights[track] += sum(end in ("offroad", "fall") for end in ends) / len(ends)
    return {t: round(w, 3) for t, w in weights.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--godot", required=True)
    parser.add_argument("--start", required=True, help="the Rev run (runs/rev/NAME) of round 0")
    parser.add_argument("--tag", default="m", help="round names: collect-TAG3, data-TAG3, STEM-TAG3")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--betas", default="0.5,0.3,0.2,0.1,0")
    parser.add_argument("--tracks", default="all")
    parser.add_argument("--heldout-tracks", default="portimao,laguna-seca")
    parser.add_argument("--eval-starts", type=int, default=1)
    parser.add_argument("--heldout-starts", type=int, default=2)
    parser.add_argument("--collect-episodes", type=int, default=3, help="per worker")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--train-runs", nargs="+", default=["sac-multitrack-1"])
    parser.add_argument("--train-backend", choices=("modal", "local"), default="modal")
    parser.add_argument("--train-batch", type=int, default=2, help="local training microbatch")
    parser.add_argument("--train-accum", type=int, default=4, help="local training gradient accumulation")
    parser.add_argument("--train-checkpointing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--train-shared-prefix", type=int, choices=(0, 1), default=1)
    parser.add_argument("--heldout-run", default="sac-multitrack-1")
    parser.add_argument("--heldout-shards", type=int, default=12)
    parser.add_argument("--train-records", type=int, default=88_000, help="base SAC records, all circuits")
    parser.add_argument("--max-train-records", type=int, default=80_000)
    parser.add_argument("--circuit-dropout", type=float, default=0.2)
    parser.add_argument("--exclude-circuits", default="", help="circuits omitted from every dataset partition")
    parser.add_argument("--baseline-eval", type=Path, help="reuse a completed evaluation of --start")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/rev/dagger-multi")
    parser.add_argument("--teachers", type=Path, default=REGISTRY)
    parser.add_argument("--until", help="local time (YYYY-MM-DDTHH:MM) after which no new round starts")
    parser.add_argument("--round-offset", type=int, default=0,
                        help="continue a previous block: rounds are numbered from offset + 1")
    parser.add_argument("--prior-collections", type=Path, nargs="*", default=[],
                        help="earlier blocks' collections, kept in the aggregated DAgger data")
    parser.add_argument("--focus-fraction", type=float, default=0.5,
                        help="share of collection rides that restart before the last eval's and "
                             "collection's failures on their circuit")
    parser.add_argument("--gpu-memory-gb", type=float, default=4.5, help="cap on the local Rev server")
    parser.add_argument("--serve-max-batch", type=int, default=8,
                        help="requests per local CUDA graph pass; reduce for larger models on the same GPU")
    parser.add_argument("--serve-temperature", type=float, help="override checkpoint temperature for riding")
    args = parser.parse_args()
    if args.serve_max_batch < 1:
        parser.error("--serve-max-batch must be at least 1")
    deadline = time.mktime(time.strptime(args.until, "%Y-%m-%dT%H:%M")) if args.until else None
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out / "config.json"
    if not config.exists():
        config.write_text(json.dumps(vars(args), default=str, indent=1) + "\n")
    log = args.out / "rev_dagger.log"
    betas = [float(b) for b in args.betas.split(",")]
    best_path = args.out / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    evaluated = {json.loads(line)["rev"] for line in (args.out / "rounds.jsonl").open()} \
        if (args.out / "rounds.jsonl").exists() else set()
    runs_dir = ROOT / "runs/rev"
    current, collections = args.start, [Path(c).resolve() for c in args.prior_collections]
    stem = re.sub(r"-[a-z]\d+$", "", args.start)
    for k in range(args.rounds + 1):
        beta = betas[min(k, len(betas) - 1)]
        number = args.round_offset + k + 1
        ride, collection = f"{current}-eval-{args.tag}", f"collect-{args.tag}{number}"
        ride_dir = args.baseline_eval.resolve() if k == 0 and args.baseline_eval else runs_dir / ride
        if k == 0 and args.baseline_eval and not (ride_dir / "summary.json").exists():
            parser.error("--baseline-eval must contain a completed summary.json")
        last = k == args.rounds or (deadline is not None and time.time() > deadline)
        need_eval = current not in evaluated
        need_collect = not last and not done(runs_dir / collection, "episodes.jsonl")
        if need_eval or need_collect:
            with Server(runs_dir / current / "checkpoint", args.port, log, args.gpu_memory_gb,
                        args.serve_max_batch, args.serve_temperature):
                if need_eval:
                    if not done(ride_dir, "summary.json"):
                        run(["uv", "run", "training/rev_drive.py", "--godot", args.godot, "--run-name", ride,
                             "--rev-url", f"http://127.0.0.1:{args.port}", "--tracks", args.tracks,
                             "--starts", args.eval_starts, "--heldout-tracks", args.heldout_tracks,
                             "--heldout-starts", args.heldout_starts, "--workers", args.workers], log)
                    result = {"round": args.round_offset + k, "rev": current, **evaluation(ride_dir),
                              "ride": str(ride_dir.relative_to(ROOT)), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    with (args.out / "rounds.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    evaluated.add(current)
                    if better(result, best):
                        best = result
                        best_path.write_text(json.dumps(best, indent=1) + "\n")
                    print(f"REV ROUND {result['round']} {current}: {result['laps']}/{result['rides']} laps on "
                          f"{result['circuits_lapped']} circuits, held out {result['heldout_laps']} laps "
                          f"| best {best['rev']} {best['laps']} laps", flush=True)
                if need_collect and deadline is not None and time.time() > deadline:
                    need_collect, last = False, True  # the deadline passed during this round's evaluation
                if need_collect:
                    previous = collections[-1] if collections else None
                    weights = track_weights(ride_dir, previous)
                    run(["uv", "run", "training/rev_drive.py", "--godot", args.godot, "--run-name", collection,
                         "--rev-url", f"http://127.0.0.1:{args.port}", "--tracks", args.tracks,
                         "--collect", args.collect_episodes, "--workers", args.workers, "--horizon", 60,
                         "--track-weights", json.dumps(weights), "--beta", beta, "--teachers", args.teachers,
                         "--seed", number, "--focus-fraction", args.focus_fraction,
                         "--focus-from", ride_dir, *([previous] if previous else [])], log)
        if last:
            print(f"REV DAGGER DONE after round {k}; best {best['rev']} {best['laps']} laps", flush=True)
            break
        collections.append(runs_dir / collection)
        data = runs_dir / f"data-{args.tag}{number}"
        if not done(data, "summary.json"):
            run(["uv", "run", "training/rev_dataset.py", "--teachers", args.teachers,
                 "--train-runs", *args.train_runs, "--heldout-run", args.heldout_run,
                 "--heldout-shards", args.heldout_shards, "--train-records", args.train_records,
                 "--circuit-dropout", args.circuit_dropout, "--dagger", *collections,
                 "--exclude-circuits", args.exclude_circuits,
                 "--max-train-records", args.max_train_records, "--seed", number, "--out", data], log)
        name = f"{stem}-{args.tag}{number}"
        if not done(runs_dir / name, "result.json"):
            if args.train_backend == "local":
                run([KEV / ".venv/bin/python", ROOT / "training/rev_train.py", "--data", data,
                     "--name", name, "--init-from", current, "--epochs", 1, "--lr", args.lr,
                     "--batch", args.train_batch, "--accum", args.train_accum,
                     "--checkpointing", args.train_checkpointing, "--shared-prefix", args.train_shared_prefix,
                     "--seed", number], log)
            else:
                run(["uvx", "--from", "modal==1.5.5", "modal", "run", "training/modal_rev.py::train", "--data", data,
                     "--name", name, "--init-from", current, "--epochs", 1, "--lr", args.lr], log)
        current = name


if __name__ == "__main__":
    main()
