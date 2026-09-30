"""Continue a Rev checkpoint on the local GPU using its pinned Kev trainer.

Run with the Kev checkout's .venv/bin/python. The checkpoint keeps temperature 1
because Rev decodes soft action probabilities into continuous controls. Temperature
fitting is reported separately and never changes those controls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rev_gpu import gpu_slot

ROOT = Path(__file__).resolve().parents[1]
KEV = Path.home() / "git_projects/references/jev-models/kev"
RUNS = ROOT / "runs/rev"
PARTITIONS = ("train", "calibration", "development")


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_name(value: str) -> str:
    if not value or Path(value).name != value or value in (".", ".."):
        raise argparse.ArgumentTypeError("use a Rev run name, without a directory path")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", type=run_name, required=True)
    parser.add_argument("--init-from", type=run_name, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--checkpointing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--shared-prefix", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="bound actual optimizer updates for a real training smoke (0 = all epochs)")
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.accum) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch, accum and lr must be positive and lr must be finite")
    if args.max_steps < 0:
        parser.error("max-steps must be nonnegative")
    return args


def train(out: Path, checkpoint: Path, command: list[str]):
    stage("training: " + " ".join(command[2:]))
    with (out / "train.log").open("w", encoding="utf-8") as log, subprocess.Popen(
            command, cwd=KEV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", env={**os.environ, "PYTHONUNBUFFERED": "1",
                                           "TOKENIZERS_PARALLELISM": "false"}) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("ep", "saved", "device", "dropped", "delta")) or "Error" in line or "Traceback" in line:
                print(line.rstrip(), flush=True)
    if process.returncode:
        raise RuntimeError(f"kev.train failed with exit {process.returncode}; see {out / 'train.log'}")
    metrics = json.loads((checkpoint / "training_metrics.json").read_text(encoding="utf-8"))
    if metrics["optimizer_steps"] < 1:
        raise RuntimeError("kev.train saved no optimizer updates")
    return metrics


def score(out: Path, checkpoint: Path):
    import torch
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions, read_meta, write_meta
    from kev.data import load_records
    from kev.metrics import fit_temperature, grouped_metrics
    from kev.predictors import LocalPredictor

    calibration = load_records(out / "data/calibration.jsonl")
    development = load_records(out / "data/development.jsonl")
    stage(f"fitting a report temperature on {len(calibration)} and scoring {len(development)} development records")
    predictor = LocalPredictor(str(checkpoint), "cuda",
                               LoadOptions(dtype=torch.float32, merge=False, temperature=1.0))
    _, calibration_rows = evaluate_records(calibration, predictor, out / "calibration")
    fitted = fit_temperature(calibration_rows, aggregation="micro")
    report, rows = evaluate_records(development, predictor, out / "development", fitted)
    fit = {"rows": "calibration.jsonl", "n": len(calibration), "value": fitted,
           "method": "min NLL, micro, kev.metrics.fit_temperature", "applied": False,
           "reason": "Rev decodes soft action probabilities into continuous controls; preserve temperature 1.0"}
    meta = read_meta(str(checkpoint))
    meta.temperature = 1.0
    meta.extra["temperature_fit"] = fit
    write_meta(str(checkpoint), meta)
    clean = [row for row in rows if row["variant"] == "clean"]
    return fitted, fit, {"raw": report["clean"], "calibrated": report["calibrated_clean"],
                         "per_question": grouped_metrics(clean, "question", 1.0),
                         "calibrated_per_question": grouped_metrics(clean, "question", fitted)}


def main():
    args = parse_args()
    python = KEV / ".venv/bin/python"
    if not python.is_file() or Path(sys.prefix).resolve() != (KEV / ".venv").resolve():
        raise RuntimeError(f"run this command with {python}")
    parent = RUNS / args.init_from
    previous = json.loads((parent / "result.json").read_text(encoding="utf-8"))
    init = parent / "checkpoint"
    if not (init / "head.pt").is_file() or previous.get("model") != "rev-0.8b":
        raise ValueError("init-from must be a completed local Rev 0.8B run")
    kev_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=KEV, text=True).strip()
    if kev_ref != previous["kev_ref"]:
        raise ValueError(f"Kev checkout is {kev_ref}, but the parent trained with {previous['kev_ref']}")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "kev"], cwd=KEV, check=True)
    source = args.data.resolve(strict=True)
    inputs = {part: source / f"{part}.jsonl" for part in PARTITIONS}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("local Rev training requires CUDA")
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / "data").mkdir()
    data = {}
    for part, path in inputs.items():
        target = out / "data" / path.name
        shutil.copyfile(path, target)
        with target.open(encoding="utf-8") as stream:
            count = sum(bool(line.strip()) for line in stream)
        if not count:
            raise ValueError(f"empty {part} partition: {path}")
        data[part] = {"source": str(path), "sha256": digest(target), "records": count}
    (out / "data_source.txt").write_text(str(source) + "\n", encoding="utf-8")
    write_json(out / "data_source.json", data)
    checkpoint = out / "checkpoint"
    recipe = {"epochs": args.epochs, "lr": args.lr, "batch": args.batch, "accum": args.accum,
              "checkpointing": args.checkpointing, "shared_prefix": args.shared_prefix,
              "dtype": "fp32", "weights_dtype": "fp32", "max_steps": args.max_steps}
    command = [str(python), "-m", "kev.train", "--data", str(out / "data/train.jsonl"),
               "--base", previous["base"], "--base_revision", previous["base_revision"],
               "--init_from", str(init), "--device", "cuda", "--seed", str(args.seed), "--out", str(checkpoint)]
    for key, value in recipe.items():
        command.extend([f"--{key}", str(value)])
    write_json(out / "run_config.json", {"args": vars(args) | {"data": str(source)}, "recipe": recipe,
               "kev_ref": kev_ref, "base": previous["base"], "base_revision": previous["base_revision"],
               "init_from": args.init_from, "parent_result_sha256": digest(parent / "result.json"),
               "data": data, "command": command, "training_backend": "local"})
    started = time.time()
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "waiting_for_gpu"}
    write_json(out / "run_status.json", status)
    stage("waiting for the local GPU")
    try:
        with gpu_slot():
            status["status"] = "training"
            write_json(out / "run_status.json", status)
            metrics = train(out, checkpoint, command)
            status["status"] = "scoring"
            write_json(out / "run_status.json", status)
            fitted, fit, development = score(out, checkpoint)
            result = {"name": args.name, "model": previous["model"], "base": previous["base"],
                      "base_revision": previous["base_revision"], "init_from": args.init_from,
                      "kev_ref": kev_ref, "recipe": recipe, "seed": args.seed,
                      "records": {part: info["records"] for part, info in data.items()},
                      "temperature": 1.0, "calibration_temperature": fitted, "temperature_fit": fit,
                      "development": development, "optimizer_steps": metrics["optimizer_steps"],
                      "training_metrics": metrics, "data_sha256": {part: info["sha256"] for part, info in data.items()},
                      "wall_seconds": round(time.time() - started), "gpu": torch.cuda.get_device_name(0),
                      "training_backend": "local"}
            write_json(out / "result.json", result)
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "run_status.json", status)
    stage(f"saved {out}: {result['optimizer_steps']} optimizer updates, temperature 1.0")


if __name__ == "__main__":
    main()
