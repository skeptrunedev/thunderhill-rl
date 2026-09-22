"""Run local FunctionGemma RL directly from its own native tool gameplay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from model_runtime import FUNCTIONGEMMA_SPEC

ROOT = Path(__file__).resolve().parents[1]


def validate_campaign(campaign, generations):
    """Check real updates, without requiring driving success or imitation."""
    if campaign.get("complete") is not True or campaign.get("smoke_only") is not False:
        raise ValueError("Local RL campaign did not finish its requested generations")
    for key in ("model", "revision", "prompt_style"):
        if campaign.get(key) != getattr(FUNCTIONGEMMA_SPEC, key):
            raise ValueError(f"Unexpected FunctionGemma {key}")
    if campaign.get("constrained_sampling_and_training") is not True:
        raise ValueError("Native sampling and training constraints are not verified")
    rows = campaign.get("generations", [])
    if len(rows) != generations:
        raise ValueError("Incorrect completed generation count")
    for row in rows:
        update = row.get("update", {})
        if update.get("optimizer_steps", 0) < 1 or update.get("actions", 0) < 1:
            raise ValueError("Generation has no optimizer update from gameplay")
        if row.get("verification", {}).get("reloaded_logits_match") is not True:
            raise ValueError("Saved RL checkpoint did not pass reload verification")
    if "final_evaluation" not in campaign:
        raise ValueError("Final recorded evaluation is missing")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--godot", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--time-budget-seconds", type=float, default=30)
    p.add_argument("--batch-candidates", default="4")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="online")
    p.add_argument("--wandb-project", default="thunderhill-rl")
    p.add_argument("--wandb-entity", default="skeptrune-org")
    return p


def run(args):
    if args.generations < 1 or args.time_budget_seconds <= 0:
        raise ValueError("Positive generations and simulation budget required")
    candidates = [int(value) for value in args.batch_candidates.split(",")]
    if not candidates or any(value < 3 for value in candidates):
        raise ValueError("Need an evaluation lane and at least two sampled RL rollouts")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "logs"
    logs.mkdir()
    started = time.monotonic()
    report = {"complete": False, "external_gpu_used": False,
              "model": FUNCTIONGEMMA_SPEC.model, "revision": FUNCTIONGEMMA_SPEC.revision,
              "prompt_style": FUNCTIONGEMMA_SPEC.prompt_style,
              "method": "reinforcement learning from the model's own native tool gameplay",
              "supervised_imitation_used": False, "lap_completion_required": False,
              "scope": "Local RL with recorded generations, sampled rollouts, and evaluations"}

    def publish():
        (out / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")

    def event(data):
        data = {"elapsed_seconds": time.monotonic() - started, **data}
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(data) + "\n")
        print(json.dumps(data), flush=True)

    try:
        destination = out / "rl"
        command = [sys.executable, "-u", str(ROOT / "training/train_full_lap_grpo.py"),
                   "--functiongemma", "--godot", args.godot, "--output", str(destination),
                   "--generations", str(args.generations), "--initial-generation", "0",
                   "--time-budget-seconds", str(args.time_budget_seconds),
                   "--batch-candidates", args.batch_candidates, "--temperature", str(args.temperature),
                   "--wandb-mode", args.wandb_mode,
                   "--wandb-project", args.wandb_project, "--wandb-entity", args.wandb_entity]
        log = logs / "rl.log"
        report.update(command=command, log=str(log), campaign_path=str(destination / "campaign.json"))
        publish()
        event({"stage": "rl", "status": "started", "log": str(log)})
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        report["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Local RL exited {result.returncode}; see {log}")
        campaign = json.loads((destination / "campaign.json").read_text())
        report["campaign"] = campaign
        validate_campaign(campaign, args.generations)
        report["complete"] = True
        return report
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["video_jobs"] = len(list(out.rglob("video_jobs/*.json")))
        publish()
        event({"status": "finished", "complete": report["complete"]})


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
