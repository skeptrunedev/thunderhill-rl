"""Resume one bounded GRPO update from a verified Gemma4 supervised warm start."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from model_runtime import GEMMA4_SPEC, read_spec

ROOT = Path(__file__).resolve().parents[1]


def validate_evaluation(summary, adapter_hash):
    if (
        summary["actions"] != 300
        or summary["reason"] != "episode_tick_limit"
        or summary["offtrack_ticks"] != 0
        or summary["recording_provenance_verified"] is not True
        or summary["adapter_sha256"] != adapter_hash
        or summary["model"] != GEMMA4_SPEC.model
        or summary["revision"] != GEMMA4_SPEC.revision
    ):
        raise ValueError(
            "Expected 300 verified on track controls from the matching Gemma4 adapter"
        )


def validate_source(source):
    adapter = source / "warmstart" / "adapter"
    if read_spec(adapter) != GEMMA4_SPEC:
        raise ValueError("Source adapter is not the pinned Gemma4 model")
    warm = json.loads((source / "warmstart" / "summary.json").read_text())
    if not all(
        warm[key] is True
        for key in ("completion_masks_verified", "reloaded_logits_match")
    ):
        raise ValueError("Source supervised training evidence failed validation")
    adapter_hash = hashlib.sha256(
        (adapter / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    if warm["adapter_sha256"] != adapter_hash:
        raise ValueError("Source supervised adapter checksum mismatch")
    evaluation = json.loads(
        (source / "warmstart-evaluation" / "summary.json").read_text()
    )
    validate_evaluation(evaluation, adapter_hash)
    decisions = source / "warmstart-evaluation" / "decisions.jsonl"
    if not decisions.is_file():
        raise ValueError("Source evaluation decisions are missing")
    return adapter, decisions, adapter_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "ok": False,
        "model": GEMMA4_SPEC.model,
        "revision": GEMMA4_SPEC.revision,
        "source": str(source),
        "scope": "One first action GRPO update from verified model driving prefix and 300 control evaluation; not lap mastery",
        "stages": [],
    }

    def stage(script, arguments, *, evaluation=False):
        command = [
            sys.executable,
            "-u",
            str(ROOT / "training" / script),
            *map(str, arguments),
        ]
        print(json.dumps({"stage_command": command}), flush=True)
        result = subprocess.run(command, cwd=ROOT)
        report["stages"].append({"command": command, "returncode": result.returncode})
        if result.returncode != 0 and not (evaluation and result.returncode == 1):
            raise RuntimeError(f"{script} failed with {result.returncode}")

    try:
        adapter, decisions, adapter_hash = validate_source(source)
        report.update(
            initial_adapter_sha256=adapter_hash,
            prefix_decisions_sha256=hashlib.sha256(decisions.read_bytes()).hexdigest(),
            prefix_actions=200,
        )
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required")
        report["gpu"] = torch.cuda.get_device_name(0)
        trained = out / "grpo"
        stage(
            "train_lap_grpo.py",
            [
                "--adapter",
                adapter,
                "--godot",
                args.godot,
                "--output",
                trained,
                "--prefix-decisions",
                decisions,
                "--prefix-actions",
                200,
                "--continuation-actions",
                9,
                "--steps",
                1,
                "--num-generations",
                16,
                "--temperature",
                1.0,
                "--learning-rate",
                1e-5,
                "--start-generation",
                1,
            ],
        )
        learning = json.loads((trained / "summary.json").read_text())
        if not all(
            learning[key] is True
            for key in ("ok", "reloaded_logits_match", "all_recording_audits_passed")
        ):
            raise RuntimeError("Training evidence failed validation")
        after = out / "after"
        stage(
            "evaluate_lap.py",
            [
                "--adapter",
                trained / "adapter",
                "--godot",
                args.godot,
                "--output",
                after,
                "--generation",
                2,
                "--max-actions",
                300,
            ],
            evaluation=True,
        )
        final = json.loads((after / "summary.json").read_text())
        validate_evaluation(final, learning["current_adapter_sha256"])
        report.update(
            ok=True,
            optimizer_steps=learning["optimizer_steps"],
            sampled_rollouts=learning["sampled_rollouts"],
            max_adapter_delta=learning["max_adapter_delta"],
            final_adapter_sha256=learning["current_adapter_sha256"],
            after_actions=final["actions"],
            after_reason=final["reason"],
            after_offtrack_ticks=final["offtrack_ticks"],
            after_sim_seconds=final["sim_seconds"],
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["video_jobs"] = len(list(out.rglob("video_jobs/*.json")))
        (out / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
