"""Bounded Gemma 4 E4B control and actual GRPO validation on one cloud GPU."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, set_seed

from model_runtime import GEMMA4_SPEC, load_base, write_spec

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmstart-dataset", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = GEMMA4_SPEC
    started = time.monotonic()
    report = {
        "ok": False,
        "model": spec.model,
        "revision": spec.revision,
        "dtype": spec.dtype,
        "scope": "Control syntax, one real first action GRPO update, adapter reload, and short evaluations; not lap mastery",
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
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required")
        report["gpu"] = torch.cuda.get_device_name(0)
        set_seed(71)
        tokenizer = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision)
        model = get_peft_model(
            load_base(spec),
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0,
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        initial = out / "initial-adapter"
        model.save_pretrained(initial)
        tokenizer.save_pretrained(initial)
        write_spec(initial, spec)
        report["initial_adapter_sha256"] = hashlib.sha256(
            (initial / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        # Evaluate genuine model controls before training. A short horizon is
        # intentionally not a successful lap, but malformed controls fail here.
        before = out / "before"
        stage(
            "evaluate_lap.py",
            [
                "--adapter",
                initial,
                "--godot",
                args.godot,
                "--output",
                before,
                "--generation",
                0,
                "--max-actions",
                10,
            ],
            evaluation=True,
        )
        baseline = json.loads((before / "summary.json").read_text())
        if baseline["actions"] != 10 or baseline["reason"] != "episode_tick_limit":
            raise RuntimeError(
                "Initial model did not complete ten valid control calls; inspect before recording"
            )
        training_adapter = initial
        evaluation_actions = 30
        start_generation = 0
        if args.warmstart_dataset:
            warm = out / "warmstart"
            stage(
                "train_lap_sft.py",
                ["--adapter", initial, "--dataset", args.warmstart_dataset,
                 "--output", warm, "--steps", 100, "--batch", 2,
                 "--learning-rate", 1e-4],
            )
            training_adapter = warm / "adapter"
            evaluation_actions = 300
            start_generation = 1
            warm_eval = out / "warmstart-evaluation"
            stage(
                "evaluate_lap.py",
                ["--adapter", training_adapter, "--godot", args.godot,
                 "--output", warm_eval, "--generation", start_generation,
                 "--max-actions", evaluation_actions],
                evaluation=True,
            )
            warm_result = json.loads((warm_eval / "summary.json").read_text())
            report["warmstart_evaluation"] = {
                k: warm_result[k] for k in ("actions", "reason", "sim_seconds")
            }
            if warm_result["actions"] != evaluation_actions or warm_result["reason"] != "episode_tick_limit":
                raise RuntimeError("Warm started model failed the longer control evaluation")
        trained = out / "grpo"
        stage(
            "train_lap_grpo.py",
            [
                "--adapter",
                training_adapter,
                "--godot",
                args.godot,
                "--output",
                trained,
                "--prefix-actions",
                1,
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
                start_generation,
            ],
        )
        learning = json.loads((trained / "summary.json").read_text())
        if not all(
            learning[k] is True
            for k in ("ok", "reloaded_logits_match", "all_recording_audits_passed")
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
                start_generation + 1,
                "--max-actions",
                evaluation_actions,
            ],
            evaluation=True,
        )
        final = json.loads((after / "summary.json").read_text())
        if final["reason"] == "invalid_model_action":
            raise RuntimeError("Trained model produced malformed controls")
        if args.warmstart_dataset and (final["actions"] != evaluation_actions or final["reason"] != "episode_tick_limit"):
            raise RuntimeError("RL model failed the longer control evaluation")
        report.update(
            ok=True,
            optimizer_steps=learning["optimizer_steps"],
            sampled_rollouts=learning["sampled_rollouts"],
            max_adapter_delta=learning["max_adapter_delta"],
            final_adapter_sha256=learning["current_adapter_sha256"],
            before_actions=baseline["actions"],
            after_actions=final["actions"],
            after_reason=final["reason"],
            video_jobs=len(list(out.rglob("video_jobs/*.json"))),
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
