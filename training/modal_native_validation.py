"""Migrate existing controls to Gemma4 native tools, then validate real trajectory RL."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from model_runtime import GEMMA4_NATIVE_SPEC, GEMMA4_SPEC, read_spec

ROOT = Path(__file__).resolve().parents[1]


def validate_source(adapter):
    if read_spec(adapter) not in (GEMMA4_SPEC, GEMMA4_NATIVE_SPEC):
        raise ValueError("Native migration requires the pinned Gemma4 adapter")
    return hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()


def validate_migration(directory, source_hash):
    summary = json.loads((directory / "summary.json").read_text())
    if read_spec(directory / "adapter") != GEMMA4_NATIVE_SPEC:
        raise ValueError("Migration output did not declare native Gemma4 tools")
    for key in ("completion_masks_verified", "reloaded_logits_match", "supervision_controls_preserved"):
        if summary.get(key) is not True:
            raise ValueError(f"Migration evidence failed: {key}")
    if summary.get("initial_adapter_sha256") != source_hash:
        raise ValueError("Migration source adapter checksum mismatch")
    actual = hashlib.sha256((directory / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()
    if summary.get("adapter_sha256") != actual:
        raise ValueError("Migration output adapter checksum mismatch")
    return summary


def validate_campaign(campaign):
    if campaign.get("complete") is not True or campaign.get("smoke_only") is not True:
        raise ValueError("Native tool trajectory smoke did not complete")
    if (campaign.get("prompt_style") != GEMMA4_NATIVE_SPEC.prompt_style
            or campaign.get("native_stop_token_id") != 50
            or campaign.get("constrained_sampling_and_training") is not True):
        raise ValueError("Native tool protocol or constrained training evidence is missing")
    generations = campaign.get("generations", [])
    if not generations:
        raise ValueError("Native tool trajectory smoke has no completed update")
    collections = [generation["collection"] for generation in generations]
    collections.append(campaign["final_evaluation"])
    for collection in collections:
        for episode in [collection["evaluation"], *collection["rollouts"]]:
            if episode.get("reason") == "invalid_model_action":
                raise ValueError("Native tool smoke contains an invalid model action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--source", type=Path, required=True, help="Existing adapter directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/opt/thunderhill/warmstart-data"))
    args = parser.parse_args()
    source = args.source.resolve()
    source_hash = validate_source(source)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"ok": False, "source": str(source), "source_adapter_sha256": source_hash,
              "scope": "Native tool format SFT migration and short complete trajectory RL validation; not lap mastery",
              "stages": []}

    def stage(script, arguments):
        command = [sys.executable, "-u", str(ROOT / "training" / script), *map(str, arguments)]
        print(json.dumps({"stage_command": command}), flush=True)
        result = subprocess.run(command, cwd=ROOT)
        report["stages"].append({"command": command, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{script} failed with {result.returncode}")

    try:
        migration = out / "native-migration"
        stage("train_lap_sft.py", ["--adapter", source, "--native-tools", "--dataset", args.dataset,
                                  "--output", migration, "--steps", 100, "--batch", 2,
                                  "--learning-rate", 1e-4])
        report["migration"] = validate_migration(migration, source_hash)
        smoke = out / "native-smoke"
        stage("train_full_lap_grpo.py", ["--adapter", migration / "adapter", "--godot", args.godot,
                                       "--output", smoke, "--smoke", "--initial-generation", 0])
        campaign = json.loads((smoke / "campaign.json").read_text())
        validate_campaign(campaign)
        report.update(ok=True, campaign=campaign)
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
