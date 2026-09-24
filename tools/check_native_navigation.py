"""Inspect native navigation conditioning using a captured real game model input.

This runs the actual NVIDIA tokenizer and replay input builder on CPU. It does
not load policy weights or run an optimizer. SDE trace placeholders are labelled
as a verification fixture and cannot be used as a training rollout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from alpagym_alpamayo_r1.inference_model import (
    AlpamayoR1InferenceModel,
    build_alpamayo_r1_forward_inputs,
)
from alpagym_alpamayo_r1.navigation import navigation_instruction
from alpagym_alpamayo_r1.tokenize_online import tokenize_for_generation
from alpagym_runtime.inference.types import BatchedModelInput, ModelInput
from training.native_source import record_navigation_checkpoint


def inspect(
    model_input_path: Path, checkpoint: Path, release_config: Path, output: Path
):
    provenance = record_navigation_checkpoint(checkpoint, release_config)
    model_input = ModelInput.from_payload(
        torch.load(model_input_path, map_location="cpu", weights_only=True)
    )
    config = json.loads((checkpoint / "config.json").read_text())
    config["legacy_inference_image_input_format"] = True
    model = SimpleNamespace(config=SimpleNamespace(**config))
    raw = build_alpamayo_r1_forward_inputs(BatchedModelInput.stack([model_input]), 4)

    def tokenize(inputs):
        data = {**inputs, "image_frames": inputs["image_frames"].float() / 127.5 - 1.0}
        return tokenize_for_generation(model, data)

    rollout = tokenize(raw)
    # Real captured observation, fixture-only SDE payload. No optimization occurs.
    replay = SimpleNamespace(
        payload_schema="alpamayo_r1.trajectory.navigation.v2",
        payload_schema_version=2,
        model_family="alpamayo_r1",
        payload={
            "model_input": asdict(model_input),
            "samples_list": torch.zeros(1),
            "timesteps": torch.zeros(1),
        },
        old_logprob=torch.tensor(0.0),
    )
    trainer_inputs, _ = AlpamayoR1InferenceModel.build_trainer_model_inputs(replay, 4)
    rebuilt = tokenize(
        {key: value.unsqueeze(0) for key, value in trainer_inputs.items()}
    )
    equality = {key: torch.equal(value, rebuilt[key]) for key, value in rollout.items()}
    if not all(equality.values()):
        raise AssertionError(f"Rollout/replay conditioning differs: {equality}")

    # Hold real images and ego history fixed, alter only the route. These are
    # explicit counterfactuals, never saved as model training experience.
    counterfactuals = {}
    for direction, sign in (("left", 1), ("right", -1)):
        route = torch.zeros_like(raw["route_xy"])
        count = route.shape[-2]
        x = torch.linspace(0, 80, count)
        route[..., 0] = x
        route[..., 1] = sign * x.square() / 100
        tokens = tokenize({**raw, "route_xy": route})
        text = model._alpagym_packer.tokenizer.decode(tokens["input_ids"][0])
        instruction = navigation_instruction(route[0])
        if f"<|route_start|>{instruction}<|route_end|>" not in text:
            raise AssertionError("Native navigation token section is absent")
        counterfactuals[direction] = dict(
            instruction=instruction,
            input_ids_sha256=hashlib.sha256(
                tokens["input_ids"].numpy().tobytes()
            ).hexdigest(),
            pixels_unchanged=torch.equal(
                tokens["pixel_values"], rollout["pixel_values"]
            ),
        )
    if (
        counterfactuals["left"]["input_ids_sha256"]
        == counterfactuals["right"]["input_ids_sha256"]
    ):
        raise AssertionError(
            "Changing only road direction did not change model input tokens"
        )
    if not all(row["pixels_unchanged"] for row in counterfactuals.values()):
        raise AssertionError("Route-only counterfactual changed image processing")
    report = dict(
        verification_only=True,
        training_eligible=False,
        real_game_model_input=str(model_input_path.resolve()),
        checkpoint=str(checkpoint.resolve()),
        navigation_provenance=provenance,
        navigation_instruction=navigation_instruction(raw["route_xy"][0]),
        rollout_replay_tensor_equality=equality,
        route_only_counterfactuals=counterfactuals,
        limitation="Verifies actual input token consumption, not learned navigation obedience or GPU optimization.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.model_input, args.checkpoint, args.release_config, args.output)


if __name__ == "__main__":
    main()
