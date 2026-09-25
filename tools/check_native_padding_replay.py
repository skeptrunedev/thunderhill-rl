"""Qualify native mixed prompt inference and exact trace replay on an existing GPU.

Uses captured game observations and explicit route counterfactuals. These outputs
are diagnostics only, never training data. No optimizer or simulator is run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace


def run(args):
    import torch
    from training.native_recipe_source import verify_installed

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        training=False,
        actual_failed_batch=False,
        checkpoint=str(args.checkpoint),
        device=args.device,
        recipe_padding=verify_installed(apply_patch=True),
    )

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    started = time.monotonic()
    from alpagym_alpamayo_r1.bundle import (
        install_alpamayo_r1_runtime_bridge,
        load_inference_model,
    )
    from alpagym_runtime.inference.types import BatchedModelInput, ModelInput
    from alpagym_runtime.replay import ActionSelection
    from alpagym_host.config import load_run_config
    from alpamayo1_x_rl.models.expert_model.cosmos_wrapper import ExpertModelCosmos
    from tools.check_native_inference import prepare

    install_alpamayo_r1_runtime_bridge()
    config = load_run_config(args.config)
    config.policy.model.path = str(args.checkpoint)
    adapter = load_inference_model(config, torch.device(args.device), torch.bfloat16)
    model = adapter._model
    model.eval()
    report["prefills"] = []

    def record_mask(module, positional, keywords):
        mask = keywords.get("attention_mask")
        if mask is None:
            raise AssertionError("Native prefill discarded its padding mask")
        lengths = mask.detach().sum(-1).cpu().tolist()
        report["prefills"].append(
            dict(mask_shape=list(mask.shape), valid_tokens=lengths)
        )

    hook = model.vlm.model.register_forward_pre_hook(record_mask, with_kwargs=True)
    report["attention_backends"] = dict(
        vlm=model.vlm.config._attn_implementation,
        expert=model.expert.config._attn_implementation,
    )
    report["sampling"] = asdict(config.policy.inference.sampling)
    payload = torch.load(args.model_input, map_location="cpu", weights_only=True)
    cpu_batch, routes = prepare(payload, 4)
    torch.save(cpu_batch, args.output / "model_input.pt")
    report["routes"] = routes
    batch = BatchedModelInput(
        **{key: value.to(args.device) for key, value in cpu_batch.items()}
    )
    try:
        with torch.inference_mode():
            rollout = adapter.sample_trajectories_from_data(
                batch,
                config.policy.inference.sampling,
                return_trace_for_rl=True,
            )
            torch.cuda.synchronize(torch.device(args.device))
            assert (
                torch.isfinite(rollout.pred_xyz).all()
                and torch.isfinite(rollout.pred_rot).all()
            )
            assert torch.isfinite(rollout.logprob).all()
            outputs = rollout.unbind()
            rows, old_logprobs = [], []
            for index, output in enumerate(outputs):
                model_input = ModelInput(
                    **{key: value[index] for key, value in vars(batch).items()}
                )
                replay = adapter.build_policy_replay_data(
                    model_input, output, ActionSelection(0, 0)
                )
                model_inputs, old = adapter.build_trainer_model_inputs(
                    replay, config.policy.model.num_context_frames
                )
                rows.append(model_inputs)
                old_logprobs.append(float(old))
            torch.save(
                {
                    "rows": [
                        {key: value.cpu() for key, value in row.items()} for row in rows
                    ],
                    "old_logprobs": old_logprobs,
                },
                args.output / "selected_replay.pt",
            )
            report.update(rollout_finite=True, selected_rollout_logprobs=old_logprobs)
            save()

            # Call the actual installed native Cosmos forward adapter, using its
            # real expert model, without constructing a second copy of the weights.
            wrapper = SimpleNamespace(expert_model=model)

            def score(indices):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Native padding qualification budget exhausted")
                inputs = {
                    key: torch.stack([rows[i][key] for i in indices]).to(args.device)
                    for key in rows[0]
                }
                result = ExpertModelCosmos.forward(wrapper, **inputs)["log_probs"]
                torch.cuda.synchronize(torch.device(args.device))
                assert torch.isfinite(result).all()
                return result.detach().cpu()

            batched = score(list(range(4)))
            report["batched_replay_logprobs"] = batched.tolist()
            save()
            singles = []
            for index in range(4):
                singles.append(score([index])[0])
                report["singleton_replay_logprobs"] = [
                    float(value) for value in singles
                ]
                save()
            singles = torch.stack(singles)
            differences = (batched - singles).abs()
            tolerance = args.atol + args.rtol * singles.abs()
            report.update(
                replay_finite=True,
                mixed_padding_observed=any(
                    len(set(row["valid_tokens"])) > 1 for row in report["prefills"]
                ),
                mixed_batch_singleton_abs_error=differences.tolist(),
                mixed_batch_singleton_allowed_error=tolerance.tolist(),
                mixed_batch_singleton_within_tolerance=bool(
                    (differences <= tolerance).all()
                ),
                selected_rollout_replay_abs_error=(torch.tensor(old_logprobs) - singles)
                .abs()
                .tolist(),
                tolerance=dict(
                    atol=args.atol,
                    rtol=args.rtol,
                    basis="explicit diagnostic tolerance for bfloat16 model arithmetic",
                ),
                status="completed",
                elapsed_seconds=time.monotonic() - started,
                cuda_scatter_failure_reproduced=False,
            )
            save()
    except BaseException as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.monotonic() - started,
        )
        save()
        raise
    hook.remove()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "config", "model-input", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--atol", type=float, default=0.0078125)
    parser.add_argument("--rtol", type=float, default=0.0078125)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
