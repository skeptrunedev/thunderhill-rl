"""Bounded native inference diagnostics, never training or a production workaround.

Captured game pixels/history stay unchanged. Route counterfactuals exercise
mixed prompt lengths. Every CPU batch is saved before CUDA transfer. Optional
synchronization attributes asynchronous failures but can hide timing races.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


@contextmanager
def scatter_diagnostics(directory, synchronize):
    """Observe the unchanged tensor operator in this isolated diagnostic process."""
    import torch

    original = torch.Tensor.masked_scatter
    index = 0

    def observed(tensor, mask, source):
        nonlocal index
        index += 1
        prefix = directory / f"scatter-{index:03d}"
        # Copy BEFORE the operation: CUDA failure can poison subsequent copies.
        mask_cpu = mask.detach().cpu()
        selected = int(torch.broadcast_to(mask_cpu, tensor.shape).sum())
        record = {
            "input_shape": list(tensor.shape),
            "input_dtype": str(tensor.dtype),
            "mask_shape": list(mask.shape),
            "source_shape": list(source.shape),
            "source_numel": source.numel(),
            "selected_elements": selected,
            "enough_source": selected <= source.numel(),
            "stack": traceback.format_stack(limit=12),
            "status": "before_operator",
        }
        torch.save(mask_cpu, prefix.with_suffix(".mask.pt"))
        if not tensor.is_floating_point():
            torch.save(
                {"input": tensor.detach().cpu(), "source": source.detach().cpu()},
                prefix.with_suffix(".ids.pt"),
            )
        write_json(prefix.with_suffix(".json"), record)
        if synchronize and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        result = original(tensor, mask, source)
        if synchronize and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        record["status"] = "operator_returned"
        write_json(prefix.with_suffix(".json"), record)
        return result

    torch.Tensor.masked_scatter = observed
    try:
        yield
    finally:
        torch.Tensor.masked_scatter = original


def prepare(payload, size):
    import torch

    rows = []
    labels = []
    for index in range(size):
        row = {
            key: value.clone()
            for key, value in payload.items()
            if isinstance(value, torch.Tensor) and key != "seed"
        }
        label = ("captured", "straight", "left", "right")[index]
        if index:
            route = torch.zeros_like(row["route_xy"])
            x = torch.linspace(0, 80, route.shape[-2])
            route[..., 0] = x
            if index > 1:
                route[..., 1] = (1 if index == 2 else -1) * x.square() / 100
            row["route_xy"] = route
        row["seed"] = torch.tensor(1000 + index, dtype=torch.int64)
        rows.append(row)
        labels.append(label)
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}, labels


def run(args):
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    payload = torch.load(args.model_input, map_location="cpu", weights_only=True)
    report = {
        "diagnostic_only": True,
        "training_eligible": False,
        "actual_failed_campaign_batch": False,
        "fixture_sha256": hashlib.sha256(args.model_input.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "synchronization": args.synchronize,
        "prepare_only": args.prepare_only,
        "limitation": "Instrumented execution changes timing; no reproduction does not rule out a race.",
        "batches": [],
        "status": "preparing",
    }
    write_json(args.output / "report.json", report)
    adapter = config = None
    try:
        if not args.prepare_only:
            from alpagym_alpamayo_r1.bundle import (
                install_alpamayo_r1_runtime_bridge,
                load_inference_model,
            )
            from alpagym_host.config import load_run_config
            from alpagym_runtime.policies.determinism import set_deterministic

            install_alpamayo_r1_runtime_bridge()
            config = load_run_config(args.config)
            config.policy.model.path = str(args.checkpoint)
            if config.policy.inference.sampling.force_determinism:
                set_deterministic()
            adapter = load_inference_model(
                config, torch.device(args.device), torch.bfloat16
            )
            report["sampling"] = asdict(config.policy.inference.sampling)
    except BaseException:
        report.update(status="model_loading_failed", error=traceback.format_exc())
        write_json(args.output / "report.json", report)
        raise
    try:
        for repeat in range(args.repeats):
            for size in args.batch_sizes:
                if time.monotonic() - started >= args.max_seconds:
                    report["status"] = "time_budget_reached_between_calls"
                    return
                directory = args.output / f"repeat-{repeat:03d}-batch-{size}"
                directory.mkdir()
                cpu_batch, labels = prepare(payload, size)
                torch.save(cpu_batch, directory / "model_input.pt")
                record = {
                    "directory": directory.name,
                    "size": size,
                    "routes": labels,
                    "session_ids": [
                        f"diagnostic-{repeat}-{size}-{i}-{label}"
                        for i, label in enumerate(labels)
                    ],
                    "status": "persisted_cpu",
                }
                report["batches"].append(record)
                write_json(directory / "identity.json", record)
                write_json(args.output / "report.json", report)
                if args.prepare_only:
                    continue
                from alpagym_alpamayo_r1.inference_model import (
                    build_alpamayo_r1_forward_inputs,
                )
                from alpagym_alpamayo_r1.tokenize_online import tokenize_for_generation
                from alpagym_runtime.inference.types import BatchedModelInput

                # Tokenize on CPU with the same model/config before any batch CUDA launch.
                raw = build_alpamayo_r1_forward_inputs(
                    BatchedModelInput(**cpu_batch),
                    config.policy.model.num_context_frames,
                )
                raw["image_frames"] = raw["image_frames"].float() / 127.5 - 1
                tokens = tokenize_for_generation(adapter._model, raw)
                torch.save(tokens, directory / "tokenized_cpu.pt")
                ids = tokens["input_ids"]
                history_id = adapter._model.config.traj_token_ids["history"]
                image_id = adapter._model.vlm.config.image_token_id
                record.update(
                    history_placeholders=(ids == history_id).sum(-1).tolist(),
                    image_placeholders=(ids == image_id).sum(-1).tolist(),
                    token_lengths=tokens["attention_mask"].sum(-1).tolist(),
                    tokenized_shapes={
                        key: list(value.shape) for key, value in tokens.items()
                    },
                )
                write_json(directory / "identity.json", record)
                batch = BatchedModelInput(
                    **{key: value.to(args.device) for key, value in cpu_batch.items()}
                )
                call_started = time.monotonic()
                prefill_index = 0

                def record_prefill(module, positional, keywords, destination=directory):
                    nonlocal prefill_index
                    prefill_index += 1
                    metadata = {
                        key: value.detach().cpu()
                        for key, value in keywords.items()
                        if isinstance(value, torch.Tensor)
                        and key
                        in (
                            "input_ids",
                            "attention_mask",
                            "image_grid_thw",
                            "video_grid_thw",
                        )
                    }
                    torch.save(
                        metadata, destination / f"prefill-{prefill_index:03d}.pt"
                    )

                hook = adapter._model.vlm.model.register_forward_pre_hook(
                    record_prefill, with_kwargs=True
                )
                try:
                    with scatter_diagnostics(directory, args.synchronize):
                        output = adapter.sample_trajectories_from_data(
                            batch,
                            config.policy.inference.sampling,
                            return_trace_for_rl=False,
                        )
                        torch.cuda.synchronize(torch.device(args.device))
                finally:
                    hook.remove()
                torch.save(
                    {
                        "pred_xyz": output.pred_xyz.cpu(),
                        "pred_rot": output.pred_rot.cpu(),
                    },
                    directory / "prediction.pt",
                )
                record.update(
                    status="completed",
                    seconds=time.monotonic() - call_started,
                    finite=bool(torch.isfinite(output.pred_xyz).all().item()),
                )
                write_json(directory / "identity.json", record)
        report["status"] = "completed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, choices=range(1, 5), default=[1, 2, 3, 4]
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=600,
        help="Deadline checked between calls, not a subprocess hard kill.",
    )
    parser.add_argument("--synchronize", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.max_seconds <= 1800:
        parser.error("repeats must be 1..10 and max-seconds 1..1800")
    if not args.prepare_only and (args.config is None or args.checkpoint is None):
        parser.error("native inference requires --config and --checkpoint")
    run(args)


if __name__ == "__main__":
    main()
