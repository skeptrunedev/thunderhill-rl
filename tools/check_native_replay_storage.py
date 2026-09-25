"""Check host replay value equality and retained CUDA bytes on captured game input.

Trace values are a serialization fixture, not gameplay, a policy, or training.
Use --device cuda:0 to verify the real CUDA retention bound. This does not replace
an actual two GPU NCCL transport and optimizer verification.
"""

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import torch


def leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for item in fields(value):
            yield from leaves(getattr(value, item.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decisions", type=int, default=64)
    args = parser.parse_args()
    if args.decisions < 2:
        raise ValueError("Retain at least two replay decisions")
    from alpagym_alpamayo_r1.inference_model import AlpamayoR1InferenceModel
    from alpagym_runtime.host_replay import host_snapshot
    from alpagym_runtime.inference.inference_engine import InferenceEngine
    from alpagym_runtime.inference.types import ModelInput, ModelOutput
    from alpagym_runtime.replay import ActionSelection

    device = torch.device(args.device)
    raw = torch.load(args.input, map_location="cpu", weights_only=True)
    model_input = ModelInput(
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw.items()
        }
    )
    trace = ModelOutput(
        pred_xyz=torch.zeros((1, 1, 64, 3), device=device),
        pred_rot=torch.eye(3, device=device).expand(1, 1, 64, 3, 3),
        logprob=torch.tensor([[-1.25]], device=device),
        extra={
            "samples_list": torch.arange(256, device=device)
            .reshape(1, 2, 64, 2)
            .float(),
            "timesteps": torch.tensor([0.0, 0.5, 1.0], device=device),
        },
    )
    selection = ActionSelection(0, 0)
    # Adapter packing does not access weights. Compare the original native
    # serializer on CPU with the runtime's new device-placement boundary.
    adapter = AlpamayoR1InferenceModel(model=None, num_context_frames=4)
    reference = adapter.build_policy_replay_data(
        host_snapshot(model_input), host_snapshot(trace), selection
    )
    engine = object.__new__(InferenceEngine)
    engine._inference_model = adapter
    baseline = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    retained = []
    for _ in range(args.decisions):
        replay = engine.build_policy_replay_data(model_input, trace, selection)
        actual, expected = list(leaves(replay)), list(leaves(reference))
        assert len(actual) == len(expected)
        for observed, original in zip(actual, expected):
            assert observed.device.type == "cpu"
            assert observed.dtype == original.dtype and observed.shape == original.shape
            assert torch.equal(observed, original)
        retained.append(replay)
    raw_bytes = sum(t.numel() * t.element_size() for t in leaves(model_input))
    report = dict(
        device=str(device),
        retained_decisions=len(retained),
        input_bytes_per_decision=raw_bytes,
        exact_tensor_values_shapes_dtypes=True,
        all_retained_replay_tensors_on_cpu=True,
        fixture_scope="actual captured sensors with constructed trace serialization values, not RL",
        cuda_retention_verified=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        growth = torch.cuda.memory_allocated(device) - baseline
        peak_growth = torch.cuda.max_memory_allocated(device) - baseline
        assert growth <= raw_bytes * 2, (growth, raw_bytes)
        assert peak_growth <= raw_bytes * 2, (peak_growth, raw_bytes)
        report.update(
            cuda_retention_verified=True,
            cuda_live_growth_bytes=growth,
            cuda_peak_growth_bytes=peak_growth,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
