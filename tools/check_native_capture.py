"""Verify host capture reconstructs native GPU preprocessing exactly, without weights.

Uses captured game pixels with labeled diagnostic motion/timestamps. This is
not gameplay, training experience, or proof of policy quality.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path


def run(args):
    import torch
    from alpagym_host.config import load_run_config
    from alpagym_runtime.inference.types import NUM_ROUTE_WAYPOINTS
    from alpagym_runtime.inference_capture import HostInputCapture, restore_policy_input
    from alpagym_runtime.policies.alpamayo.policy import AlpamayoPolicy
    from alpagym_runtime.types import (
        CameraImage,
        EgoPose,
        PolicyInput,
        Pose,
        RouteWaypoint,
        Trajectory,
        Vec3,
    )
    from torchvision.io import encode_jpeg

    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["ALPAGYM_INFERENCE_CAPTURE_DIR"] = str(args.output / "capture")
    config = load_run_config(args.config)
    fixture = torch.load(args.model_input, weights_only=True, map_location="cpu")
    frames = fixture["camera_frames"]
    encoded = [bytes(encode_jpeg(frame.contiguous()).tolist()) for frame in frames]
    device = torch.device(args.device)

    def policy(name):
        return AlpamayoPolicy(
            None, name, config.policy, device, torch.bfloat16, seed=1000
        )

    original = policy("capture-equivalence-original")
    capture = HostInputCapture(
        "capture-equivalence-original", config.policy.model.num_context_frames
    )
    count = config.policy.model.num_historical_waypoints
    report = {
        "diagnostic_only": True,
        "motion": "synthetic timestamp/pose sequence",
        "pixels": "actual captured game frames encoded as JPEG",
        "steps": [],
    }
    for step in range(8):
        now = (count + step + 1) * 100000
        poses = tuple(
            EgoPose(i * 100000, Pose(Vec3(i * 0.1, 0, 0)))
            for i in (range(count + 2) if step == 0 else [count + step + 1])
        )
        images = []
        nframes = config.policy.model.num_context_frames if step == 0 else 1
        for camera_index, name in enumerate(config.policy.model.use_cameras):
            for frame_index in range(nframes):
                images.append(
                    CameraImage(
                        name,
                        encoded[(camera_index * 4 + frame_index + step) % len(encoded)],
                        now - (nframes - frame_index - 1) * 100000,
                    )
                )
        value = PolicyInput(
            step,
            now,
            now + 100000,
            tuple(images),
            Trajectory(poses),
            tuple(RouteWaypoint(i * 2, 0) for i in range(NUM_ROUTE_WAYPOINTS)),
            now,
            (),
        )
        identity = capture.record(
            value, original._buffers.ego_history, 1000 + step, config.policy.model
        )
        seed = torch.tensor(1000 + step, device=device)
        expected = original._preprocess(value, seed=seed)
        snapshot = torch.load(identity["path"], weights_only=True, map_location="cpu")
        restored = restore_policy_input(snapshot)
        reconstructed = policy(f"capture-reconstruction-{step}")
        actual = reconstructed._preprocess(restored, seed=seed)
        actual_fields = asdict(actual)
        equality = {
            key: torch.equal(value, actual_fields[key])
            for key, value in asdict(expected).items()
        }
        report["steps"].append(
            {
                "step": step,
                "tensor_equal": equality,
                "buffered_frames": len(restored.camera_images),
                "buffered_poses": len(restored.ego_trajectory.poses),
            }
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not all(equality.values()):
            raise AssertionError(f"Native preprocessing mismatch: {equality}")
        reconstructed.close()
    report["passed"] = True
    report["device"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
