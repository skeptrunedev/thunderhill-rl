"""Bounded real gameplay, native optimization, export comparison, and fresh reload.

This orchestrates NVIDIA entrypoints. It contains no optimizer or training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def compare_exports(base: Path, trained: Path) -> dict:
    """Compare every exported tensor, quantizing the baseline to export precision."""
    import torch
    from safetensors import safe_open

    def index(directory):
        result = {}
        for shard in directory.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():  # noqa: SIM118 (safetensors handle is not a mapping)
                    if name in result:
                        raise ValueError(f"Duplicate tensor {name}")
                    result[name] = shard
        if not result:
            raise ValueError(f"No weights in {directory}")
        return result

    before, after = index(base), index(trained)
    if before.keys() != after.keys():
        raise ValueError("Export tensor keys differ from the converted baseline")
    changed, checked = [], 0
    for name in sorted(before):
        with (
            safe_open(before[name], framework="pt", device="cpu") as a,
            safe_open(after[name], framework="pt", device="cpu") as b,
        ):
            new = b.get_tensor(name)
            old = a.get_tensor(name).to(new.dtype)
            if old.shape != new.shape or not torch.isfinite(new).all():
                raise ValueError(f"Invalid exported tensor {name}")
            checked += new.numel()
            if not torch.equal(old, new):
                delta = (new.float() - old.float()).abs()
                changed.append(
                    {
                        "name": name,
                        "changed_elements": int(torch.count_nonzero(delta)),
                        "max_abs_change": float(delta.max()),
                    }
                )
    return {
        "tensors_checked": len(after),
        "elements_checked": checked,
        "changed_tensors": changed,
        "weights_changed": bool(changed),
    }


def evaluate(run_dir: Path, model: Path, destination: Path, version: int) -> None:
    """Freshly load native weights and drive the actual game with identical seeds."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import grpc
    from alpagym_alpamayo_r1.bundle import install_alpamayo_r1_runtime_bridge
    from alpagym_host.config import load_run_config
    from alpagym_runtime.alpasim.driver_server import EgodriverGrpcServicer
    from alpagym_runtime.policies.factory import (
        build_inference_engine,
        build_policy_factory,
    )

    from training.alpagym_bridge import (
        SCENE_ID,
        GodotRuntime,
        driver_grpc,
        runtime,
        runtime_grpc,
    )

    destination.mkdir(parents=True, exist_ok=False)
    install_alpamayo_r1_runtime_bridge()
    config = load_run_config(run_dir / "resolved_config.yaml")
    config.policy.model.path = str(model)
    config.policy.model.device = "cuda:0"
    config.policy.inference.return_trace_for_rl = False
    engine = build_inference_engine(config)
    thread = threading.Thread(target=engine.run_loop, daemon=True)
    thread.start()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    driver_grpc.add_EgodriverServiceServicer_to_server(
        EgodriverGrpcServicer(build_policy_factory(config, engine)), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    game = json.loads((run_dir / "game_config.json").read_text())
    game.update(
        recording_root=str(destination),
        model_path=str(model),
        model_name=f"Alpamayo 1.5 evaluation {model.name}",
    )
    service = GodotRuntime(
        game,
        destination / "unused",
        identity_provider=lambda _: {
            "policy_version": version,
            "policy_version_source": "evaluation_identity_not_live_cosmos_version",
            "batch_id": "fixed-seed-evaluation",
            "is_validation": True,
            "active": True,
        },
    )
    runtime_server = grpc.server(ThreadPoolExecutor(max_workers=4))
    runtime_grpc.add_RuntimeServiceServicer_to_server(service, runtime_server)
    runtime_port = runtime_server.add_insecure_port("127.0.0.1:0")
    runtime_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{runtime_port}") as channel:
            reply = runtime_grpc.RuntimeServiceStub(channel).simulate(
                runtime.SimulationRequest(
                    available_drivers=[
                        runtime.SimulationRequest.DriverAddress(
                            ip="127.0.0.1", port=port
                        )
                    ],
                    rollout_specs=[
                        runtime.RolloutSpec(
                            scenario_id=SCENE_ID,
                            nr_rollouts=2,
                            session_uuids=["evaluation_seed_0", "evaluation_seed_1"],
                        )
                    ],
                    n_concurrent_per_driver=1,
                ),
                timeout=900,
            )
        rows = [
            {
                "success": row.success,
                "error": row.error,
                "metrics": dict(row.aggregated_metrics),
            }
            for row in reply.rollout_returns
        ]
        write_json(
            destination / "evaluation.json",
            {
                "model": str(model),
                "policy_version": version,
                "policy_version_source": "evaluation_identity_not_live_cosmos_version",
                "episodes": rows,
            },
        )
        if len(rows) != 2 or not all(row["success"] for row in rows):
            raise RuntimeError("Fresh native checkpoint evaluation failed")
    finally:
        runtime_server.stop(0).wait()
        server.stop(0).wait()
        engine.shutdown()
        thread.join(timeout=30)


def validate(
    source: Path, model: Path, output: Path, *, seconds: float, budget: float
) -> dict:
    from training.native_source import record_navigation_checkpoint, verify_source
    from training.nvidia_alpagym import prepare

    verify_source(source, apply_patch=True)
    started = time.monotonic()
    run_dir = prepare(
        source,
        output,
        model,
        godot="godot",
        max_steps=2,
        rollouts=4,
        episode_seconds=seconds,
        concurrency=1,
        max_wall_seconds=budget / 2,
        max_video_seconds=budget / 4,
        model_name="Alpamayo 1.5 native RL",
    )
    report = {
        "run_dir": str(run_dir),
        "state": "running",
        "optimizer_updates_verified": False,
        "checkpoint_reload_verified": False,
    }
    write_json(run_dir / "validation.json", report)

    def launch(arguments, logfile, *, module=True):
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Native validation wall budget exhausted")
        from training.nvidia_alpagym import (
            owned_subreaper,
            stop_process_tree,
            wait_process,
        )

        with owned_subreaper(), (run_dir / "logs" / logfile).open("w") as log:
            process = subprocess.Popen(
                [sys.executable, *(["-m"] if module else []), *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_process(process, time.monotonic() + remaining)
            finally:
                stop_process_tree(process)

    try:
        launch(
            [
                "tools.check_camera",
                "--godot",
                "godot",
                "--offscreen",
                "--rendering-method",
                "mobile",
                "--require-hardware",
                "--output",
                str(run_dir / "camera-preflight"),
            ],
            "camera-preflight.log",
        )
        launch(
            [
                "training.native_validation",
                "evaluate",
                str(run_dir),
                str(model),
                str(run_dir / "recordings/baseline"),
                "0",
            ],
            "baseline-evaluation.log",
        )
        launch(["training.nvidia_alpagym", "run", str(run_dir)], "native-training.log")
        exports = sorted(
            run_dir.glob("cosmos/**/safetensors/step_*"),
            key=lambda p: int(p.name.split("_")[-1]),
        )
        if not exports:
            raise RuntimeError("NVIDIA produced no exported checkpoint")
        trained = exports[-1]
        # Native export flattens VLM keys. Use NVIDIA's official inverse and
        # forward converters to reload in the exact native rollout keyspace.
        scripts = source / "packages/policies/alpamayo_r1/scripts"
        inference = run_dir / "converted-inference"
        reloaded = run_dir / "reloaded-checkpoint"
        launch(
            [
                str(scripts / "convert_alpagym_checkpoint_to_inference.py"),
                "--input",
                str(trained),
                "--output",
                str(inference),
            ],
            "convert-inference.log",
            module=False,
        )
        processor = json.loads((model / "config.json").read_text())["vlm_name_or_path"]
        launch(
            [
                str(scripts / "convert_release_to_alpagym_checkpoint.py"),
                "--input",
                str(inference),
                "--output",
                str(reloaded),
                "--vlm-name-or-path",
                processor,
            ],
            "convert-reload.log",
            module=False,
        )
        # Use the stable mounted snapshot path, not Modal's internal volume path.
        source_release = model.parent / (
            "hub/models--nvidia--Alpamayo-1.5-10B/snapshots/"
            "7aba8293c09993f2e125c6819df05d7fa3e873ea/config.json"
        )
        record_navigation_checkpoint(reloaded, source_release)
        comparison = compare_exports(model, reloaded)
        frozen_changes = [
            row
            for row in comparison["changed_tensors"]
            if row["name"].startswith("vlm.")
        ]
        comparison["frozen_vlm_unchanged"] = not frozen_changes
        write_json(run_dir / "weight_comparison.json", comparison)
        gradients = []
        for logfile in (run_dir / "logs").rglob("*.log"):
            gradients.extend(
                float(x)
                for x in re.findall(
                    r"AlpaGym trainer minibatch[^\n]*grad_norm=([0-9.eE+\-]+)",
                    logfile.read_text(errors="replace"),
                )
            )
        report.update(
            export=str(trained),
            weights_changed=comparison["weights_changed"],
            nonzero_gradient_minibatches=sum(
                math.isfinite(g) and g > 0 for g in gradients
            ),
        )
        if (
            not comparison["weights_changed"]
            or not comparison["frozen_vlm_unchanged"]
            or not report["nonzero_gradient_minibatches"]
        ):
            raise RuntimeError("No verified nonzero native optimizer update")
        report["optimizer_updates_verified"] = True
        launch(
            [
                "training.native_validation",
                "evaluate",
                str(run_dir),
                str(reloaded),
                str(run_dir / "recordings/reloaded"),
                "0",
            ],
            "reload-evaluation.log",
        )
        report["checkpoint_reload_verified"] = True
        # This second rendering pass includes the freshly reloaded evaluations.
        launch(
            [
                "tools.render_video_queue",
                str(run_dir),
                "--godot",
                "godot",
                "--ffmpeg",
                "ffmpeg",
                "--workers",
                "1",
                "--recover-interrupted",
            ],
            "validation-videos.log",
        )
        status_path = run_dir / "run_status.json"
        status = json.loads(status_path.read_text())
        status.update(
            optimizer_updates_verified=True,
            checkpoint_reload_verified=True,
            verification_report="validation.json",
        )
        write_json(status_path, status)
        report["state"] = "completed"
    except BaseException as error:
        # Each child has been reaped by launch() before reaching this handler.
        # Even baseline failures may have flushed attempts worth preserving.
        from training.nvidia_alpagym import write_status

        if not (run_dir / "run_status.json").exists():
            write_status(
                run_dir,
                "failed",
                runtime_stopped=True,
                error="Validation stopped before trainer startup",
            )
        try:
            launch(
                [
                    "tools.render_video_queue",
                    str(run_dir),
                    "--godot",
                    "godot",
                    "--ffmpeg",
                    "ffmpeg",
                    "--workers",
                    "1",
                    "--recover-interrupted",
                ],
                "validation-failure-videos.log",
            )
        except (
            OSError,
            RuntimeError,
            TimeoutError,
            subprocess.SubprocessError,
        ) as video_error:
            report["video_error"] = str(video_error)
        report.update(state="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(run_dir / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("run_dir", type=Path)
    ev.add_argument("model", type=Path)
    ev.add_argument("destination", type=Path)
    ev.add_argument("version", type=int)
    run = sub.add_parser("run")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seconds", type=float, default=5)
    run.add_argument("--budget", type=float, default=3300)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.run_dir, args.model, args.destination, args.version)
    else:
        print(
            json.dumps(
                validate(
                    args.source,
                    args.model,
                    args.output,
                    seconds=args.seconds,
                    budget=args.budget,
                )
            )
        )


if __name__ == "__main__":
    main()
