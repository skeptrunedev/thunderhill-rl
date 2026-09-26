"""Bounded real gameplay, native optimization, export comparison, and fresh reload.

This orchestrates NVIDIA entrypoints. It contains no optimizer or training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from training.nvidia_alpagym import NVIDIA_CLRL_GROUP_SIZE


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def verify_scatter_diagnostics(run_dir: Path) -> dict:
    """Require actual completed native inference operators, never just an env flag."""
    import hashlib

    root = run_dir / "inference-capture"
    if list(root.glob("*/scatter-failure.json")):
        raise ValueError("Scatter diagnostic recorded a failed operator")
    receipts = []
    for path in sorted(root.glob("*/scatter-last-completed.json")):
        marker_bytes = (path.parent / "scatter-mode.json").read_bytes()
        receipt_bytes = path.read_bytes()
        marker = json.loads(marker_bytes)
        receipt = json.loads(receipt_bytes)
        if (
            marker.get("enabled") is not True
            or marker.get("mode") != "synchronous_scatter_attribution_v1"
            or marker.get("performance_representative") is not False
            or receipt.get("stage") != "operator_completed"
            or receipt.get("thread") != "alpagym-infer"
            or receipt.get("operator") not in {
                "aten.masked_scatter.default", "aten.masked_scatter_.default"
            }
            or receipt.get("enough_source") is not True
            or not 0 <= receipt["selected_elements"] <= receipt["source_numel"]
            or not receipt.get("dispatch", {}).get("ordered_requests")
        ):
            raise ValueError(f"Invalid native scatter diagnostic receipt: {path}")
        receipts.append({
            "path": str(path.relative_to(run_dir)),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        })
    if not receipts:
        raise ValueError("No completed native scatter diagnostic operators were observed")
    return {"passed": True, "mode": "synchronous_scatter_attribution_v1",
            "performance_representative": False, "completed_receipts": receipts}


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


def evaluate(
    run_dir: Path, model: Path, destination: Path, version: int,
    *, episodes: int = 2, rpc_timeout_seconds: float = 900,
) -> None:
    """Freshly load native weights and drive the actual game with identical seeds."""
    if episodes < 1:
        raise ValueError("Evaluation requires at least one episode")
    if not math.isfinite(rpc_timeout_seconds) or rpc_timeout_seconds <= 0:
        raise ValueError("Evaluation RPC timeout must be finite and positive")
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from training.native_recipe_source import verify_installed

    verify_installed(apply_patch=True)

    import grpc
    from alpagym_alpamayo_r1.bundle import install_alpamayo_r1_runtime_bridge
    from alpagym_host.config import load_run_config
    from alpagym_runtime.alpasim.driver_server import EgodriverGrpcServicer
    from alpagym_runtime.policies.factory import (
        build_inference_engine,
        build_policy_factory,
    )

    from training.alpagym_bridge import (
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
                    # Matched evaluation: episode i always uses the same start scene.
                    rollout_specs=[
                        runtime.RolloutSpec(
                            scenario_id=service.scene_ids[i % len(service.scene_ids)],
                            nr_rollouts=1,
                            session_uuids=[f"evaluation_seed_{i}"],
                        )
                        for i in range(episodes)
                    ],
                    n_concurrent_per_driver=1,
                ),
                timeout=rpc_timeout_seconds,
            )
        rows = [
            {
                "session_uuid": row.rollout_uuid,
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
                "scenario_ids": list(service.scene_ids),
                "initial_speed_m_s": game.get("initial_speed_m_s", 0.0),
                "randomized_start": game.get("randomized_start"),
                "episodes": rows,
            },
        )
        if len(rows) != episodes or not all(row["success"] for row in rows):
            raise RuntimeError("Fresh native checkpoint evaluation failed")
    finally:
        runtime_server.stop(0).wait()
        server.stop(0).wait()
        engine.shutdown()
        thread.join(timeout=30)


def validate(
    source: Path, model: Path, output: Path, *, seconds: float, budget: float,
    initial_speed_m_s: float = 0.0, randomized_starts: int = 0, start_seed: int = 0,
    rollouts: int = NVIDIA_CLRL_GROUP_SIZE,
    resume_run_dir: Path | None = None,
    concurrency: int = 1, replay_fixture: Path | None = None,
    scatter_diagnostics: bool = False,
) -> dict:
    from training.native_source import record_navigation_checkpoint, verify_source
    from training.nvidia_alpagym import prepare

    verify_source(source, apply_patch=True)
    started = time.monotonic()
    attempt = ""
    if resume_run_dir is None:
        run_dir = prepare(
            source,
            output,
            model,
            godot="godot",
            max_steps=2,
            rollouts=rollouts,
            episode_seconds=seconds,
            concurrency=concurrency,
            max_wall_seconds=budget / 2,
            max_video_seconds=budget / 4,
            model_name="Alpamayo 1.5 native RL",
            initial_speed_m_s=initial_speed_m_s,
            randomized_start_count=randomized_starts,
            randomized_start_seed=start_seed,
            scatter_diagnostics=scatter_diagnostics,
        )
        report = {
            "run_dir": str(run_dir),
            "state": "running",
            "optimizer_updates_verified": False,
            "checkpoint_reload_verified": False,
            "initial_speed_m_s": initial_speed_m_s,
            "randomized_starts": randomized_starts,
            "start_seed": start_seed,
            "rollouts": rollouts,
            "scatter_diagnostics": scatter_diagnostics,
        }
        write_json(run_dir / "validation.json", report)

    else:
        import uuid

        run_dir = resume_run_dir.resolve(strict=True)
        manifest = json.loads((run_dir / "launch_manifest.json").read_text())
        if manifest.get("scatter_diagnostics", False) != scatter_diagnostics:
            raise ValueError("Resume must preserve the original diagnostic mode")
        if manifest["native_source_patch"] != verify_source(source):
            raise ValueError("Resume must use the original reviewed native source patches")
        previous = json.loads((run_dir / "validation.json").read_text())
        if previous.get("state") != "failed":
            raise ValueError("Resume requires a previously failed validation")
        attempt = "resume-" + uuid.uuid4().hex
        archive = run_dir / attempt
        archive.mkdir()
        for name in ("validation.json", "run_status.json"):
            (archive / name).write_bytes((run_dir / name).read_bytes())
        report = {
            "run_dir": str(run_dir), "state": "running",
            "optimizer_updates_verified": False, "checkpoint_reload_verified": False,
            "initial_speed_m_s": previous["initial_speed_m_s"],
            "scatter_diagnostics": scatter_diagnostics,
            "resumed_from": str(archive / "validation.json"),
            "resume_attempt": attempt, "retrained": False,
            "verification_source_identity": json.loads((output / "source_identity.json").read_text()),
        }

    def launch(arguments, logfile, *, module=True):
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Native validation wall budget exhausted")
        from training.nvidia_alpagym import (
            owned_subreaper,
            stop_process_tree,
            wait_process,
        )

        with owned_subreaper(), (run_dir / "logs" / (f"{attempt}-{logfile}" if attempt else logfile)).open("w") as log:
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
        import importlib.metadata

        from training.native_cosmos_source import verify_installed as verify_cosmos

        verify_cosmos(apply_patch=True)
        cosmos_source = importlib.metadata.distribution("cosmos-rl").locate_file("")
        launch([
            "tools.check_native_weight_stream", "--upstream", str(source),
            "--cosmos-source", str(cosmos_source),
            "--output", str(run_dir / "weight-stream-verification.json"),
        ], "weight-stream-verification.log")
        handoff = json.loads((run_dir / "weight-stream-verification.json").read_text())
        if (handoff.get("passed") is not True
                or handoff.get("receive_lifetime", {}).get("passed") is not True):
            raise RuntimeError("Native weight transfer stream ordering failed qualification")
        report["weight_stream_verification"] = handoff
        if resume_run_dir is None:
            if replay_fixture is not None:
                import hashlib

                report["replay_fixture_sha256"] = hashlib.sha256(replay_fixture.read_bytes()).hexdigest()
                launch([
                    "tools.check_native_replay_storage", "--input", str(replay_fixture),
                    "--output", str(run_dir / "replay-storage-verification.json"),
                    "--device", "cuda:0", "--decisions", "64",
                ], "replay-storage-verification.log")
                report["replay_storage_verification"] = json.loads((run_dir / "replay-storage-verification.json").read_text())
                launch([
                    "tools.check_native_capture", "--model-input", str(replay_fixture),
                    "--config", str(run_dir / "resolved_config.yaml"),
                    "--output", str(run_dir / "capture-verification"), "--device", "cuda:0",
                ], "capture-verification.log")
                capture = json.loads((run_dir / "capture-verification/report.json").read_text())
                if capture.get("passed") is not True:
                    raise RuntimeError("Native observation capture did not reconstruct exactly")
                report["capture_verification"] = capture
                launch([
                    "tools.check_native_padding_replay", "--model-input", str(replay_fixture),
                    "--config", str(run_dir / "resolved_config.yaml"),
                    "--checkpoint", str(model),
                    "--output", str(run_dir / "padding-replay-verification"), "--device", "cuda:0",
                ], "padding-replay-verification.log")
                padding_replay = json.loads((run_dir / "padding-replay-verification/report.json").read_text())
                if (padding_replay.get("status") != "completed"
                        or padding_replay.get("mixed_batch_singleton_within_tolerance") is not True):
                    raise RuntimeError("Native padded batch replay failed qualification")
                report["padding_replay_verification"] = padding_replay
            launch(
                [
                    "tools.check_camera",
                    "--godot",
                    "godot",
                    "--offscreen",
                    "--rendering-method",
                    "mobile",
                    "--require-hardware",
                    "--initial-capture-timeout", "120", "--trace-render",
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
        if os.environ.get("ALPAGYM_SKIP_ALL_PADDING_MINIBATCHES") == "1":
            padding_reports = list(run_dir.glob("cosmos/**/padding_skip_verification.json"))
            if len(padding_reports) != 1:
                raise RuntimeError("Missing unique native padding state equivalence proof")
            report["padding_skip_verification"] = json.loads(padding_reports[0].read_text())
            if not report["padding_skip_verification"]["passed"]:
                raise RuntimeError("Native padding state equivalence proof failed")
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
        conversion_root = run_dir / attempt if attempt else run_dir
        inference = conversion_root / "converted-inference"
        reloaded = conversion_root / "reloaded-checkpoint"
        report["reloaded_checkpoint"] = str(reloaded)
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
        padding_skipped = 0
        for logfile in (run_dir / "logs").rglob("*.log"):
            log_text = logfile.read_text(errors="replace")
            padding_skipped += log_text.count("AlpaGym skipped all-padding minibatch")
            gradients.extend(
                float(x)
                for x in re.findall(
                    r"AlpaGym trainer optimizer step[^\n]*grad_norm=([0-9.eE+\-]+)",
                    log_text,
                )
            )
        report.update(
            padding_minibatches_skipped=padding_skipped,
            export=str(trained),
            weights_changed=comparison["weights_changed"],
            nonzero_gradient_optimizer_steps=sum(
                math.isfinite(g) and g > 0 for g in gradients
            ),
        )
        if (
            not comparison["weights_changed"]
            or not comparison["frozen_vlm_unchanged"]
            or not report["nonzero_gradient_optimizer_steps"]
        ):
            raise RuntimeError("No verified nonzero native optimizer update")
        if scatter_diagnostics:
            report["scatter_diagnostics_verification"] = verify_scatter_diagnostics(run_dir)
        report["optimizer_updates_verified"] = True
        launch(
            [
                "training.native_validation",
                "evaluate",
                str(run_dir),
                str(reloaded),
                str(run_dir / "recordings" / (attempt or "reloaded")),
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
                    *(["--retry-failed"] if attempt else []),
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
        report.update(state="failed", error_type=type(error).__name__, error=str(error))
        write_json(run_dir / "validation.json", report)
        print(json.dumps(report), flush=True)
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
                    *(["--retry-failed"] if attempt else []),
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
    ev.add_argument("--episodes", type=int, default=2)
    ev.add_argument("--rpc-timeout-seconds", type=float, default=900)
    run = sub.add_parser("run")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seconds", type=float, default=5)
    run.add_argument("--budget", type=float, default=3300)
    run.add_argument("--initial-speed-m-s", type=float, default=0.0)
    run.add_argument("--randomized-starts", type=int, default=0)
    run.add_argument("--start-seed", type=int, default=0)
    run.add_argument("--rollouts", type=int, default=NVIDIA_CLRL_GROUP_SIZE)
    run.add_argument("--resume-run-dir", type=Path)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--replay-fixture", type=Path)
    run.add_argument("--scatter-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(
            args.run_dir, args.model, args.destination, args.version,
            episodes=args.episodes,
            rpc_timeout_seconds=args.rpc_timeout_seconds,
        )
    else:
        print(
            json.dumps(
                validate(
                    args.source,
                    args.model,
                    args.output,
                    seconds=args.seconds,
                    budget=args.budget,
                    initial_speed_m_s=args.initial_speed_m_s,
                    randomized_starts=args.randomized_starts,
                    start_seed=args.start_seed,
                    rollouts=args.rollouts,
                    resume_run_dir=args.resume_run_dir,
                    concurrency=args.concurrency,
                    replay_fixture=args.replay_fixture,
                    scatter_diagnostics=args.scatter_diagnostics,
                )
            )
        )


if __name__ == "__main__":
    main()
