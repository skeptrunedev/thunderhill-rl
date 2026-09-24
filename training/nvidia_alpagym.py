"""Prepare and launch NVIDIA AlpaGym unchanged against the Godot runtime bridge.

Preparation is CPU only. The explicit run command requires the installed upstream
CUDA environment and its two GPU topology. This module contains no RL optimizer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ALPAGYM_REVISION = "972d160eed0e23d388497851504a3a233fec5879"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT.parent / "thunderhill-references" / "alpagym"


def load_upstream(source: Path) -> Path:
    source = source.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != ALPAGYM_REVISION:
        raise ValueError(f"Expected AlpaGym {ALPAGYM_REVISION}, found {head}")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("Upstream AlpaGym has modified tracked files")
    # Preparation imports only the upstream host package, not its CUDA runtime.
    host_src = source / "packages/host/src"
    if str(host_src) not in sys.path:
        sys.path.insert(0, str(host_src))
    return source


def prepare(
    source: Path,
    output: Path,
    model: Path,
    *,
    godot: str,
    max_steps: int = 1,
    rollouts: int = 2,
    episode_seconds: float = 30,
    concurrency: int = 1,
    max_wall_seconds: float = 3600,
) -> Path:
    if max_steps < 1 or rollouts < 2 or concurrency < 1:
        raise ValueError(
            "Require positive steps and concurrency, and at least two rollouts per group"
        )
    if not math.isfinite(episode_seconds) or episode_seconds <= 0:
        raise ValueError("Episode duration must be finite and positive")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("Wall time budget must be finite and positive")
    if abs(round(episode_seconds * 10) - episode_seconds * 10) > 1e-8:
        raise ValueError("Episode duration must use whole 0.1 second samples")
    source = load_upstream(source)
    from hydra import compose, initialize_config_dir
    from alpagym_host.config import register_config_schema, RewardTermConfig
    from alpagym_host.run_artifacts import (
        build_artifact_paths,
        build_run_config,
        write_run_artifacts,
    )
    import yaml

    register_config_schema()
    policy_configs = (
        source / "packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs"
    )
    with initialize_config_dir(
        config_dir=str(source / "packages/host/src/alpagym_host/conf"),
        version_base=None,
    ):
        config = compose(
            config_name="default",
            overrides=[
                f"hydra.searchpath=[file://{policy_configs}]",
                "experiment=alpamayo_1_5_local_2gpu_smoke",
                "transport=nccl",
                "reward=metrics",
            ],
        )
    config.run_root = str(output.resolve())
    config.policy.model.path = str(model.resolve())
    config.policy.model.use_cameras = ["camera_front_wide_120fov"]
    config.expected_valid_steps = math.ceil(episode_seconds / 0.2)
    config.dataset.scene_ids = ["thunderhill-east-standing"]
    # No prerecorded driving or ground truth actions are used for warmup/reward.
    config.alpasim.wizard_args.force_gt_duration_us = 0
    config.alpasim.wizard_args.n_sim_steps = config.expected_valid_steps
    config.reward.terms = [
        RewardTermConfig(kind="metric", metric_name="progress", scale=1.0),
        RewardTermConfig(kind="metric", metric_name="collision_any", scale=-10.0),
        RewardTermConfig(kind="metric", metric_name="offroad", scale=-5.0),
    ]
    config.cosmos.train.max_num_steps = max_steps
    config.cosmos.train.num_epochs = max_steps
    config.cosmos.rollout.n_generation = rollouts
    # Keep the full sibling group, rather than the smoke preset's single episode.
    # NVIDIA owns advantage computation, minibatching and each optimizer step.
    config.cosmos.train.train_batch_per_replica = rollouts
    config.cosmos.rollout.backend = "thunderhill_alpagym_rollout"
    config.cosmos.logging.logger = ["console", "wandb"]
    config.cosmos.logging.project_name = "thunderhill-rl"
    config.cosmos.logging.experiment_name = "thunderhill-alpagym"
    run_config = build_run_config(config, build_artifact_paths(config))
    write_run_artifacts(run_config)
    paths = run_config.artifact_paths
    paths.alpasim_scene_ids_path.write_text(
        yaml.safe_dump({"scene_ids": list(config.dataset.scene_ids)})
    )
    game = {
        "godot_binary": godot,
        "project_path": str(REPO_ROOT / "godot"),
        "episode_seconds": episode_seconds,
        "concurrency": concurrency,
        "recording_root": str(paths.run_dir / "recordings"),
    }
    (paths.run_dir / "game_config.json").write_text(json.dumps(game, indent=2) + "\n")
    (paths.run_dir / "launch_manifest.json").write_text(
        json.dumps(
            {
                "alpagym_source": str(source),
                "alpagym_revision": ALPAGYM_REVISION,
                "status": "prepared_only",
                "max_wall_seconds": max_wall_seconds,
                "gpu_training_verified": False,
                "topology": "local_disaggregated_2gpu",
                "minimum_gpus": 2,
                "upstream_recommended_vram_gb_per_gpu": 40,
                "trainer": "alpagym_runtime.cosmos.trainer.AlpagymGRPOTrainer",
                "simulator": "Godot through the AlpaSim gRPC protocol",
            },
            indent=2,
        )
        + "\n"
    )
    return paths.run_dir


def cosmos_command(source: Path, config) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(source),
        "--package",
        "alpagym-runtime",
        "python",
        "-m",
        "cosmos_rl.launcher.launch_all",
        "--config",
        str(config.artifact_paths.cosmos_config_path),
        "--policy",
        str(config.cosmos.launch.policy_replicas),
        "--rollout",
        str(config.cosmos.launch.rollout_replicas),
        "--num-workers",
        "1",
        "--worker-idx",
        "0",
        "--port",
        str(config.cosmos.launch.controller_port),
        "--log-dir",
        str(config.artifact_paths.log_dir),
        "training.alpagym_worker",
    ]


def stop_process_tree(process: subprocess.Popen) -> None:
    """Stop the launched group and children that created their own sessions."""
    import psutil

    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The parent may have exited while same-group children are still running.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


@contextmanager
def owned_subreaper():
    """Contain daemonized run children without scanning or killing host services.

    This launcher is a dedicated process. Existing children are excluded by PID
    and creation time; only descendants adopted during this scope are cleaned.
    """
    import psutil

    if sys.platform != "linux":
        raise RuntimeError("The NVIDIA launcher requires Linux child subreaper support")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot read child subreaper state")
    parent = psutil.Process()
    baseline = {
        (child.pid, child.create_time()) for child in parent.children(recursive=True)
    }
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    try:
        yield
    finally:
        try:
            # Run process groups have already stopped. Daemons have reparented to
            # this process, including Redis after its original controller exits.
            children = []
            for child in parent.children(recursive=True):
                try:
                    if (child.pid, child.create_time()) not in baseline:
                        children.append(child)
                        child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)
        finally:
            if libc.prctl(36, previous.value, 0, 0, 0) != 0:
                raise OSError(
                    ctypes.get_errno(), "Cannot restore child subreaper state"
                )


def stop_bridge(process: subprocess.Popen) -> None:
    """Let the bridge flush recordings while its Godot child remains alive."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    stop_process_tree(process)


def write_status(run_dir: Path, state: str, **details) -> None:
    status = {
        "state": state,
        "updated_at_unix": time.time(),
        "optimizer_updates_verified": False,
        **details,
    }
    target = run_dir / "run_status.json"
    temporary = target.with_suffix(".json.pending")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(target)


def wait_process(
    process: subprocess.Popen,
    deadline: float,
    companion: subprocess.Popen | None = None,
) -> None:
    while process.poll() is None:
        if companion is not None and companion.poll() is not None:
            raise RuntimeError("Godot bridge exited while Cosmos was running")
        if time.monotonic() >= deadline:
            raise TimeoutError("Prepared wall time budget exhausted")
        time.sleep(0.1)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def run(run_dir: Path) -> None:
    with owned_subreaper():
        _run_owned(run_dir)


def _run_owned(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
    previous_status = run_dir / "run_status.json"
    if previous_status.exists():
        raise ValueError(
            "Run already attempted; prepare a new run to preserve its status and artifacts"
        )
    processes: list[subprocess.Popen] = []
    bridge = None
    budget = float(manifest["max_wall_seconds"])
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Invalid prepared wall time budget")
    deadline = time.monotonic() + budget
    write_status(run_dir, "started", max_wall_seconds=budget)
    try:
        source = load_upstream(Path(manifest["alpagym_source"]))
        from alpagym_host.config import load_run_config
        from alpagym_host.config_validation import validate_run_config
        from alpagym_host.endpoint_registry import FileTopologyRegistry
        from alpagym_host.transport_env import apply_transport_env_vars

        config = load_run_config(run_dir / "resolved_config.yaml")
        validate_run_config(config, "run")
        registry = FileTopologyRegistry(config.artifact_paths.topology_registry_dir)
        if config.artifact_paths.topology_registry_dir.exists():
            raise ValueError("Run already has runtime state; prepare a fresh run")
        runtime = [
            "uv",
            "run",
            "--no-sync",
            "--project",
            str(source),
            "--package",
            "alpagym-runtime",
            "python",
        ]
        preflight = subprocess.Popen(
            runtime
            + [
                "-c",
                "import torch; assert torch.cuda.device_count() >= 2, 'The official topology requires two CUDA GPUs'",
            ],
            start_new_session=True,
        )
        processes.append(preflight)
        wait_process(preflight, deadline)
        apply_transport_env_vars(config.transport)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            map(str, (REPO_ROOT, REPO_ROOT / "training", REPO_ROOT / "tools"))
        ) + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        environment["PYTHONUNBUFFERED"] = "1"
        # Render requested observations, not unused presentation frames between RPCs.
        environment.setdefault("THUNDERHILL_AGENT_OFFSCREEN", "1")
        environment.setdefault("WANDB_ENTITY", "skeptrune-org")
        bridge_command = runtime + [
            "-m",
            "training.alpagym_bridge",
            "--resolved-config",
            str(run_dir / "resolved_config.yaml"),
            "--game-config",
            str(run_dir / "game_config.json"),
        ]
        with (run_dir / "logs/bridge.log").open("w") as log:
            bridge = subprocess.Popen(
                bridge_command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(bridge)
            startup_deadline = min(deadline, time.monotonic() + 60)
            while not registry.list_alpasim_runtimes():
                if bridge.poll() is not None:
                    raise RuntimeError(f"Godot bridge exited; inspect {log.name}")
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError(
                        f"Godot bridge did not publish its endpoint; inspect {log.name}"
                    )
                time.sleep(0.1)
            cosmos = subprocess.Popen(
                cosmos_command(source, config),
                cwd=REPO_ROOT,
                env=environment,
                start_new_session=True,
            )
            processes.append(cosmos)
            wait_process(cosmos, deadline, companion=bridge)
    except BaseException as error:
        write_status(
            run_dir, "failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    else:
        write_status(
            run_dir,
            "completed",
            launcher_exit_code=0,
            note="Launcher exited successfully. Verify saved checkpoints and optimizer metrics separately.",
        )
    finally:
        for process in reversed(processes):
            if process is bridge:
                stop_bridge(process)
            else:
                stop_process_tree(process)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Write configuration only, without GPU allocation or training"
    )
    prep.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prep.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts/alpagym")
    prep.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Official converted Alpamayo checkpoint directory",
    )
    prep.add_argument("--godot", default="godot")
    prep.add_argument("--max-steps", type=int, default=1)
    prep.add_argument(
        "--rollouts",
        type=int,
        default=2,
        help="Sibling rollouts per scene, NVIDIA n_generation",
    )
    prep.add_argument("--episode-seconds", type=float, default=30)
    prep.add_argument("--concurrency", type=int, default=1)
    prep.add_argument("--max-wall-seconds", type=float, default=3600)
    launch = commands.add_parser(
        "run", help="Run the prepared configuration using the installed NVIDIA stack"
    )
    launch.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(
            prepare(
                args.source,
                args.output,
                args.model,
                godot=args.godot,
                max_steps=args.max_steps,
                rollouts=args.rollouts,
                episode_seconds=args.episode_seconds,
                concurrency=args.concurrency,
                max_wall_seconds=args.max_wall_seconds,
            )
        )
    else:
        run(args.run_dir)


if __name__ == "__main__":
    main()
