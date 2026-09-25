"""Launch NVIDIA AlpaGym with a reviewed navigation input patch and Godot bridge.

Preparation is CPU only. The explicit run command requires the installed upstream
CUDA environment and its two GPU topology. This module contains no RL optimizer.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from training.alpagym_metrics import (
    REWARD_SCALES,
    REWARD_VERSION,
    validate_reward_terms,
)
from training.native_source import validate_navigation_checkpoint, verify_source

ALPAGYM_REVISION = "972d160eed0e23d388497851504a3a233fec5879"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT.parent / "thunderhill-references" / "alpagym"


def load_upstream(source: Path) -> Path:
    source = source.resolve()
    verify_source(source)
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
    initial_speed_m_s: float = 0.0,
    randomized_start_count: int = 0,
    randomized_start_seed: int = 0,
    concurrency: int = 1,
    max_wall_seconds: float = 3600,
    model_name: str | None = None,
    ffmpeg: str = "ffmpeg",
    max_video_seconds: float = 3600,
    scatter_diagnostics: bool = False,
) -> Path:
    from training.episode_config import game_scene_ids
    from training.lap_policy import RoadTelemetry

    if type(randomized_start_count) is not int or randomized_start_count < 0:
        raise ValueError("Randomized start count must be zero (off) or positive")
    initial_conditions = {"initial_speed_m_s": initial_speed_m_s}
    if randomized_start_count:
        # Each start is its own scene, so a GRPO sibling group shares one start.
        initial_conditions["randomized_start"] = dict(
            count=randomized_start_count,
            seed=randomized_start_seed,
            track_sha256=RoadTelemetry().track_sha256,
        )
    scenarios = game_scene_ids(initial_conditions)
    if type(scatter_diagnostics) is not bool:
        raise ValueError("Scatter diagnostics must be an explicit boolean")
    if max_steps < 1 or rollouts < 2 or concurrency < 1:
        raise ValueError(
            "Require positive steps and concurrency, and at least two rollouts per group"
        )
    if not math.isfinite(episode_seconds) or episode_seconds <= 0:
        raise ValueError("Episode duration must be finite and positive")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("Wall time budget must be finite and positive")
    if not math.isfinite(max_video_seconds) or max_video_seconds <= 0:
        raise ValueError("Video time budget must be finite and positive")
    if abs(round(episode_seconds * 10) - episode_seconds * 10) > 1e-8:
        raise ValueError("Episode duration must use whole 0.1 second samples")
    source = load_upstream(source)
    import yaml
    from alpagym_host.config import RewardTermConfig, register_config_schema
    from alpagym_host.run_artifacts import (
        build_artifact_paths,
        build_run_config,
        write_run_artifacts,
    )
    from hydra import compose, initialize_config_dir

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
                "policy=alpamayo_r1",
                "topology=local_disaggregated_2gpu",
                "transport=nccl",
                "transport.nccl_env.NCCL_DEBUG=INFO",
                "reward=metrics",
            ],
        )
    config.run_root = str(output.resolve())
    config.policy.model.path = str(model.resolve())
    config.expected_valid_steps = math.ceil(episode_seconds / 0.2)
    config.dataset.scene_ids = list(scenarios)
    # No prerecorded driving or ground truth actions are used for warmup/reward.
    config.alpasim.wizard_args.force_gt_duration_us = 0
    config.alpasim.wizard_args.control_timestep_us = 200_000
    config.alpasim.wizard_args.n_sim_steps = config.expected_valid_steps
    config.reward.terms = [
        RewardTermConfig(kind="metric", metric_name=name, scale=scale)
        for name, scale in REWARD_SCALES.items()
    ]
    config.cosmos.train.max_num_steps = max_steps
    config.cosmos.train.num_epochs = max_steps
    config.cosmos.rollout.n_generation = rollouts
    # Keep the full sibling group, rather than the smoke preset's single episode.
    # NVIDIA owns advantage computation, minibatching and each optimizer step.
    config.cosmos.train.train_batch_per_replica = rollouts
    config.cosmos.rollout.backend = "thunderhill_alpagym_rollout"
    # Recording identity is published around each explicit generation call.
    config.cosmos.rollout.prefetch_rollout = False
    # Full lap replay is much larger than the short native scene preset.
    # Native on-policy dispatch bounds retained data to one sibling group and
    # waits for its optimizer acknowledgment before issuing the next group.
    config.cosmos.rollout.batch_size = 1
    config.cosmos.train.train_policy.on_policy = True
    config.cosmos.train.train_policy.allowed_outdated_steps = 0
    # Cosmos validates that on_policy implies its default sync interval of one.
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
        "ffmpeg_binary": ffmpeg,
        "project_path": str(REPO_ROOT / "godot"),
        "episode_seconds": episode_seconds,
        **initial_conditions,
        "scenario_ids": scenarios,
        "concurrency": concurrency,
        "recording_root": str(paths.run_dir / "recordings"),
        "model_name": model_name or model.resolve().name,
        "model_path": str(model.resolve()),
        "training_scope": "trajectory_expert_rl_frozen_language_and_vision",
    }
    (paths.run_dir / "game_config.json").write_text(json.dumps(game, indent=2) + "\n")
    (paths.run_dir / "launch_manifest.json").write_text(
        json.dumps(
            {
                "alpagym_source": str(source),
                "run_id": paths.run_dir.name,
                "alpagym_revision": ALPAGYM_REVISION,
                "native_source_patch": verify_source(source),
                "status": "prepared_only",
                "scatter_diagnostics": scatter_diagnostics,
                "max_wall_seconds": max_wall_seconds,
                "max_video_seconds": max_video_seconds,
                "gpu_training_verified": False,
                "reward_version": REWARD_VERSION,
                "topology": "local_disaggregated_2gpu",
                "minimum_gpus": 2,
                "upstream_recommended_vram_gb_per_gpu": 40,
                "trainer": "alpagym_runtime.cosmos.trainer.AlpagymGRPOTrainer",
                "configuration_profile": "upstream_default_alpamayo_r1",
                "training_scope": game["training_scope"],
                "simulator": "Godot through the AlpaSim gRPC protocol",
            },
            indent=2,
        )
        + "\n"
    )
    return paths.run_dir


def validate_runtime_files(source: Path, config, game: dict) -> None:
    """Check local installation and assets before starting distributed workers."""
    policy = config.cosmos.train.train_policy
    if (
        not policy.on_policy
        or policy.allowed_outdated_steps != 0
        or config.cosmos.rollout.batch_size != 1
        or config.cosmos.rollout.prefetch_rollout
    ):
        raise ValueError("Full lap replay requires bounded native on-policy dispatch")
    seconds = float(game["episode_seconds"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Game episode duration must be finite and positive")
    if math.ceil(seconds / 0.2) != config.expected_valid_steps:
        raise ValueError(
            "Game episode duration disagrees with NVIDIA replay step budget"
        )
    if Path(game["model_path"]).resolve() != Path(config.policy.model.path).resolve():
        raise ValueError("Recording model identity disagrees with NVIDIA checkpoint")
    python = source / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(
            f"Official NVIDIA runtime missing at {python}; run tools/setup_alpagym.py"
        )
    for executable in (
        "uv",
        "redis-server",
        game["godot_binary"],
        game["ffmpeg_binary"],
    ):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required executable unavailable: {executable}")
    project = Path(game["project_path"]) / "project.godot"
    if not project.is_file():
        raise FileNotFoundError(f"Godot project missing: {project}")
    checkpoint = Path(config.policy.model.path)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(
                f"Converted NVIDIA checkpoint missing {checkpoint / name}"
            )
    model_config = json.loads((checkpoint / "config.json").read_text())
    if model_config.get("model_type") != "alpamayo_reasoning_vla_expert":
        raise ValueError("Use NVIDIA's converted Alpamayo expert checkpoint")
    validate_navigation_checkpoint(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weights = set(json.loads(index.read_text())["weight_map"].values())
    else:
        weights = {"model.safetensors"}
    if not weights:
        raise ValueError("Checkpoint weight index is empty")
    for name in weights:
        weight = checkpoint / name
        if not weight.is_file() or weight.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint weight missing or empty: {weight}")
    subprocess.run(
        [game["godot_binary"], "--headless", "--version"], check=True, timeout=15
    )
    subprocess.run(
        [game["ffmpeg_binary"], "-version"],
        check=True,
        timeout=15,
        stdout=subprocess.DEVNULL,
    )


def validate_godot_run_config(config, game: dict) -> None:
    """Use NVIDIA's training checks without AlpaSim Wizard's GT warmup requirement.

    These validators are pinned together with the upstream checkout. The public
    host validator also validates deployment of AlpaSim, which we do not launch.
    """
    from alpagym_host.config_validation import (
        _validate_cosmos_grpo_batch_geometry,
        _validate_cosmos_mode,
        _validate_nccl_test_model,
        _validate_policy_model_path,
        _validate_training_policy_config,
        _validate_transport_config,
    )

    if str(config.execution.backend) != "local_process":
        raise ValueError("Godot launcher supports only local_process deployment")
    if config.execution.slurm.autoresume:
        raise ValueError("Godot launcher does not support Slurm autoresume")
    if config.alpasim.wizard_args.force_gt_duration_us != 0:
        raise ValueError("Godot episodes must not use recorded ground truth warmup")
    if config.alpasim.wizard_args.control_timestep_us != 200_000:
        raise ValueError("Godot bridge requires 0.2 second replanning")
    from training.episode_config import game_scene_ids

    scenarios = game_scene_ids(game)
    if game.get("scenario_ids", scenarios) != scenarios or game.get(
        "scenario_id", scenarios[0]
    ) != scenarios[0]:
        raise ValueError("Game scene identity differs from initial conditions")
    if (
        list(config.dataset.scene_ids or []) != scenarios
        or config.dataset.test_suite_id
    ):
        raise ValueError("Godot launcher requires the supported Thunderhill scene")
    _validate_training_policy_config(config)
    _validate_cosmos_grpo_batch_geometry(config.cosmos)
    _validate_transport_config(config)
    _validate_policy_model_path(config)
    _validate_cosmos_mode(config)
    _validate_nccl_test_model(config)


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
        "--fail-fast-workers",
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
    cleanup_state = {"completed": False}
    try:
        yield cleanup_state
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
            _, survivors = psutil.wait_procs(alive, timeout=5)
            if survivors:
                raise RuntimeError("Run descendants remain alive after cleanup")
        finally:
            if libc.prctl(36, previous.value, 0, 0, 0) != 0:
                raise OSError(
                    ctypes.get_errno(), "Cannot restore child subreaper state"
                )
        cleanup_state["completed"] = True


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
    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
    status = {
        "scatter_diagnostics": manifest.get("scatter_diagnostics", False),
        "run_id": run_dir.name,
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


def render_finished_run(run_dir: Path) -> None:
    """Render preserved attempts after training processes release the GPU."""
    game = json.loads((run_dir / "game_config.json").read_text())
    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
    budget = float(manifest["max_video_seconds"])
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Prepared video time budget must be finite and positive")
    deadline = time.monotonic() + budget
    status_path = run_dir / "video_status.json"

    def report(state, **details):
        pending = status_path.with_suffix(".pending")
        pending.write_text(
            json.dumps(dict(state=state, max_video_seconds=budget, **details), indent=2)
            + "\n"
        )
        pending.replace(status_path)

    report("rendering")
    try:
        with owned_subreaper(), (run_dir / "logs/video-render.log").open("w") as log:
            renderer = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools/render_video_queue.py"),
                    str(run_dir),
                    "--godot",
                    game["godot_binary"],
                    "--ffmpeg",
                    game["ffmpeg_binary"],
                    "--workers",
                    "1",
                    "--recover-interrupted",
                ],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_process(renderer, deadline)
            finally:
                stop_process_tree(renderer)
        indexes = sorted(run_dir.rglob("videos/index.json"))
        count = sum(len(json.loads(path.read_text())["videos"]) for path in indexes)
        report(
            "completed" if count else "no_recordings",
            videos=count,
            indexes=[str(path.relative_to(run_dir)) for path in indexes],
        )
    except BaseException as error:
        report("failed", error_type=type(error).__name__, error=str(error))
        raise


def run(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    if (run_dir / "run_status.json").exists():
        raise ValueError("Run already attempted; prepare a new run")
    cleanup = None
    try:
        with owned_subreaper() as cleanup:
            _run_owned(run_dir)
    finally:
        path = run_dir / "run_status.json"
        if path.is_file():
            status = json.loads(path.read_text())
            if status.get("state") == "stopping":
                if cleanup is None or not cleanup["completed"]:
                    write_status(run_dir, "cleanup_failed", runtime_stopped=False)
                    raise RuntimeError(
                        "Cleanup did not finish; recording recovery is blocked"
                    )
                outcome = status.pop("outcome")
                status.pop("state")
                status.pop("updated_at_unix", None)
                write_status(run_dir, outcome, **status, runtime_stopped=True)
                # Preserve the original training error if video rendering also fails.
                training_failed = sys.exc_info()[0] is not None
                try:
                    render_finished_run(run_dir)
                except BaseException:
                    if not training_failed:
                        raise
                    print(
                        f"Video rendering failed; inspect {run_dir / 'video_status.json'}",
                        file=sys.stderr,
                    )


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
        if manifest.get("native_source_patch") != verify_source(source):
            raise ValueError(
                "Native navigation input contract changed; prepare a fresh run"
            )
        from alpagym_host.config import load_run_config
        from alpagym_host.endpoint_registry import FileTopologyRegistry
        from alpagym_host.transport_env import apply_transport_env_vars

        config = load_run_config(run_dir / "resolved_config.yaml")
        if manifest.get("reward_version") != REWARD_VERSION:
            raise ValueError(
                "Reward contract changed; prepare a fresh normalized reward run"
            )
        validate_reward_terms(config.reward.terms)
        game = json.loads((run_dir / "game_config.json").read_text())
        validate_godot_run_config(config, game)
        validate_runtime_files(source, config, game)
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
                (
                    "from training.native_cosmos_source import verify_installed; "
                    "verify_installed(apply_patch=True); "
                    "from training.native_recipe_source import verify_installed as verify_recipe; "
                    "verify_recipe(apply_patch=True); "
                    "import torch; import alpagym_runtime.cosmos.entrypoint; "
                    "from alpagym_alpamayo_r1.bundle import install_alpamayo_r1_runtime_bridge; "
                    "install_alpamayo_r1_runtime_bridge(); "
                    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "assert torch.cuda.device_count() >= 2, 'The official topology requires two CUDA GPUs'"
                ),
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
        environment["ALPAGYM_SCATTER_DIAGNOSTICS"] = (
            "1" if manifest.get("scatter_diagnostics", False) else "0"
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["ALPAGYM_INFERENCE_CAPTURE_DIR"] = str(
            run_dir / "inference-capture"
        )
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
            run_dir,
            "stopping",
            outcome="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    else:
        write_status(
            run_dir,
            "stopping",
            outcome="completed",
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
    prep.add_argument(
        "--ffmpeg", default="ffmpeg", help="Required encoder, checked before training"
    )
    prep.add_argument(
        "--model-name", help="Recording label; defaults to checkpoint directory name"
    )
    prep.add_argument("--max-steps", type=int, default=1)
    prep.add_argument(
        "--rollouts",
        type=int,
        default=2,
        help="Sibling rollouts per scene, NVIDIA n_generation",
    )
    prep.add_argument("--episode-seconds", type=float, default=30)
    prep.add_argument(
        "--initial-speed-m-s",
        type=float,
        default=0.0,
        help="Start-line speed, or the cap for curvature-derived randomized start speeds",
    )
    prep.add_argument(
        "--randomized-starts",
        type=int,
        default=0,
        help="Number of seeded track-position start scenes (0 keeps the start line)",
    )
    prep.add_argument("--start-seed", type=int, default=0)
    prep.add_argument("--scatter-diagnostics", action="store_true")
    prep.add_argument("--concurrency", type=int, default=1)
    prep.add_argument("--max-wall-seconds", type=float, default=3600)
    prep.add_argument(
        "--max-video-seconds",
        type=float,
        default=3600,
        help="Separate post-training video rendering budget",
    )
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
                initial_speed_m_s=args.initial_speed_m_s,
                randomized_start_count=args.randomized_starts,
                randomized_start_seed=args.start_seed,
                scatter_diagnostics=args.scatter_diagnostics,
                concurrency=args.concurrency,
                max_wall_seconds=args.max_wall_seconds,
                model_name=args.model_name,
                ffmpeg=args.ffmpeg,
                max_video_seconds=args.max_video_seconds,
            )
        )
    else:
        run(args.run_dir)


if __name__ == "__main__":
    main()
