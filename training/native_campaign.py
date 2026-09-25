"""One bounded NVIDIA RL campaign, with native checkpoints and actual gameplay.

A configured wall deadline is a budget stop, not successful trainer completion.
The original launcher status, interrupted attempts, and native resume state stay
intact. This module never implements optimization or restarts a trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

TOTAL_SECONDS = 12 * 60 * 60
FINAL_RESERVE_SECONDS = int(3.5 * 60 * 60)
EVALUATION_SECONDS = 2 * 60 * 60


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def validated_prerequisites(
    validation_run: Path, model: Path, gpu: str | None = None
) -> dict:
    """Require observed optimization, fresh reload, and hardware rendered gameplay."""
    from training.native_source import validate_navigation_checkpoint

    report_path = validation_run / "validation.json"
    report = json.loads(report_path.read_text())
    required = (
        "optimizer_updates_verified",
        "checkpoint_reload_verified",
        "weights_changed",
    )
    if report.get("state") != "completed" or not all(
        report.get(key) is True for key in required
    ):
        raise ValueError(
            "The native validation has not completed optimization and fresh reload"
        )
    launch = json.loads((validation_run / "launch_manifest.json").read_text())
    camera = json.loads((validation_run / "camera-preflight/summary.json").read_text())
    adapter = camera.get("renderer", {}).get("adapter", "")
    if camera.get("ok") is not True or camera.get("views_per_observation") != 4:
        raise ValueError("Validation lacks successful four camera hardware capture")
    if gpu is None:
        matches = [
            name for name in ("H100", "L40S") if name.casefold() in adapter.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(f"No reviewed campaign GPU matches renderer {adapter!r}")
        gpu = matches[0]
    if gpu.casefold() not in adapter.casefold() or any(
        word in adapter.casefold()
        for word in ("llvmpipe", "lavapipe", "software", "swiftshader")
    ):
        raise ValueError(
            f"Validation renderer {adapter!r} does not establish {gpu} support"
        )
    if launch.get("topology") != "local_disaggregated_2gpu":
        raise ValueError("Campaign requires the validated native two GPU topology")
    game = json.loads((validation_run / "game_config.json").read_text())
    if Path(game["model_path"]).resolve() != model.resolve():
        raise ValueError("Campaign initial model differs from the validated model")
    return dict(
        validation_run=str(validation_run),
        validation_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        navigation_provenance=validate_navigation_checkpoint(model),
        native_source_patch=launch["native_source_patch"],
        reward_version=launch["reward_version"],
        renderer=adapter,
        topology=launch["topology"],
        gpu=gpu,
        initial_speed_m_s=game.get("initial_speed_m_s", 0.0),
        scenario_id=game.get("scenario_id", "thunderhill-east-standing"),
    )


def prepare_campaign(
    source: Path,
    output: Path,
    model: Path,
    *,
    episode_seconds: float,
    rollouts: int,
    concurrency: int,
    checkpoint_every: int,
    training_seconds: float,
    video_seconds: float,
    initial_speed_m_s: float = 0.0,
    max_steps: int = 100_000,
) -> Path:
    """CPU only preparation using NVIDIA's own config serialization."""
    from training.nvidia_alpagym import prepare, load_upstream

    load_upstream(source)
    from alpagym_host.config import load_run_config
    from alpagym_host.run_artifacts import write_run_artifacts

    if checkpoint_every < 1 or checkpoint_every > max_steps:
        raise ValueError("Checkpoint interval must be within the configured step count")
    run_dir = prepare(
        source,
        output,
        model,
        godot="godot",
        max_steps=max_steps,
        rollouts=rollouts,
        episode_seconds=episode_seconds,
        initial_speed_m_s=initial_speed_m_s,
        concurrency=concurrency,
        max_wall_seconds=training_seconds,
        max_video_seconds=video_seconds,
        model_name="Alpamayo 1.5 native RL",
    )
    config = load_run_config(run_dir / "resolved_config.yaml")
    config.cosmos.train.ckpt.enable_checkpoint = True
    config.cosmos.train.ckpt.export_safetensors = True
    config.cosmos.train.ckpt.save_freq = checkpoint_every
    # NVIDIA prunes the corresponding safetensors export with each old resume
    # bundle. Its existing retention policy keeps five, including its best link.
    config.cosmos.train.ckpt.max_keep = 5
    config.cosmos.logging.experiment_name = "thunderhill-campaign-" + run_dir.name
    write_run_artifacts(config)
    return run_dir


def completed_exports(run_dir: Path) -> list[dict]:
    """Use native completion markers, never the existence of a step directory.

    The selected topology has one policy saving rank and one rollout GPU. Native
    checkpoint.py writes the rank marker after model, optimizer, scheduler, and
    RNG/data position files. Export occurs before that checkpoint save.
    """
    result = []
    for policy in run_dir.glob("cosmos/**/checkpoints/step_*/policy"):
        try:
            files = [
                "cosmos_config",
                "model_rank_0.pth",
                "optimizer_rank_0.pth",
                "scheduler_rank_0.pth",
                "extra_info_rank_0.pth",
            ]
            if not (policy / ".rank_0_complete").is_file():
                continue
            if any(
                not (policy / name).is_file() or (policy / name).stat().st_size == 0
                for name in files
            ):
                continue
            exported = policy.parents[2] / "safetensors" / policy.parent.name
            if not (exported / "config.json").is_file() or not list(
                exported.glob("*.safetensors")
            ):
                continue
            result.append(
                dict(
                    step=int(policy.parent.name.removeprefix("step_")),
                    checkpoint=str(policy),
                    export=str(exported),
                )
            )
        except FileNotFoundError:
            # Native checkpoint pruning can remove a candidate during diagnostics.
            # Missing candidates are incomplete/unavailable, never selected.
            continue
    return sorted(result, key=lambda row: row["step"])


def summarize(run_dir: Path) -> dict:
    """Report observed rewards separately from optimizer/checkpoint evidence."""
    generations = {}
    for path in sorted(run_dir.rglob("rollout_metrics/policy-*.json")):
        report = json.loads(path.read_text())
        if report["is_validation"]:
            continue
        version = report["policy_version"]
        generations.setdefault(version, []).extend(report["episodes"])
    rows = []
    for version, episodes in sorted(generations.items()):
        rows.append(
            dict(
                policy_version=version,
                rollouts=len(episodes),
                reward_mean=statistics.mean(row["reward"] for row in episodes),
                reward_min=min(row["reward"] for row in episodes),
                reward_max=max(row["reward"] for row in episodes),
                legal_progress_m_mean=statistics.mean(
                    row["metrics"]["legal_progress_m"] for row in episodes
                ),
                speed_m_s_mean=statistics.mean(
                    row["metrics"]["mean_progress_speed_m_s"] for row in episodes
                ),
                laps_completed=sum(
                    row["metrics"].get("lap_completed", 0) for row in episodes
                ),
                stalls=sum(row["metrics"].get("stalled", 0) for row in episodes),
            )
        )
    steps, gradient_records = set(), set()
    for path in (run_dir / "logs").rglob("*.log"):
        log = path.read_text(errors="replace")
        steps.update(
            int(value)
            for value in re.findall(
                r"AlpaGym trainer step start current_step=(\d+)", log
            )
        )
        for line in log.splitlines():
            if "AlpaGym trainer minibatch" in line:
                match = re.search(r"grad_norm=([0-9.eE+\-]+)", line)
                if match and math.isfinite(float(match[1])) and float(match[1]) > 0:
                    gradient_records.add(line)
    queued, rendered, recorded = set(), set(), set()
    missing_recordings = []
    for provenance in run_dir.rglob("recordings/**/provenance.json"):
        attempt = provenance.parent
        found = False
        for recording in (attempt / "environment").rglob("*.jsonl"):
            with recording.open() as stream:
                header_line = stream.readline()
            if not header_line.endswith("\n"):
                continue
            header = json.loads(header_line)
            if header.get("type") == "episode" and header.get("policy_display"):
                recorded.add((str(attempt.relative_to(run_dir)), header["episode_id"]))
                found = True
        if not found:
            missing_recordings.append(str(attempt.relative_to(run_dir)))
    for path in run_dir.rglob("video_jobs/*.json"):
        queued.add(
            (
                str(path.parent.parent.relative_to(run_dir)),
                json.loads(path.read_text())["episode_id"],
            )
        )
    for path in run_dir.rglob("videos/index.json"):
        index = json.loads(path.read_text())
        rendered.update(
            (str(path.parent.parent.relative_to(run_dir)), row["episode_id"])
            for row in index["videos"]
        )

    return dict(
        updated_at_unix=time.time(),
        generations=rows,
        completed_rollouts=sum(row["rollouts"] for row in rows),
        trainer_steps_started=sorted(steps),
        nonzero_gradient_log_records=len(gradient_records),
        completed_native_checkpoints=completed_exports(run_dir),
        video_jobs=len(queued),
        rendered_videos=len(rendered),
        video_backlog=len(queued - rendered),
        raw_recorded_episodes=len(recorded),
        raw_episodes_without_jobs=sorted(recorded - queued),
        raw_episodes_without_videos=sorted(recorded - rendered),
        attempts_without_recordings=missing_recordings,
        complete_video_coverage=bool(recorded)
        and not missing_recordings
        and recorded <= rendered,
        optimizer_updates_inferred_from_rollouts=False,
    )


def compare_evaluations(run_dir: Path, expected_episodes: int) -> dict:
    """Pair actual game outcomes by fixed seed and use NVIDIA's reward dispatcher."""
    from types import SimpleNamespace
    from alpagym_host.config import load_run_config
    from alpagym_runtime.rewards.compute import compute_reward
    from alpagym_runtime.types import EpisodeMetrics

    reward_config = load_run_config(run_dir / "resolved_config.yaml").reward
    reports = [
        json.loads((run_dir / "recordings" / stage / "evaluation.json").read_text())
        for stage in ("baseline", "final")
    ]
    episodes = [
        {row["session_uuid"]: row for row in report["episodes"]} for report in reports
    ]
    if (
        episodes[0].keys() != episodes[1].keys()
        or len(episodes[0]) != expected_episodes
    ):
        raise ValueError(
            "Baseline/final evaluation seeds do not match the requested episodes"
        )
    pairs = []
    metric_names = (
        "legal_progress_m",
        "mean_progress_speed_m_s",
        "crashed",
        "offroad",
        "lap_completed",
    )
    for session in sorted(episodes[0]):
        outcomes = []
        for stage in episodes:
            row = stage[session]
            if not row["success"]:
                raise ValueError(
                    "Evaluation contains a failed simulator protocol attempt"
                )
            # The configured metric reward reads only these actual simulator
            # metrics. This creates no training episode, replay, or action labels.
            reward = compute_reward(
                SimpleNamespace(metrics=EpisodeMetrics(aggregated=row["metrics"])),
                None,
                reward_config,
            ).total
            outcomes.append(
                dict(
                    reward=reward,
                    **{name: row["metrics"][name] for name in metric_names},
                )
            )
        pairs.append(
            dict(
                session_uuid=session,
                baseline=outcomes[0],
                final=outcomes[1],
                delta={
                    name: outcomes[1][name] - outcomes[0][name] for name in outcomes[0]
                },
            )
        )
    return dict(
        paired_episodes=pairs,
        mean_delta={
            name: statistics.mean(row["delta"][name] for row in pairs)
            for name in pairs[0]["delta"]
        },
        improvement_claimed=False,
        limitation="Matched episodes are a diagnostic comparison, not proof of general improvement.",
    )


def run_campaign(
    source: Path,
    model: Path,
    output: Path,
    validation_run: Path,
    *,
    deadline_unix: float,
    episode_seconds: float,
    rollouts: int,
    concurrency: int,
    checkpoint_every: int,
    initial_speed_m_s: float = 0.0,
    max_steps: int = 100_000,
    evaluation_episodes: int = 8,
) -> dict:
    from training.native_source import record_navigation_checkpoint, verify_source
    from training.nvidia_alpagym import owned_subreaper, stop_process_tree
    from training.alpagym_metrics import REWARD_VERSION

    started = time.time()
    if evaluation_episodes not in (2, 4, 8):
        raise ValueError("Choose 2, 4, or 8 matched evaluation episodes")
    if (
        not math.isfinite(deadline_unix)
        or not 3600 < deadline_unix - started <= TOTAL_SECONDS
    ):
        raise ValueError("Campaign deadline must leave between one and twelve hours")
    certificate = validated_prerequisites(validation_run, model)
    if certificate["initial_speed_m_s"] != initial_speed_m_s:
        raise ValueError("Campaign initial speed differs from validated initial state")
    patch = verify_source(source, apply_patch=True)
    if (
        certificate["native_source_patch"] != patch
        or certificate["reward_version"] != REWARD_VERSION
    ):
        raise ValueError(
            "Validated navigation/reward contract differs from the campaign"
        )
    remaining = deadline_unix - time.time()
    reserve = min(FINAL_RESERVE_SECONDS, remaining / 2)
    video_budget = min(1800.0, reserve / 3)
    run_dir = prepare_campaign(
        source,
        output,
        model,
        episode_seconds=episode_seconds,
        initial_speed_m_s=initial_speed_m_s,
        rollouts=rollouts,
        concurrency=concurrency,
        checkpoint_every=checkpoint_every,
        training_seconds=remaining - reserve,
        video_seconds=video_budget,
        max_steps=max_steps,
    )
    report = dict(
        state="starting",
        run_dir=str(run_dir),
        started_at_unix=started,
        deadline_unix=deadline_unix,
        validated_prerequisites=certificate,
        evaluation_episodes=evaluation_episodes,
        single_native_run=True,
        optimizer_restarts=0,
        max_steps=max_steps,
        checkpoint_every_steps=checkpoint_every,
        native_checkpoint_retention=5,
        reserved_final_seconds=reserve,
        evaluation_phase_limit_seconds=EVALUATION_SECONDS,
        phases=[],
        training_outcome=None,
    )
    write_json(output / "campaign.json", report)

    def update():
        write_json(run_dir / "campaign_progress.json", summarize(run_dir))
        write_json(output / "campaign.json", report)

    def launch(arguments, name, cap_seconds, *, allow_failure=False):
        available = min(cap_seconds, deadline_unix - time.time() - 30)
        if available <= 0:
            raise TimeoutError(f"Campaign has no budget for {name}")
        phase = dict(name=name, started_at_unix=time.time(), max_seconds=available)
        report["phases"].append(phase)
        update()
        process = None
        try:
            with (
                owned_subreaper(),
                (run_dir / "logs" / (name + ".log")).open("w") as log,
            ):
                process = subprocess.Popen(
                    [sys.executable, *arguments],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                limit = time.monotonic() + available
                try:
                    while process.poll() is None:
                        remaining = limit - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                f"Campaign phase budget exhausted: {name}"
                            )
                        try:
                            process.wait(timeout=min(30, remaining))
                        except subprocess.TimeoutExpired:
                            update()
                    phase["exit_code"] = process.returncode
                    if process.returncode and not allow_failure:
                        raise subprocess.CalledProcessError(
                            process.returncode, process.args
                        )
                    return process.returncode
                finally:
                    stop_process_tree(process)
        finally:
            phase["finished_at_unix"] = time.time()
            update()

    error = None
    try:
        launch(
            [
                "-m",
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
            "camera-preflight",
            900,
        )
        launch(
            [
                "-m",
                "training.native_validation",
                "evaluate",
                str(run_dir),
                str(model),
                str(run_dir / "recordings/baseline"),
                "0",
                "--episodes",
                str(evaluation_episodes),
                "--rpc-timeout-seconds",
                str(EVALUATION_SECONDS - 60),
            ],
            "baseline-evaluation",
            EVALUATION_SECONDS,
        )
        # Charge startup and baseline against the total allowance, not to an
        # independent budget that could extend the billed GPU function.
        training_budget = deadline_unix - time.time() - reserve
        if training_budget <= 0:
            raise TimeoutError("Startup consumed the campaign training allowance")
        manifest_path = run_dir / "launch_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["max_wall_seconds"] = training_budget
        write_json(manifest_path, manifest)
        report["state"] = "training"
        code = launch(
            ["-m", "training.nvidia_alpagym", "run", str(run_dir)],
            "native-training",
            training_budget + video_budget + 120,
            allow_failure=True,
        )
        status = json.loads((run_dir / "run_status.json").read_text())
        report["native_launcher_status"] = status
        planned_stop = (
            status.get("runtime_stopped") is True
            and status.get("state") == "failed"
            and status.get("error_type") == "TimeoutError"
            and status.get("error") == "Prepared wall time budget exhausted"
        )
        native_completed = (
            status.get("state") == "completed" and status.get("runtime_stopped") is True
        )
        report["native_process_exit_code"] = code
        if code and not planned_stop and not native_completed:
            raise RuntimeError(
                f"Native trainer failed outside its planned budget: {status}"
            )
        if code and native_completed:
            # The native launcher renders after marking training complete. Keep
            # a rendering failure separate from the actual optimizer outcome.
            report["native_postprocessing_failed"] = True
        report["training_outcome"] = (
            "planned_budget_stop" if planned_stop else "native_completed"
        )
        checkpoints = completed_exports(run_dir)
        if not checkpoints:
            raise RuntimeError(
                "Campaign produced no complete native optimizer checkpoint"
            )
        chosen = checkpoints[-1]
        report["selected_checkpoint"] = chosen
        report["state"] = "finalizing"
        scripts = source / "packages/policies/alpamayo_r1/scripts"
        inference, reloaded = (
            run_dir / "converted-inference",
            run_dir / "reloaded-checkpoint",
        )
        launch(
            [
                str(scripts / "convert_alpagym_checkpoint_to_inference.py"),
                "--input",
                chosen["export"],
                "--output",
                str(inference),
            ],
            "convert-inference",
            900,
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
            "convert-reload",
            900,
        )
        release = (
            model.parent
            / "hub/models--nvidia--Alpamayo-1.5-10B/snapshots/7aba8293c09993f2e125c6819df05d7fa3e873ea/config.json"
        )
        record_navigation_checkpoint(reloaded, release)
        launch(
            [
                "-m",
                "training.native_campaign",
                "compare",
                str(model),
                str(reloaded),
                str(run_dir / "weight_comparison.json"),
            ],
            "compare-weights",
            900,
        )
        comparison = json.loads((run_dir / "weight_comparison.json").read_text())
        if not comparison["weights_changed"] or not comparison["frozen_vlm_unchanged"]:
            raise RuntimeError(
                "Campaign did not verify an expert weight update with frozen VLM"
            )
        report["weights_changed"] = True
        launch(
            [
                "-m",
                "training.native_validation",
                "evaluate",
                str(run_dir),
                str(reloaded),
                str(run_dir / "recordings/final"),
                str(chosen["step"]),
                "--episodes",
                str(evaluation_episodes),
                "--rpc-timeout-seconds",
                str(EVALUATION_SECONDS - 60),
            ],
            "final-evaluation",
            EVALUATION_SECONDS,
        )
        report["checkpoint_reload_verified"] = True
        report["evaluation_comparison"] = compare_evaluations(
            run_dir, evaluation_episodes
        )
        report["state"] = report["training_outcome"]
    except BaseException as caught:
        error = caught
        report.update(
            state="failed", error_type=type(caught).__name__, error=str(caught)
        )
    finally:
        # Native run handles interrupted training attempts. Evaluation attempts
        # share this same recording root and require a final queue pass as well.
        status_path = run_dir / "run_status.json"
        if not status_path.exists():
            from training.nvidia_alpagym import write_status

            write_status(
                run_dir,
                "failed",
                runtime_stopped=True,
                error="Campaign stopped before native trainer startup",
            )
        if (
            not isinstance(error, (KeyboardInterrupt, SystemExit))
            and deadline_unix - time.time() > 60
        ):
            try:
                launch(
                    [
                        "-m",
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
                    "campaign-videos",
                    deadline_unix - time.time() - 45,
                )
            except (
                OSError,
                RuntimeError,
                TimeoutError,
                subprocess.SubprocessError,
            ) as render_error:
                report["video_error"] = str(render_error)
        report["finished_at_unix"] = time.time()
        report["elapsed_seconds"] = time.time() - started
        update()
        report["summary"] = json.loads((run_dir / "campaign_progress.json").read_text())
        report["all_video_jobs_rendered"] = report["summary"]["complete_video_coverage"]
        write_json(output / "campaign.json", report)
    if error is not None:
        raise error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    for name in ("base", "trained", "output"):
        compare.add_argument(name, type=Path)
    for command in ("prepare", "run"):
        item = commands.add_parser(command)
        for name in ("source", "model", "output"):
            item.add_argument("--" + name, type=Path, required=True)
        item.add_argument("--episode-seconds", type=float, required=True)
        item.add_argument("--initial-speed-m-s", type=float, default=0.0)
        item.add_argument("--rollouts", type=int, required=True)
        item.add_argument("--concurrency", type=int, required=True)
        item.add_argument("--checkpoint-every", type=int, default=2)
        item.add_argument("--max-steps", type=int, default=100_000)
        if command == "run":
            item.add_argument("--validation-run", type=Path, required=True)
            item.add_argument("--deadline-unix", type=float, required=True)
            item.add_argument(
                "--evaluation-episodes", type=int, choices=(2, 4, 8), default=8
            )
        else:
            item.add_argument("--training-seconds", type=float, default=36000)
            item.add_argument("--video-seconds", type=float, default=1800)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "compare":
        from training.native_validation import compare_exports

        result = compare_exports(args["base"], args["trained"])
        result["frozen_vlm_unchanged"] = not any(
            row["name"].startswith("vlm.") for row in result["changed_tensors"]
        )
        write_json(args["output"], result)
    elif command == "prepare":
        print(prepare_campaign(**args))
    else:
        import signal

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Campaign interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, terminate)
        print(json.dumps(run_campaign(**args)))


if __name__ == "__main__":
    main()
