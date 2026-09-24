"""NVIDIA worker entrypoint with recording identity, no training modifications.

AlpaGym's backend discards Cosmos's current_weight_version argument. Our simulator
needs that identity for video overlays. The rollout wrapper publishes identity and observes completed gameplay metrics
while leaving the upstream rollout and training computation unchanged. Prefetch must remain disabled: prefetched
sessions would otherwise start outside this versioned call.
"""

from __future__ import annotations

import json
import math
import os
import socket
from statistics import mean
import tempfile
from pathlib import Path
import uuid


def publish_identity(
    directory: Path,
    port: int,
    *,
    policy_version: int,
    batch_id: str,
    is_validation: bool,
    active: bool,
) -> Path:
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError("Cosmos must supply a nonnegative policy weight version")
    if not 0 < port < 65536 or not batch_id:
        raise ValueError("Recording identity requires a driver port and batch ID")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"driver_{port}.json"
    pending = path.with_suffix(".pending")
    pending.write_text(
        json.dumps(
            dict(
                policy_version=policy_version,
                batch_id=batch_id,
                is_validation=is_validation,
                active=active,
            )
        )
        + "\n"
    )
    pending.replace(path)
    return path


def _finite_metrics(values):
    result = {str(key): float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite simulator diagnostic metric")
    return result


def summarize_rollouts(results, identity, reward_config):
    """Observe native completed episodes without changing their reward or replay.

    A weight version is not an optimizer step count. Partial batches sharing a
    version remain separate observations; conditional lap times only summarize
    episodes that actually reported a completed lap.
    """
    episodes = []
    for result in results:
        for episode in result.completions:
            if episode.metrics is None or episode.reward is None:
                raise ValueError("NVIDIA episode lacks computed metrics or reward")
            metrics = _finite_metrics(episode.metrics.aggregated)
            components = {}
            for index, term in enumerate(reward_config.terms):
                if term.kind != "metric":
                    raise ValueError(
                        "Racing diagnostics require gameplay metric rewards"
                    )
                components[f"{index}_{term.metric_name}"] = (
                    float(term.scale) * metrics[term.metric_name]
                )
            reward = float(episode.reward.total)
            if not math.isfinite(reward) or not math.isclose(
                sum(components.values()), reward, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("Reported components differ from NVIDIA reward")
            if "lap_time_seconds" in metrics and metrics.get("lap_completed") != 1:
                raise ValueError("Only completed laps can have lap time metrics")
            episodes.append(
                dict(
                    session_uuid=episode.session_uuid,
                    scene_id=episode.scene_id,
                    num_steps=episode.num_steps,
                    is_valid=episode.is_valid,
                    reward=reward,
                    metrics=metrics,
                    reward_components=_finite_metrics(components),
                    reward_report=_finite_metrics(episode.reward.report_metrics),
                )
            )
    if not episodes:
        raise ValueError("Cannot report an empty completed rollout batch")
    prefix = "evaluation" if identity["is_validation"] else "rollout"
    charts = {
        "policy_version": identity["policy_version"],
        f"{prefix}/count": len(episodes),
    }
    measurements = {"reward": [episode["reward"] for episode in episodes]}
    for key in sorted({key for episode in episodes for key in episode["metrics"]}):
        measurements[key] = [
            episode["metrics"][key] for episode in episodes if key in episode["metrics"]
        ]
    for key in sorted(
        {key for episode in episodes for key in episode["reward_components"]}
    ):
        measurements["reward_component/" + key] = [
            episode["reward_components"][key] for episode in episodes
        ]
    for key, values in measurements.items():
        charts.update(
            {
                f"{prefix}/{key}_mean": mean(values),
                f"{prefix}/{key}_min": min(values),
                f"{prefix}/{key}_max": max(values),
                f"{prefix}/{key}_count": len(values),
            }
        )
    return dict(
        schema_version=1,
        **identity,
        episodes=episodes,
        charts=charts,
        metric_source="NVIDIA EpisodeOutput",
        training_reward_modified=False,
        optimizer_updates_inferred=False,
    )


def publish_metrics(directory, report):
    """Atomically publish one immutable batch even if W&B is unavailable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    batch_id = report["batch_id"]
    if not batch_id or any(c not in "0123456789abcdef" for c in batch_id):
        raise ValueError("Expected the generated hexadecimal batch ID")
    path = directory / f"policy-{report['policy_version']:06d}-{batch_id}.json"
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, prefix=".pending-", delete=False
    ) as file:
        pending = Path(file.name)
        try:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
            os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    return path


class RolloutDiagnostics:
    """Distinct optional W&B run, never sharing the native trainer's writer.

    WANDB_MODE and WANDB_ENTITY retain their standard SDK meanings. No frames,
    model inputs, credentials, full config or arbitrary source code are uploaded.
    Every batch is durably published before the optional W&B operation.
    """

    def __init__(self, run_config, driver_port):
        self.config = run_config
        self.directory = Path(run_config.artifact_paths.run_dir) / "rollout_metrics"
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}-{driver_port}"
        self.run = None

    def record(self, results, identity):
        report = summarize_rollouts(results, identity, self.config.reward)
        report["worker_id"] = self.worker_id
        path = publish_metrics(self.directory, report)
        if "wandb" not in self.config.cosmos.logging.logger:
            return path
        try:
            if identity.get("fixture") and os.environ.get("WANDB_MODE") not in {
                "offline",
                "dryrun",
                "disabled",
            }:
                raise ValueError(
                    "Synthetic fixture diagnostics require offline or disabled W&B"
                )
            if self.run is None:
                import wandb

                logging_config = self.config.cosmos.logging
                self.run = wandb.init(
                    project=logging_config.project_name,
                    group=Path(self.config.artifact_paths.run_dir).name,
                    name=f"{logging_config.experiment_name}-game-metrics-{self.worker_id}",
                    job_type="gameplay-diagnostics",
                    dir=str(self.directory),
                    id=uuid.uuid4().hex,
                    reinit="create_new",
                    config=dict(
                        metric_source="NVIDIA EpisodeOutput",
                        training_reward_modified=False,
                        native_trainer_run_is_separate=True,
                        fixture=bool(identity.get("fixture", False)),
                    ),
                    settings=wandb.Settings(
                        disable_git=True, save_code=False, console="off"
                    ),
                )
                self.run.define_metric("policy_version")
                for namespace in ("rollout", "evaluation"):
                    self.run.define_metric(
                        namespace + "/*", step_metric="policy_version"
                    )
                tracking = dict(
                    run_id=self.run.id,
                    run_directory=self.run.dir,
                    project=self.run.project,
                    entity=self.run.entity,
                    mode=self.run.settings.mode,
                    native_trainer_run_is_separate=True,
                )
                (self.directory / f"tracking-{self.worker_id}.json").write_text(
                    json.dumps(tracking, indent=2) + "\n"
                )
            self.run.log(report["charts"])
        except Exception as error:
            raise RuntimeError(
                f"Gameplay W&B logging failed; durable metrics preserved at {path}"
            ) from error
        return path

    def close(self):
        if self.run is not None:
            self.run.finish()
            self.run = None


def register_recorded_rollout():
    from alpagym_runtime.cosmos.rollout_backend import AlpagymRollout
    from cosmos_rl.rollout.rollout_base import RolloutRegistry

    @RolloutRegistry.register("thunderhill_alpagym_rollout")
    class RecordedAlpagymRollout(AlpagymRollout):
        def rollout_generation(
            self,
            payloads,
            stream,
            data_packer,
            data_fetcher=None,
            is_validation=False,
            current_weight_version=None,
        ):
            if self._run_config.cosmos.rollout.prefetch_rollout:
                raise ValueError("Video identity requires disabled rollout prefetch")
            if self._driver_server is None:
                raise RuntimeError("NVIDIA driver must be initialized before rollout")
            directory = self._run_config.artifact_paths.run_dir / "rollout_identity"
            identity = dict(
                policy_version=current_weight_version,
                batch_id=uuid.uuid4().hex,
                is_validation=is_validation,
            )
            publish_identity(
                directory, self._driver_server.port, active=True, **identity
            )
            try:
                results = super().rollout_generation(
                    payloads,
                    stream,
                    data_packer,
                    data_fetcher=data_fetcher,
                    is_validation=is_validation,
                    current_weight_version=current_weight_version,
                )
                if getattr(self, "_gameplay_diagnostics", None) is None:
                    self._gameplay_diagnostics = RolloutDiagnostics(
                        self._run_config, self._driver_server.port
                    )
                self._gameplay_diagnostics.record(results, identity)
                return results
            finally:
                publish_identity(
                    directory, self._driver_server.port, active=False, **identity
                )

        def shutdown(self):
            try:
                super().shutdown()
            finally:
                diagnostics = getattr(self, "_gameplay_diagnostics", None)
                if diagnostics is not None:
                    diagnostics.close()

    return RecordedAlpagymRollout


def main():
    # Import upstream registrations first, then add the recording adapter.
    from alpagym_runtime.cosmos.entrypoint import main as nvidia_main

    register_recorded_rollout()
    nvidia_main()


if __name__ == "__main__":
    main()
