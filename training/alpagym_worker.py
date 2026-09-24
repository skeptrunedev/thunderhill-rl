"""NVIDIA worker entrypoint with recording identity, no training modifications.

AlpaGym's backend discards Cosmos's current_weight_version argument. Our simulator
needs that identity for video overlays. The only override publishes it while the
unchanged upstream rollout runs. Prefetch must remain disabled: prefetched
sessions would otherwise start outside this versioned call.
"""

from __future__ import annotations

import json
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
                return super().rollout_generation(
                    payloads,
                    stream,
                    data_packer,
                    data_fetcher=data_fetcher,
                    is_validation=is_validation,
                    current_weight_version=current_weight_version,
                )
            finally:
                publish_identity(
                    directory, self._driver_server.port, active=False, **identity
                )

    return RecordedAlpagymRollout


def main():
    # Import upstream registrations first, then add the recording adapter.
    from alpagym_runtime.cosmos.entrypoint import main as nvidia_main

    register_recorded_rollout()
    nvidia_main()


if __name__ == "__main__":
    main()
