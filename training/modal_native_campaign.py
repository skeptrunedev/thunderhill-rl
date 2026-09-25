"""Launch one validated, idempotent, twelve hour native RL attempt on Modal.

Preparation checks run on CPU before a GPU function is requested. GPU selection
comes from a completed real validation receipt. No automatic training retries or
optimizer resets are performed.
"""

from pathlib import Path

import modal

from training.modal_native import REMOTE, ROOT, UPSTREAM, cache, image, runs

app = modal.App("thunderhill-native-campaign")
reservations = modal.Dict.from_name(
    "thunderhill-native-campaign-reservations", create_if_missing=True
)
runtime_image = (
    image.env({"NVIDIA_DRIVER_CAPABILITIES": "all"})
    .add_local_dir(
        str(ROOT / "training"),
        REMOTE + "/training",
        ignore=["alpamayo", ".venv", "__pycache__", "results", "*.md"],
    )
    .add_local_dir(
        str(ROOT / "tools"), REMOTE + "/tools", ignore=["__pycache__", "*.md"]
    )
)
tracking_secret = modal.Secret.from_name("thunderhill-wandb")


@app.function(
    image=runtime_image,
    cpu=2,
    memory=4096,
    timeout=180,
    retries=0,
    volumes={"/model-cache": cache, "/runs": runs},
    secrets=[tracking_secret],
    include_source=False,
)
def prerequisites(validation_run: str):
    import json
    import os
    import subprocess

    if not os.environ.get("WANDB_API_KEY") or not os.environ.get("WANDB_ENTITY"):
        raise RuntimeError("Online W&B secret is incomplete")
    runs.reload()
    cache.reload()
    command = (
        "import json,sys; from pathlib import Path; "
        "from training.native_campaign import validated_prerequisites; "
        "print(json.dumps(validated_prerequisites(Path(sys.argv[1]), Path(sys.argv[2]))))"
    )
    result = subprocess.run(
        [
            UPSTREAM + "/.venv/bin/python",
            "-c",
            command,
            validation_run,
            "/model-cache/alpagym-converted-1.5",
        ],
        cwd=REMOTE,
        check=True,
        capture_output=True,
        text=True,
        timeout=150,
    )
    return json.loads(result.stdout)


@app.function(
    image=runtime_image,
    cpu=16,
    memory=262144,
    timeout=43200,
    retries=0,
    max_containers=1,
    scaledown_window=1,
    volumes={"/model-cache": cache, "/runs": runs},
    secrets=[tracking_secret],
    include_source=False,
)
def campaign(campaign_id: str, source_revision: str, certificate: dict, settings: dict):
    import json
    import os
    import signal
    import subprocess
    import threading
    import time

    started = time.time()
    hard_deadline = started + 43200
    destination = Path("/runs/native-campaigns") / campaign_id
    destination.mkdir(parents=True, exist_ok=False)
    os.chdir(REMOTE)
    os.environ.update(
        WANDB_MODE="online",
        WANDB_PROJECT="thunderhill-rl",
        WANDB_RUN_GROUP=campaign_id,
        WANDB_DIR=str(destination),
    )
    identity = dict(
        repository="skeptrunedev/thunderhill-rl",
        commit=source_revision,
        clean_worktree_at_launch=True,
        campaign_id=campaign_id,
        gpu=certificate["gpu"] + ":2",
        started_at_unix=started,
        hard_deadline_unix=hard_deadline,
        total_gpu_function_limit_seconds=43200,
        settings=settings,
        validated_prerequisites=certificate,
    )
    (destination / "source_identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    runs.commit()
    print(f"Campaign artifacts: {destination}", flush=True)
    commit_stop, commit_errors = threading.Event(), []

    def persist():
        while not commit_stop.wait(300):
            try:
                runs.commit()
            except Exception as error:
                commit_errors.append(str(error))
                return

    persistence = threading.Thread(
        target=persist, name="persist-campaign-artifacts", daemon=True
    )
    persistence.start()
    process = None
    outcome = dict(state="starting", campaign_id=campaign_id)
    try:
        if not os.environ.get("WANDB_API_KEY") or not os.environ.get("WANDB_ENTITY"):
            raise RuntimeError("Online W&B credentials are required for this campaign")
        from training.native_network import configure_container_hostname

        (destination / "container-network.json").write_text(
            json.dumps(configure_container_hostname(), indent=2) + "\n"
        )
        # Placement can change between validation and this allocation. Reject any
        # pair lacking actual CUDA peer access; there is no software fallback.
        with (destination / "gpu-topology.json").open("w") as log:
            subprocess.run(
                [
                    UPSTREAM + "/.venv/bin/python",
                    "-c",
                    "import json,torch; n=torch.cuda.device_count(); "
                    "assert n == 2, 'Native campaign needs exactly two GPUs'; "
                    "names=[torch.cuda.get_device_name(i) for i in range(n)]; "
                    "peers=[torch.cuda.can_device_access_peer(0,1),torch.cuda.can_device_access_peer(1,0)]; "
                    "print(json.dumps(dict(devices=names,bidirectional_peer_access=peers))); "
                    "assert all(peers), 'CUDA peer access unavailable'",
                ],
                check=True,
                stdout=log,
                timeout=120,
            )
        observed = json.loads((destination / "gpu-topology.json").read_text())
        if not all(
            certificate["gpu"].casefold() in name.casefold()
            for name in observed["devices"]
        ):
            raise RuntimeError("Allocated GPUs differ from the validated GPU model")
        # Verify the CPU gate's immutable report again inside the allocation.
        check = subprocess.run(
            [
                UPSTREAM + "/.venv/bin/python",
                "-c",
                "import json,sys; from pathlib import Path; from training.native_campaign import validated_prerequisites; "
                "print(json.dumps(validated_prerequisites(Path(sys.argv[1]),Path(sys.argv[2]))))",
                certificate["validation_run"],
                "/model-cache/alpagym-converted-1.5",
            ],
            cwd=REMOTE,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if json.loads(check.stdout) != certificate:
            raise RuntimeError("Validated model or receipt changed before allocation")
        command = [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 800x600x24",
            UPSTREAM + "/.venv/bin/python",
            "-m",
            "training.native_campaign",
            "run",
            "--source",
            UPSTREAM,
            "--model",
            "/model-cache/alpagym-converted-1.5",
            "--output",
            str(destination),
            "--validation-run",
            certificate["validation_run"],
            # Five minutes remain for stopping/reaping and final volume persistence.
            "--deadline-unix",
            str(hard_deadline - 300),
        ]
        for key, value in settings.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        with (destination / "supervisor.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                if commit_errors:
                    raise RuntimeError(
                        "Periodic artifact persistence failed: " + commit_errors[0]
                    )
                if time.time() >= hard_deadline - 180:
                    raise TimeoutError("GPU function reached its cleanup reserve")
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        outcome.update(
            state="finished", campaign_report=str(destination / "campaign.json")
        )
    except BaseException as error:
        outcome.update(
            state="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        commit_stop.set()
        persistence.join(timeout=30)
        outcome.update(
            finished_at_unix=time.time(), elapsed_seconds=time.time() - started
        )
        (destination / "allocation_status.json").write_text(
            json.dumps(outcome, indent=2) + "\n"
        )
        runs.commit()
        reservations.put(
            campaign_id,
            dict(**outcome, artifacts=str(destination), commit=source_revision),
        )
        print(f"Campaign artifacts: {destination}", flush=True)
    return str(destination)


@app.local_entrypoint()
def main(
    campaign_id: str,
    validation_run: str,
    episode_seconds: float,
    rollouts: int,
    concurrency: int,
    checkpoint_every: int = 2,
    max_steps: int = 100000,
    evaluation_episodes: int = 8,
):
    import json
    import math
    import re
    import subprocess
    import time

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", campaign_id):
        raise ValueError(
            "Use a stable campaign ID with 3 to 80 lowercase letters, digits, underscores or hyphens"
        )
    if (
        not math.isfinite(episode_seconds)
        or episode_seconds <= 0
        or abs(round(episode_seconds * 10) - episode_seconds * 10) > 1e-8
        or rollouts < 2
        or concurrency < 1
        or checkpoint_every < 1
        or max_steps < checkpoint_every
        or evaluation_episodes not in (2, 4, 8)
    ):
        raise ValueError("Invalid explicit campaign settings")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Commit campaign sources before GPU allocation: " + dirty)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    settings = dict(
        episode_seconds=episode_seconds,
        rollouts=rollouts,
        concurrency=concurrency,
        checkpoint_every=checkpoint_every,
        max_steps=max_steps,
        evaluation_episodes=evaluation_episodes,
    )
    certificate = prerequisites.remote(validation_run)
    reserved = reservations.put(
        campaign_id,
        dict(
            state="reserved",
            commit=revision,
            reserved_at_unix=time.time(),
            settings=settings,
            validated_prerequisites=certificate,
        ),
        skip_if_exists=True,
    )
    if not reserved:
        raise RuntimeError(
            "This campaign ID was already attempted; refusing a duplicate GPU run"
        )
    print(
        json.dumps(
            dict(
                campaign_id=campaign_id,
                gpu=certificate["gpu"] + ":2",
                total_limit_hours=12,
                settings=settings,
            ),
            indent=2,
        )
    )
    print(
        campaign.with_options(gpu=certificate["gpu"] + ":2").remote(
            campaign_id, revision, certificate, settings
        )
    )


if __name__ == "__main__":
    raise SystemExit("Use the Modal CLI with this module's local entrypoint")
