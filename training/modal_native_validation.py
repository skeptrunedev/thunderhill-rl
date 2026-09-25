"""Allocate bounded native validation GPUs after CPU preparation has succeeded."""

import modal

from training.modal_native import REMOTE, ROOT, UPSTREAM, cache, image, runs

app = modal.App("thunderhill-native-gpu-validation")

runtime_image = (
    image.env({
        "NVIDIA_DRIVER_CAPABILITIES": "all", "NCCL_DEBUG": "INFO",
        "ALPAGYM_SKIP_ALL_PADDING_MINIBATCHES": "1",
    })
    .add_local_file(
        str(ROOT / "artifacts/native-navigation-verification/native_model_input.pt"),
        REMOTE + "/native-replay-fixture.pt",
    )
    .add_local_dir(
        str(ROOT / "training"),
        REMOTE + "/training",
        ignore=["alpamayo", ".venv", "__pycache__", "results", "*.md"],
    )
    .add_local_dir(
        str(ROOT / "tools"), REMOTE + "/tools", ignore=["__pycache__", "*.md"]
    )
)


def repair_archived_recordings(run_dir, destination, deadline):
    """Retry reviewed video failures only in a stopped archive, within this allocation."""
    import hashlib
    import json
    import os
    import signal
    import subprocess
    import time
    from pathlib import Path

    from training.native_campaign import summarize

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(Path("/runs/native-campaigns").resolve()):
        raise ValueError("Repair target must be an archived native campaign")
    status_path = run_dir / "run_status.json"
    original_status = status_path.read_bytes()
    status = json.loads(original_status)
    launch = json.loads((run_dir / "launch_manifest.json").read_text())
    if (status.get("runtime_stopped") is not True
            or status.get("state") not in {"completed", "failed"}
            or not status.get("run_id") or status["run_id"] != launch.get("run_id")):
        raise ValueError("Recording repair requires a matching stopped terminal campaign")
    remaining = deadline - time.monotonic()
    receipt = {"run_dir": str(run_dir), "status": "starting",
               "original_run_status_sha256": hashlib.sha256(original_status).hexdigest(),
               "before": summarize(run_dir), "budget_seconds": max(0, remaining)}
    receipt_path = destination / "recording-repair.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    process = None
    try:
        if remaining <= 1:
            raise TimeoutError("Fresh validation left no recording repair budget")
        command = ["xvfb-run", "-a", "-s", "-screen 0 800x600x24",
                   UPSTREAM + "/.venv/bin/python", REMOTE + "/tools/render_video_queue.py",
                   str(run_dir), "--godot", "godot", "--ffmpeg", "ffmpeg",
                   "--retry-failed", "--workers", "2"]
        with (destination / "recording-repair.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        receipt["returncode"] = process.returncode
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        if not summarize(run_dir)["complete_video_coverage"]:
            raise RuntimeError("Recording repair finished without complete coverage")
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if process is not None:
            # Godot/ffmpeg inherit this dedicated process group. Stop the group
            # even if its wrapper already exited; never leave a timed out render.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        receipt["after"] = summarize(run_dir)
        receipt["original_run_status_unchanged"] = status_path.read_bytes() == original_status
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if not receipt["original_run_status_unchanged"]:
            raise RuntimeError("Archived campaign status changed during recording repair")


@app.function(
    image=runtime_image,
    gpu="H100:2",
    cpu=16,
    memory=393216,
    timeout=3600,
    retries=0,
    volumes={"/model-cache": cache, "/runs": runs},
    include_source=False,
)
def validate_gpu(source_revision: str, resume_run_dir: str = "", seconds: float = 5, concurrency: int = 1, repair_recordings: str = ""):
    import json
    import os
    import subprocess
    import time
    import uuid
    from pathlib import Path

    repair_deadline = time.monotonic() + 3450
    os.chdir(REMOTE)
    os.environ["WANDB_MODE"] = "offline"
    if resume_run_dir and (not resume_run_dir.startswith("/runs/native-validation-") or ".." in Path(resume_run_dir).parts):
        raise ValueError("Resume path must identify an existing native validation run")
    destination = Path("/runs") / ("native-validation-" + uuid.uuid4().hex)
    destination.mkdir()
    (destination / "source_identity.json").write_text(
        json.dumps(
            {
                "repository": "skeptrunedev/thunderhill-rl",
                "commit": source_revision,
                "clean_worktree_at_launch": True,
                "resume_run_dir": resume_run_dir or None,
                "repair_recordings": repair_recordings or None,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validation artifacts: {destination}", flush=True)
    from training.native_network import configure_container_hostname

    (destination / "container-network.json").write_text(
        json.dumps(configure_container_hostname(), indent=2) + "\n"
    )
    resources = (destination / "gpu-resources.csv").open("w")
    sampler = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv", "-l", "5",
    ], stdout=resources, stderr=subprocess.STDOUT)
    try:
        with (destination / "gpu-topology.txt").open("w") as log:
            topology = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        (destination / "gpu-topology-status.json").write_text(
            json.dumps(
                {
                    "exit_code": topology.returncode,
                    "note": "NVML topology visibility is diagnostic; actual CUDA peer access is mandatory below.",
                }
            )
            + "\n"
        )
        subprocess.run(
            [
                UPSTREAM + "/.venv/bin/python",
                "-c",
                (
                    "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count() == 2; "
                    "assert torch.cuda.can_device_access_peer(0,1) and "
                    "torch.cuda.can_device_access_peer(1,0), 'CUDA peer access unavailable'"
                ),
            ],
            check=True,
        )
        with (destination / "supervisor.log").open("w") as log:
            subprocess.run(
                [
                    "xvfb-run",
                    "-a",
                    "-s",
                    "-screen 0 800x600x24",
                    UPSTREAM + "/.venv/bin/python",
                    "-m",
                    "training.native_validation",
                    "run",
                    "--source",
                    UPSTREAM,
                    "--model",
                    "/model-cache/alpagym-converted-1.5",
                    "--output",
                    str(destination),
                    "--replay-fixture",
                    REMOTE + "/native-replay-fixture.pt",
                    "--seconds",
                    str(seconds),
                    "--concurrency",
                    str(concurrency),
                    "--initial-speed-m-s",
                    "8",
                    "--budget",
                    "3300",
                    *(["--resume-run-dir", resume_run_dir] if resume_run_dir else []),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3450,
            )
        if repair_recordings:
            repair_archived_recordings(repair_recordings, destination, repair_deadline)
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sampler.wait()
        resources.close()
        runs.commit()
        print(f"Validation artifacts: {destination}")
    return str(destination)


@app.function(
    image=runtime_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=600,
    retries=0,
    volumes={"/model-cache": cache, "/runs": runs},
    include_source=False,
)
def diagnose_inference(source_revision: str, dispatch: str, config: str,
                       hard_deadline_unix: float,
                       checkpoint: str = "/model-cache/alpagym-converted-1.5"):
    """One bounded exact observation replay, never an optimizer or a new campaign."""
    import json
    import os
    import subprocess
    import time
    import traceback
    import uuid
    from pathlib import Path

    os.chdir(REMOTE)
    for value in (dispatch, config):
        if not value.startswith("/runs/") or ".." in Path(value).parts:
            raise ValueError("Diagnostic observation/config must be existing /runs artifacts")
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    if not checkpoint.startswith(("/runs/", "/model-cache/")) or ".." in Path(checkpoint).parts:
        raise ValueError("Checkpoint must be an existing mounted native bundle")
    started = time.time()
    remaining = min(570, int(hard_deadline_unix - started - 30))
    if remaining < 60:
        raise ValueError("Original campaign deadline leaves insufficient diagnostic time")
    destination = Path("/runs/native-diagnostics") / ("exact-" + uuid.uuid4().hex)
    destination.mkdir(parents=True)
    receipt = {
        "diagnostic_only": True, "training_eligible": False,
        "source_revision": source_revision, "dispatch": dispatch, "config": config,
        "checkpoint": checkpoint,
        "checkpoint_identity": "base expert with frozen VLM" if checkpoint == "/model-cache/alpagym-converted-1.5" else "explicit supplied native bundle",
        "exact_trained_expert_state_claimed": False,
        "CUDA_LAUNCH_BLOCKING": "1 (isolated diagnostic process only)",
        "hard_deadline_unix": hard_deadline_unix,
        "subprocess_budget_seconds": remaining, "status": "starting",
    }
    receipt_path = destination / "supervisor.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Inference diagnostic artifacts: {destination}", flush=True)
    # Apply exactly the reviewed source and recipe adaptations before any model
    # imports. This mirrors native_validation without starting its trainer/game.
    bootstrap = (
        "import json,runpy,sys; from pathlib import Path; "
        "from training.native_source import verify_source; "
        "from training.native_recipe_source import verify_installed; "
        f"proof={{'runtime':verify_source(Path({UPSTREAM!r}),apply_patch=True),"
        "'recipe':verify_installed(apply_patch=True)}; "
        f"Path({str(destination / 'native-source.json')!r}).write_text(json.dumps(proof,indent=2)); "
        "sys.argv=['tools.check_native_inference',*sys.argv[1:]]; "
        "runpy.run_module('tools.check_native_inference',run_name='__main__')"
    )
    command = [
        UPSTREAM + "/.venv/bin/python", "-c", bootstrap,
        "--dispatch", dispatch, "--config", config, "--checkpoint", checkpoint,
        "--output", str(destination / "inference"), "--device", "cuda:0",
        "--repeats", "10", "--max-seconds", str(remaining), "--synchronize",
    ]
    try:
        with (destination / "inference.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=remaining, check=False,
                                     env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"})
        receipt.update(status="completed" if process.returncode == 0 else "failed",
                       returncode=process.returncode)
    except subprocess.TimeoutExpired:
        receipt.update(status="bounded_timeout")
    except BaseException:
        receipt.update(status="supervisor_failed", error=traceback.format_exc())
        raise
    finally:
        receipt["elapsed_seconds"] = time.time() - started
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        runs.commit()
        print(f"Inference diagnostic artifacts: {destination}", flush=True)
    return {"artifacts": str(destination), **receipt}


@app.local_entrypoint()
def main(resume_run_dir: str = "", seconds: float = 5, concurrency: int = 1,
         diagnostic_dispatch: str = "", diagnostic_config: str = "",
         hard_deadline_unix: float = 0,
         diagnostic_checkpoint: str = "/model-cache/alpagym-converted-1.5",
         repair_recordings: str = ""):
    import subprocess

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Commit validation sources before GPU launch: " + dirty)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if diagnostic_dispatch:
        if repair_recordings:
            raise ValueError("Recording repair runs only after successful full qualification")
        import time
        if not diagnostic_config or hard_deadline_unix - time.time() < 90:
            raise ValueError("Diagnostic mode requires config and original campaign deadline")
        print(diagnose_inference.remote(revision, diagnostic_dispatch, diagnostic_config,
                                       hard_deadline_unix, diagnostic_checkpoint))
    else:
        print(validate_gpu.remote(revision, resume_run_dir, seconds, concurrency, repair_recordings))
