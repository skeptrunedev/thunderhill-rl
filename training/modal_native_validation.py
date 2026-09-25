"""Allocate bounded native validation GPUs after CPU preparation has succeeded."""

import modal

from training.modal_native import REMOTE, ROOT, UPSTREAM, cache, image, runs

app = modal.App("thunderhill-native-gpu-validation")

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


@app.function(
    image=runtime_image,
    gpu="L40S:2",
    cpu=16,
    memory=131072,
    timeout=3600,
    retries=0,
    volumes={"/model-cache": cache, "/runs": runs},
    include_source=False,
)
def validate_gpu(source_revision: str):
    import json
    import os
    import subprocess
    import uuid
    from pathlib import Path

    os.chdir(REMOTE)
    os.environ["WANDB_MODE"] = "offline"
    destination = Path("/runs") / ("native-validation-" + uuid.uuid4().hex)
    destination.mkdir()
    (destination / "source_identity.json").write_text(
        json.dumps(
            {
                "repository": "skeptrunedev/thunderhill-rl",
                "commit": source_revision,
                "clean_worktree_at_launch": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validation artifacts: {destination}", flush=True)
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
                    "--seconds",
                    "5",
                    "--budget",
                    "2400",
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3450,
            )
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


@app.local_entrypoint()
def main():
    import subprocess

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Commit validation sources before GPU launch: " + dirty)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(validate_gpu.remote(revision))
