"""Bounded real CUDA retention and native NCCL qualification, without training."""

from pathlib import Path

import modal

from training.modal_native import REMOTE, ROOT, UPSTREAM, image, runs

app = modal.App("thunderhill-native-replay-qualification")
runtime_image = image.env({"NCCL_DEBUG": "INFO"})
for relative in (
    "training/native_source.py",
    "training/native_network.py",
    "tools/setup_alpagym.py",
    "tools/check_native_replay_storage.py",
    "tools/patches/alpagym-native-navigation.patch",
    "tools/patches/alpagym-native-navigation.json",
    "tools/patches/alpagym-padding-skip.patch",
    "tools/patches/alpagym-padding-skip.json",
    "tools/patches/alpagym-host-replay.patch",
    "tools/patches/alpagym-host-replay.json",
):
    runtime_image = runtime_image.add_local_file(
        str(ROOT / relative), REMOTE + "/" + relative
    )


@app.function(
    image=runtime_image,
    gpu="H100:2",
    cpu=8,
    memory=32768,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=1,
    volumes={"/runs": runs},
    include_source=False,
    serialized=True,
)
def qualify(captured_payload: bytes, source_revision: str):
    import json
    import os
    import subprocess
    import uuid

    os.chdir(REMOTE)
    destination = Path("/runs") / ("native-replay-qualification-" + uuid.uuid4().hex)
    destination.mkdir()
    python = UPSTREAM + "/.venv/bin/python"
    report = dict(state="running", source_revision=source_revision, training=False)
    (destination / "input.pt").write_bytes(captured_payload)
    try:
        subprocess.run(
            [python, "tools/setup_alpagym.py", "--checkout", UPSTREAM, "--source-only"],
            check=True,
            timeout=90,
        )
        subprocess.run(
            [
                python,
                "-c",
                "import json,torch; from training.native_network import configure_container_hostname; "
                "print(json.dumps(configure_container_hostname())); "
                "assert torch.cuda.device_count()==2; "
                "assert torch.cuda.can_device_access_peer(0,1) and torch.cuda.can_device_access_peer(1,0)",
            ],
            check=True,
            timeout=60,
        )
        with (destination / "snapshot.log").open("w") as log:
            subprocess.run(
                [
                    python,
                    "-m",
                    "tools.check_native_replay_storage",
                    "--input",
                    str(destination / "input.pt"),
                    "--output",
                    str(destination / "cuda-retention.json"),
                    "--device",
                    "cuda:0",
                    "--decisions",
                    "64",
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        # Reuse NVIDIA's existing real two-process NCCL roundtrip harness.
        # This adds no unit tests and uses actual pynccl on the GPU pair.
        with (destination / "nccl.log").open("w") as log:
            subprocess.run(
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    UPSTREAM
                    + "/packages/runtime/src/alpagym_runtime/transport/nccl/tests/"
                    "test_real_nccl_subprocess.py::test_real_nccl_subprocess_roundtrip",
                    "--basetemp",
                    str(destination / "nccl-artifacts"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        report.update(
            state="passed",
            cuda_retention=json.loads(
                (destination / "cuda-retention.json").read_text()
            ),
            native_nccl_roundtrip=True,
        )
    except BaseException as error:
        report.update(state="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        runs.commit()
        print(f"Replay qualification artifacts: {destination}", flush=True)
    return dict(artifact_root=str(destination), **report)


@app.local_entrypoint()
def main(model_input: str, output: str):
    import json
    import subprocess

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Commit sources before GPU qualification: " + dirty)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = qualify.remote(Path(model_input).read_bytes(), revision)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
