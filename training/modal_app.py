"""Run the bounded Gemma validation on one Modal H100.

Launch from the repository root:
    modal run --detach training/modal_app.py --run-id gemma4-diagnostic-01

This module defines cloud resources but launches nothing when imported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
GODOT_SOURCE = Path(
    os.environ.get(
        "THUNDERHILL_GODOT",
        str(
            Path.home()
            / ".local/share/thunderhill-tools/godot-4.7.2/Godot_v4.7.2-stable_linux.x86_64"
        ),
    )
)
REMOTE_ROOT = Path("/opt/thunderhill")
RUNS = Path("/runs")
GODOT = "/usr/local/bin/godot"
PYTHON = "/opt/thunderhill/training/.venv/bin/python"
ARTIFACT_VOLUME = "thunderhill-runs-v2"
CACHE_VOLUME = "thunderhill-huggingface-v2"
STAGES = {
    "diagnostic": "modal_diagnostic.py",
    "warmstart": "modal_diagnostic.py",
    "warmstart-rl": "modal_warmstart_rl.py",
    "full-lap-smoke": "train_full_lap_grpo.py",
    "full-lap": "train_full_lap_grpo.py",
}
WARMSTART_DATASET = ROOT / "artifacts/lap-policy-dataset-v5-recovery"

app = modal.App("thunderhill-gemma-validation")
# Video job publication requires atomic hard links, supported by Volume v2.
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True, version=2)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.10.9")
    .add_local_file(
        str(ROOT / "training/pyproject.toml"),
        "/opt/thunderhill/training/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "training/uv.lock"), "/opt/thunderhill/training/uv.lock", copy=True
    )
    .run_commands("uv sync --frozen --project /opt/thunderhill/training --no-dev")
    .add_local_file(str(GODOT_SOURCE), GODOT, copy=True)
    .run_commands(
        "chmod 755 /usr/local/bin/godot", "/usr/local/bin/godot --headless --version"
    )
    .add_local_dir(
        str(ROOT / "godot"),
        "/opt/thunderhill/godot",
        copy=True,
        ignore=[".godot", "**/.DS_Store"],
    )
    .run_commands(
        "/usr/local/bin/godot --headless --path /opt/thunderhill/godot --editor --import"
    )
    .env(
        {
            "HF_HOME": "/model-cache",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        str(ROOT / "tools"),
        "/opt/thunderhill/tools",
        ignore=["__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        str(ROOT / "training"),
        "/opt/thunderhill/training",
        ignore=[".venv", "__pycache__", "**/*.pyc"],
    )
    .add_local_dir(str(WARMSTART_DATASET), "/opt/thunderhill/warmstart-data")
)


def validate_request(stage: str, run_id: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"Stage must be one of {sorted(STAGES)}")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError(
            "Run ID must be 1 to 80 letters, digits, underscores or hyphens"
        )
    return STAGES[stage]


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=1800,
    max_containers=1, retries=0,
    volumes={str(RUNS): artifacts, "/model-cache": cache},
)
def run(stage: str, run_id: str, source: dict, source_run: str = "") -> dict:
    if stage in ("full-lap", "full-lap-smoke"):
        raise ValueError("Full lap stages require run_campaign")
    return _execute_run(stage, run_id, source, source_run, 1800, 1680)


@app.function(
    image=image, gpu="H100", cpu=16, memory=131072, timeout=21600,
    max_containers=1, retries=0,
    volumes={str(RUNS): artifacts, "/model-cache": cache},
)
def run_campaign(stage: str, run_id: str, source: dict, source_run: str,
                 batch_candidates: str = "4,8,16,32,64") -> dict:
    if stage not in ("full-lap", "full-lap-smoke"):
        raise ValueError("Campaign stage must be full-lap or full-lap-smoke")
    return _execute_run(stage, run_id, source, source_run, 21600, 21480, batch_candidates)


def validate_source(stage, source_run):
    if stage in ("warmstart-rl", "full-lap", "full-lap-smoke"):
        validate_request("warmstart", source_run)
    elif source_run:
        raise ValueError("Source run requires a stage that resumes a checkpoint")


def validate_candidates(value):
    if not re.fullmatch(r"[1-9][0-9]*(,[1-9][0-9]*)*", value):
        raise ValueError("Batch candidates must be comma separated positive integers")
    sizes = [int(x) for x in value.split(",")]
    if len(sizes) != len(set(sizes)) or any(n > 64 for n in sizes):
        raise ValueError("Batch candidates must be unique and no larger than 64")


def _execute_run(stage, run_id, source, source_run, function_timeout, child_timeout,
                 batch_candidates="4,8,16,32,64"):
    script = validate_request(stage, run_id)
    validate_source(stage, source_run)
    validate_candidates(batch_candidates)
    artifacts.reload()
    # Reserve a unique parent, keeping the child's --output nonexistent as the
    # existing training CLIs require. Never replace an earlier run.
    if stage in ("full-lap", "full-lap-smoke"):
        adapter = RUNS / source_run / "experiment" / "grpo" / "adapter"
        for filename in ("adapter_model.safetensors", "adapter_config.json", "model_spec.json"):
            if not (adapter / filename).is_file():
                raise FileNotFoundError(f"Source adapter is incomplete: {adapter / filename}")
    root = RUNS / run_id
    root.mkdir(exist_ok=False)
    output = root / "experiment"
    command = [
        PYTHON,
        "-u",
        str(REMOTE_ROOT / "training" / script),
        "--godot",
        GODOT,
        "--output",
        str(output),
    ]
    if stage == "warmstart":
        command.extend(["--warmstart-dataset", "/opt/thunderhill/warmstart-data"])
    elif stage == "warmstart-rl":
        command.extend(["--source", str(RUNS / source_run / "experiment")])
    elif stage in ("full-lap", "full-lap-smoke"):
        command.extend([
            "--adapter", str(adapter), "--generations", "3", "--initial-generation", "2",
            "--time-budget-seconds", "900", "--batch-candidates", batch_candidates,
        ])
        if stage == "full-lap-smoke":
            command.append("--smoke")
    manifest = {
        "stage": stage,
        "run_id": run_id,
        "command": command,
        "source": source,
        "source_run": source_run or None,
        "gpu_requested": "H100",
        "function_timeout_seconds": function_timeout,
        "child_timeout_seconds": child_timeout,
        "artifact_volume": ARTIFACT_VOLUME,
        "video_rendering": "Download the complete run and render its video_jobs locally",
    }
    (root / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    artifacts.commit()
    started = time.monotonic()
    process = None
    status = {"ok": False, "run_id": run_id}
    try:
        with (root / "run.log").open("w", buffering=1) as log:
            # Inherit stdout so Modal logs remain live; the diagnostic owns its
            # detailed output. Capture the same bytes in a persistent log via
            # a small reader thread without shell interpolation.
            import threading

            process = subprocess.Popen(
                command,
                cwd=REMOTE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            def stream_output():
                assert process is not None and process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)

            reader = threading.Thread(target=stream_output, daemon=True)
            reader.start()
            try:
                deadline = time.monotonic() + child_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, child_timeout)
                    try:
                        returncode = process.wait(timeout=min(60, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        # Commit consistent filesystem snapshots while the child
                        # keeps writing. Live logs may be partial; immutable video
                        # jobs remain the authority for finished recordings.
                        artifacts.commit()
            except subprocess.TimeoutExpired:
                # Keep Godot alive while the trainer flushes and audits episodes.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status["timeout"] = True
                returncode = process.returncode
            reader.join(timeout=10)
            if reader.is_alive():
                raise RuntimeError("Child output remained open after termination")
        status.update(
            ok=returncode == 0 and not status.get("timeout", False),
            returncode=returncode,
            elapsed_seconds=time.monotonic() - started,
        )
        if not status["ok"]:
            raise RuntimeError(
                f"Stage {stage} exited {returncode}; inspect {run_id}/run.log"
            )
        return status
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        cache.commit()


@app.local_entrypoint()
def main(run_id: str, stage: str = "diagnostic", source_run: str = "",
         batch_candidates: str = "4,8,16,32,64"):
    validate_request(stage, run_id)
    validate_source(stage, source_run)
    validate_candidates(batch_candidates)
    source = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
    }
    if stage in ("full-lap", "full-lap-smoke"):
        result = run_campaign.remote(stage, run_id, source, source_run, batch_candidates)
    else:
        result = run.remote(stage, run_id, source, source_run)
    print(json.dumps(result, indent=2))
