"""Evaluate a completed Rev checkpoint in real Godot rides on one Modal L4.

Uses the training image and pinned Kev code, with the same bf16 fused serving
path as modal_rev_serve.py. Every control decision and full Godot recording is
kept, including failed attempts. No training or checkpoint changes occur.

  uvx --from modal==1.5.5 modal run --detach training/modal_rev_eval.py::evaluate \
      --checkpoint rev-4b-s1 --name rev-4b-s1-eval-modal
"""

import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_rev import HF, KEV_REF, KEV_ROOT, ROOT, RUNS, hf_cache, image, runs  # noqa: E402

REMOTE = "/opt/thunderhill"
GODOT = (Path.home() / ".local/share/thunderhill-tools/godot-4.7.2"
         / "Godot_v4.7.2-stable_linux.x86_64")
TRACKS = ("aragon,assen,balaton-park,brno,catalunya,goiania,jerez,le-mans,lusail,mandalika,misano,"
          "motegi,mugello,phillip-island,red-bull-ring,sachsenring,sepang,silverstone,thunderhill-east,valencia")
HELDOUT_TRACKS = "buriram,cota,portimao,laguna-seca"

app = modal.App("thunderhill-rev-eval")
eval_image = (
    image.apt_install("libfontconfig1")
    .uv_pip_install("gymnasium>=1.1")
    .add_local_file(str(GODOT), "/usr/local/bin/godot", copy=True)
    .add_local_dir(str(ROOT / "godot"), f"{REMOTE}/godot", copy=True,
                   ignore=[".godot", "builds", "**/.DS_Store"])
    .run_commands("chmod 755 /usr/local/bin/godot",
                  f"godot --headless --path {REMOTE}/godot --editor --import")
    .add_local_dir(str(ROOT / "training"), f"{REMOTE}/training", ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "tools"), f"{REMOTE}/tools", ignore=["__pycache__"])
    .add_local_python_source("modal_rev")
)


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"invalid run name: {value!r}")
    return value


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=1) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stop(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _card(checkpoint: Path, temperature: float, used: bool = False) -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8019/v1/models", timeout=10) as response:
        card = json.load(response)["models"][0]
    if Path(card["run"]).resolve() != checkpoint.resolve():
        raise RuntimeError(f"wrong checkpoint served: {card['run']}")
    if card["dtype"] != "bfloat16" or card["temperature"] != temperature:
        raise RuntimeError(f"wrong precision or checkpoint temperature: {card}")
    graphs = card.get("cuda_graphs")
    if graphs is None or graphs["failed"] or (used and not graphs["captured"]):
        raise RuntimeError(f"required CUDA graphs unavailable: {graphs}")
    return card


@app.function(image=eval_image, gpu="L4", cpu=8, memory=32768, timeout=3600,
              retries=0, single_use_containers=True, volumes={RUNS: runs, HF: hf_cache})
def evaluate_remote(checkpoint: str, name: str, tracks: str, heldout_tracks: str,
                    starts: int, heldout_starts: int, workers: int, source_commit: str) -> dict:
    from kev.checkpoint import read_meta

    runs.reload()
    checkpoint, name = _name(checkpoint), _name(name)
    out = Path(RUNS) / name
    prefix = Path(RUNS) / name
    if out.exists() or list(Path(RUNS).glob(f"{name}.*")):
        raise FileExistsError(f"{out} or its attempt artifacts exist; choose a fresh name")
    init = Path(RUNS) / checkpoint
    ck = init / "checkpoint"
    if not (init / "result.json").is_file() or not (ck / "head.pt").is_file():
        raise FileNotFoundError(f"no completed Rev checkpoint at {init}")
    result = json.loads((init / "result.json").read_text())
    metrics = json.loads((ck / "training_metrics.json").read_text())
    meta = read_meta(str(ck))
    actual_ref = subprocess.check_output(["git", "-C", KEV_ROOT, "rev-parse", "HEAD"], text=True).strip()
    if actual_ref != KEV_REF or result["kev_ref"] != KEV_REF:
        raise ValueError("checkpoint and serving Kev pins must match")
    if result["base"] != meta.base or result["base_revision"] != meta.base_revision:
        raise ValueError("checkpoint and training result base pins differ")
    if metrics["optimizer_steps"] <= 0:
        raise ValueError("checkpoint has no actual optimizer updates")
    hashes = {str(p.relative_to(init)): _hash(p) for p in sorted(init.rglob("*"))
              if p.is_file() and (p.is_relative_to(ck) or p.name == "result.json")}
    source_hashes = {}
    for directory in ("godot", "training", "tools"):
        for path in sorted((Path(REMOTE) / directory).rglob("*")):
            if path.is_file() and not set(path.parts) & {".godot", "__pycache__"}:
                source_hashes[str(path.relative_to(REMOTE))] = _hash(path)
    server_cmd = [sys.executable, f"{REMOTE}/training/rev_serve.py", "--run", str(ck),
                  "--port", "8019", "--max-batch", "8", "--gpu-memory-gb", "20",
                  "--dtype", "bf16", "--fused", "1"]
    drive_cmd = [sys.executable, f"{REMOTE}/training/rev_drive.py", "--godot", "/usr/local/bin/godot",
                 "--run-name", name, "--runs-dir", RUNS, "--rev-url", "http://127.0.0.1:8019",
                 "--tracks", tracks, "--heldout-tracks", heldout_tracks, "--starts", str(starts),
                 "--heldout-starts", str(heldout_starts), "--workers", str(workers)]
    provenance = dict(checkpoint=checkpoint, name=name, source_commit=source_commit, kev_ref=actual_ref,
                      base=meta.base, base_revision=meta.base_revision, temperature=meta.temperature,
                      dtype="bf16", quantization=None, fused=True, cuda_graphs_required=True,
                      optimizer_steps=metrics["optimizer_steps"], checkpoint_sha256=hashes,
                      source_sha256=source_hashes, tracks=tracks.split(","),
                      heldout_tracks=heldout_tracks.split(",") if heldout_tracks else [],
                      starts=starts, heldout_starts=heldout_starts, workers=workers,
                      server_command=server_cmd, drive_command=drive_cmd,
                      started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="starting")
    sidecars = {key: Path(f"{prefix}.{key}") for key in ("provenance.json", "server.log", "drive.log",
                                                           "server-initial.json", "server-final.json")}
    _write(sidecars["provenance.json"], provenance)
    server = driver = None
    server_log = driver_log = None
    began = time.monotonic()
    try:
        server_log = sidecars["server.log"].open("w")
        server = subprocess.Popen(server_cmd, cwd=KEV_ROOT, stdout=server_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                card = _card(ck, meta.temperature)
                break
            except OSError:
                if server.poll() is not None:
                    raise RuntimeError("Rev server failed to start; see server.log")
                if time.monotonic() - began > 600:
                    raise TimeoutError("Rev server startup exceeded 600 seconds")
                time.sleep(5)
        _write(sidecars["server-initial.json"], card)
        print(f"REV SERVER UP {checkpoint}: bf16, temperature {meta.temperature}, CUDA graphs required", flush=True)
        driver_log = sidecars["drive.log"].open("w")
        driver = subprocess.Popen(drive_cmd, cwd=REMOTE, stdout=driver_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        provenance["status"] = "evaluating"
        _write(sidecars["provenance.json"], provenance)
        offset, steps, episodes = 0, 0, set()
        while True:
            try:
                driver.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            card = _card(ck, meta.temperature)
            _write(sidecars["server-final.json"], card)
            decisions = out / "decisions.jsonl"
            if decisions.exists():
                with decisions.open() as stream:
                    stream.seek(offset)
                    while line := stream.readline():
                        if not line.endswith("\n"):
                            stream.seek(offset)
                            break
                        try:
                            row = json.loads(line)
                        except ValueError:
                            break
                        steps += 1
                        episodes.add(row["episode_id"])
                        offset = stream.tell()
            print(f"REV EVAL {name}: {steps} control steps, {len(episodes)} rides started, "
                  f"graphs {card['cuda_graphs']}", flush=True)
            if card["cuda_graphs"]["captured"] == 0 and steps > 0 and time.monotonic() - began > 180:
                raise RuntimeError("CUDA graphs did not capture after live simulator requests")
            if driver.returncode is not None:
                break
            if time.monotonic() - began > 3300:
                raise TimeoutError("evaluation exceeded 3300 seconds; preserving its partial rides")
        if driver.returncode:
            raise subprocess.CalledProcessError(driver.returncode, drive_cmd)
        _write(sidecars["server-final.json"], _card(ck, meta.temperature, used=True))
        rows = [json.loads(line) for line in (out / "eval/eval.jsonl").open()]
        expected = len(provenance["tracks"]) * starts + len(provenance["heldout_tracks"]) * heldout_starts
        if len(rows) != expected:
            raise RuntimeError(f"evaluation completed {len(rows)} rides, expected {expected}")
        summary = json.loads((out / "summary.json").read_text())
        provenance.update(status="complete", summary=summary)
    except Exception:
        provenance.update(status="failed", error=traceback.format_exc())
        print(provenance["error"], flush=True)
    finally:
        try:
            _stop(driver)
        finally:
            try:
                if server is not None and server.poll() is None:
                    try:
                        _write(sidecars["server-final.json"], _card(ck, meta.temperature))
                    except Exception as exc:
                        provenance["final_metadata_error"] = str(exc)
            finally:
                _stop(server)
        for stream in (driver_log, server_log):
            if stream is not None:
                stream.close()
        # rev_drive owns mkdir; a startup failure still gets a preserved attempt.
        out.mkdir(parents=True, exist_ok=True)
        provenance["wall_seconds"] = round(time.monotonic() - began, 1)
        _write(sidecars["provenance.json"], provenance)
        for key, sidecar in sidecars.items():
            if sidecar.exists():
                sidecar.rename(out / key)
        runs.commit()
        hf_cache.commit()
    return provenance


@app.function(image=image.add_local_python_source("modal_rev"), cpu=2, memory=8192,
              timeout=1800, volumes={RUNS: runs})
def pull_remote(name: str) -> bytes:
    """All recordings, decisions, logs, metadata, and partial attempt artifacts."""
    runs.reload()
    directory = Path(RUNS) / _name(name)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(directory.iterdir()):
            tar.add(path, arcname=path.name)
    return buffer.getvalue()


@app.local_entrypoint()
def evaluate(checkpoint: str = "rev-4b-s1", name: str = "rev-4b-s1-eval-modal", tracks: str = TRACKS,
             heldout_tracks: str = HELDOUT_TRACKS, starts: int = 4, heldout_starts: int = 4,
             workers: int = 8, source_commit: str = ""):
    checkpoint, name = _name(checkpoint), _name(name)
    local = ROOT / "runs/rev" / name
    if local.exists():
        raise SystemExit(f"{local} exists; choose a fresh name")
    if starts < 1 or heldout_starts < 1 or not 1 <= workers <= 8:
        raise SystemExit("positive start counts and 1 through 8 workers are required")
    if not source_commit:
        source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = evaluate_remote.remote(checkpoint, name, tracks, heldout_tracks, starts, heldout_starts,
                                    workers, source_commit)
    local.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(pull_remote.remote(name)), mode="r:gz") as tar:
        tar.extractall(local, filter="data")
    print(f"pulled every evaluation artifact to {local}", flush=True)
    if result["status"] != "complete":
        raise RuntimeError(f"evaluation failed; preserved artifacts at {local}: {result.get('error')}")
    trained = [entry for entry in result["summary"].values() if not entry["held_out"]]
    heldout = [entry for entry in result["summary"].values() if entry["held_out"]]
    print(f"REV EVAL {name}: {sum(e['laps'] for e in trained)}/{sum(e['rides'] for e in trained)} laps, "
          f"held out {sum(e['laps'] for e in heldout)}/{sum(e['rides'] for e in heldout)}", flush=True)
