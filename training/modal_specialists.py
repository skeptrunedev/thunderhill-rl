"""Specialist SAC lanes on Modal: one container per circuit, each running
sac_specialists.py (the same sac_async recipe, stop rule and apex bar as the
local unit) on headless Godot actors and a small-GPU learner.

  uvx --from modal==1.5.5 modal deploy training/modal_specialists.py
  uvx --from modal==1.5.5 modal run training/modal_specialists.py::upload
  uvx --from modal==1.5.5 modal run training/modal_specialists.py::launch --tracks jerez,le-mans
  uvx --from modal==1.5.5 modal run training/modal_specialists.py::status
  uvx --from modal==1.5.5 modal run training/modal_specialists.py::sync

The image carries Godot 4.7.2, the main checkout's godot/ project with every
circuit's generated imagery and meshes as built here (they are not committed and
are not rebuilt remotely); sac_async.py's dependencies are resolved by uv when a
lane starts (~50 s; warming them into the image did not make uv reuse them). The
volume
thunderhill-specialists is mounted as the repo's runs/: upload puts the general
warm-start checkpoint there, and each lane keeps its run in
runs/sac/specialists-modal/specialist-TRACK with its lane registry
runs/sac/specialists-modal/TRACK.json.

launch spawns the deployed lane function once per circuit, so the lanes live on
the deployed app and never depend on this client. A lane that fails or is
preempted resumes from its volume checkpoint (and replay buffer, when its last
shutdown saved one); launching a circuit again resumes it the same way.

sync pulls each published lane's best checkpoint, evaluation rows and best-lap
recordings into the same paths here, recomputes the apex bar locally with
tools/apex_report.py, and publishes circuits that meet it to
runs/sac/specialists.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/opt/thunderhill"
GODOT = (Path.home() / ".local/share/thunderhill-tools/godot-4.7.2"
         / "Godot_v4.7.2-stable_linux.x86_64")
WARM_START = "runs/sac/sac-multitrack-1/checkpoints/step_40672729.pt"
LANES = "runs/sac/specialists-modal"
CPU, WORKERS, GPU = 16.0, 24, "T4"

app = modal.App("thunderhill-specialists")
volume = modal.Volume.from_name("thunderhill-specialists", create_if_missing=True)
# stop::TRACK -> True asks that circuit's lane to stop cleanly (checkpoint and replay
# buffer saved); a running container does not see files put on the volume meanwhile.
control = modal.Dict.from_name("thunderhill-specialists-control", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "gzip", "libfontconfig1")
    .pip_install("uv==0.11.2")
    .env({"PYTHONUNBUFFERED": "1", "UV_LINK_MODE": "copy"})
    .add_local_file(str(GODOT), "/usr/local/bin/godot", copy=True)
    .add_local_dir(str(ROOT / "godot"), f"{REMOTE}/godot", copy=True,
                   ignore=[".godot", "builds", "**/.DS_Store"])
    # The import cache headless workers load resources from.
    .run_commands("chmod 755 /usr/local/bin/godot",
                  f"godot --headless --path {REMOTE}/godot --editor --import")
    .add_local_dir(str(ROOT / "training"), f"{REMOTE}/training", ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "tools"), f"{REMOTE}/tools", ignore=["__pycache__"])
)


@app.function(image=image, gpu=GPU, cpu=CPU, memory=32768, timeout=24 * 3600,
              retries=modal.Retries(max_retries=5, initial_delay=30.0),
              volumes={f"{REMOTE}/runs": volume})
def lane(track: str, workers: int = WORKERS, extra: list[str] | None = None):
    """sac_specialists.py on one circuit until it meets the bar (or is stuck)."""
    command = ["uv", "run", "--script", f"{REMOTE}/training/sac_specialists.py",
               "--godot", "/usr/local/bin/godot", "--tracks", track, "--reuse", "",
               "--runs-dir", f"{REMOTE}/{LANES}", "--registry", f"{REMOTE}/{LANES}/{track}.json",
               "--workers", str(workers), "--min-free-gb", "0", "--no-wandb", *(extra or [])]
    print(" ".join(command), flush=True)
    control.pop(f"stop::{track}", None)
    run_dir = Path(f"{REMOTE}/{LANES}/specialist-{track}")
    process = subprocess.Popen(command, cwd=REMOTE)
    while process.poll() is None:
        time.sleep(30)
        if control.get(f"stop::{track}", False):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "STOP").touch()  # sac_specialists stops its trainer cleanly
            control.pop(f"stop::{track}", None)
    volume.commit()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


@app.function(image=image, cpu=2, memory=4096, timeout=900)
def check_image():
    """How long sac_async's script environment takes to set up (a lane resolves it
    at start, ~50 s), and that Godot runs."""
    started = time.monotonic()
    help_text = subprocess.run(["uv", "run", "--script", f"{REMOTE}/training/sac_async.py", "--help"],
                               cwd=REMOTE, capture_output=True, text=True, check=True)
    version = subprocess.run(["godot", "--headless", "--version"], capture_output=True, text=True)
    return dict(seconds=round(time.monotonic() - started, 1),
                downloaded="Downloading" in help_text.stderr, godot=version.stdout.strip(),
                focus_file="--focus-file" in help_text.stdout)


@app.local_entrypoint()
def check():
    print(check_image.remote())


@app.local_entrypoint()
def stop(tracks: str):
    """Ask these circuits' lanes to stop cleanly; launch resumes them."""
    for track in (t.strip() for t in tracks.split(",") if t.strip()):
        control[f"stop::{track}"] = True
        print(f"{track}: stop requested")


@app.local_entrypoint()
def upload():
    """The general warm-start checkpoint onto the volume (once)."""
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(ROOT / WARM_START), WARM_START.removeprefix("runs/"))
    print(f"uploaded {WARM_START}")


@app.local_entrypoint()
def launch(tracks: str, workers: int = WORKERS, extra: str = ""):
    """Spawn one lane per circuit on the deployed app."""
    deployed = modal.Function.from_name(app.name, "lane")
    for track in (t.strip() for t in tracks.split(",") if t.strip()):
        call = deployed.spawn(track, workers, extra.split() if extra else None)
        print(f"{track}: {call.object_id}")


def _read(path: str) -> bytes | None:
    try:
        return b"".join(volume.read_file(path.removeprefix("runs/")))
    except FileNotFoundError:
        return None


def _download(path: str):
    local = ROOT / path
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("wb") as stream:
        volume.read_file_into_fileobj(path.removeprefix("runs/"), stream)


def _lane_tracks(lanes: str) -> list[str]:
    return sorted(Path(entry.path).name.removeprefix("specialist-")
                  for entry in volume.listdir(lanes.removeprefix("runs/"))
                  if Path(entry.path).name.startswith("specialist-"))


@app.local_entrypoint()
def status(lanes: str = LANES):
    """Each lane's phase, latest and best evaluation (apex values from the lane)."""
    sys.path.insert(0, str(ROOT / "training"))
    from sac_specialists import describe, rank

    for track in _lane_tracks(lanes):
        run = f"{lanes}/specialist-{track}"
        state = json.loads(_read(f"{run}/specialist.json") or b'{"phase": "learn"}')
        rows = [json.loads(line) for line in (_read(f"{run}/eval/eval.jsonl") or b"").splitlines()]
        apex = {r["checkpoint"]: r for r in map(json.loads, (_read(f"{run}/apex.jsonl") or b"")
                                                .splitlines())}
        groups = {}
        for row in rows:
            groups.setdefault(row["checkpoint"], []).append(row)
        evals = []
        for checkpoint, group in groups.items():
            laps = [r["lap_time_s"] for r in group if r["lap_time_s"] is not None]
            evals.append(dict(checkpoint=checkpoint, steps=group[0]["policy_step"], laps=len(laps),
                              best_lap_s=min(laps) if laps else None,
                              progress_m=sum(r["legal_progress_m"] for r in group) / len(group),
                              **{k: apex.get(checkpoint, {}).get(k) for k in (
                                  "apexes_hit", "apexes_total", "missed_apex_stations_m",
                                  "upright_median_throttle")}))
        published = _read(f"{lanes}/{track}.json")
        print(f"{track}: {state['phase']}{' PUBLISHED' if published else ''}, {len(evals)} evals")
        if evals:
            print(f"  latest {describe(evals[-1])}\n  best   {describe(max(evals, key=rank))}")


@app.local_entrypoint()
def sync(starts: int = 6, lanes: str = LANES, registry: str = "runs/sac/specialists.json"):
    """Pull published lanes and publish those meeting the bar, measured here."""
    sys.path.insert(0, str(ROOT / "training"))
    from sac_specialists import evaluations, meets_bar, publish_entry, summary

    registry = ROOT / registry
    for track in _lane_tracks(lanes):
        lane_entry = _read(f"{lanes}/{track}.json")
        if lane_entry is None:
            continue
        lane_entry = json.loads(lane_entry)[track]
        run = f"{lanes}/specialist-{track}"
        for name in ("eval/eval.jsonl", "specialist.json", "train.log"):
            _download(f"{run}/{name}")
        run_dir = ROOT / run
        rows = [json.loads(line) for line in (run_dir / "eval/eval.jsonl").open()]
        best_rows = [r for r in rows if r["checkpoint"] == Path(lane_entry["checkpoint"])
                     .relative_to(run).as_posix()]
        _download(lane_entry["checkpoint"])
        for row in best_rows:
            if row.get("recording"):
                _download(f"{run}/{row['recording']}")
        (run_dir / "apex.jsonl").unlink(missing_ok=True)
        evaluation, = evaluations(run_dir, starts, track, only=best_rows[0]["checkpoint"])
        entry = dict(lane_entry, **summary(evaluation))
        entry.update(checkpoint=lane_entry["checkpoint"], meets_bar=meets_bar(evaluation, starts),
                     synced_from="modal")
        if not entry["meets_bar"]:
            print(f"{track}: lane published but the local apex check fails: "
                  f"{json.dumps(summary(evaluation))}")
            continue
        publish_entry(registry, track, entry)
        print(f"{track}: published {entry['checkpoint']} ({entry['best_lap_s']} s, "
              f"{entry['apexes_hit']}/{entry['apexes_total']} apexes)")
