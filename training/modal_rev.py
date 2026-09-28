"""Train Rev on Modal: a Qwen base model with a fresh LoRA and pointer head, from scratch.

Rev is our Jev-style decision model for riding (records from rev_dataset.py). It uses
the open-source Kev code (github.com/jaredpalmer/kev, pinned at KEV_REF) as its
architecture, trainer and scorer, but its weights start from the plain Qwen base: no
Kev checkpoint (--init_from) and no Kev replay data. The recipe is the from-scratch
first stage Kev documents for its 0.8B size (2 epochs, lr 1e-4, batch 8, LoRA r16,
bf16 autocast over fp32 masters).

DAgger rounds continue from the previous round's Rev (--init-from NAME, a run on this
app's volume; nothing outside it is accepted, so a round can never start from Kev's
weights) for --epochs 1 at a lower learning rate.

After training, one temperature is fitted on calibration.jsonl (min NLL on raw logits,
kev.metrics.fit_temperature) and written into the checkpoint, and development.jsonl is
scored at that temperature. Everything is pulled to runs/rev/NAME/.

  uvx --from modal==1.5.5 modal run --detach training/modal_rev.py::train \\
      --data runs/rev/data-v1 --name rev-0.8b-v1
  uvx --from modal==1.5.5 modal run training/modal_rev.py::train \\
      --data runs/rev/data-r1 --name rev-0.8b-r1 --init-from rev-0.8b-v1 --epochs 1 --lr 5e-5
"""

import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

KEV_REF = "9c41005b2180347c3c646dfc9e50c4428483ec6b"   # the local serving checkout uses the same commit
KEV_ROOT, RUNS, HF = "/kev", "/runs", "/hf"
BASES = {  # Qwen base checkpoints, pinned to the revisions Kev's recipes use
    "0.8b": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "4b": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
}
RECIPES = {"0.8b": dict(epochs=2, lr=1e-4, batch=8, accum=1, checkpointing=0),
           "4b": dict(epochs=2, lr=5e-5, batch=4, accum=2, checkpointing=1)}
PARTITIONS = ("train", "calibration", "development")
ROOT = Path(__file__).resolve().parents[1]

app = modal.App("thunderhill-rev")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/jaredpalmer/kev.git {KEV_ROOT} "
                  f"&& git -C {KEV_ROOT} checkout --quiet {KEV_REF}")
    .uv_pip_install(f"kev[serve] @ file://{KEV_ROOT}")
    # Gated DeltaNet kernels for the Qwen3.5 hybrid bases (same pins as Kev's Modal app).
    .uv_pip_install("flash-linear-attention==0.5.2", "triton>=3.7.1")
    .env({"HF_HOME": HF, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1", "TRITON_CACHE_DIR": f"{HF}/triton-cache"})
)
runs = modal.Volume.from_name("thunderhill-rev-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("thunderhill-rev-hf", create_if_missing=True)


def stage(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@app.function(image=image, gpu="H100", cpu=4, memory=(32768, 131072), timeout=4 * 3600, retries=0,
              volumes={RUNS: runs, HF: hf_cache})
def train_remote(name: str, data: dict, size: str, epochs: int, seed: int, init_from: str = "",
                 lr: float = 0.0) -> dict:
    import torch
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions, read_meta, write_meta
    from kev.data import load_records
    from kev.metrics import fit_temperature, grouped_metrics
    from kev.predictors import LocalPredictor

    out = Path(RUNS) / name
    if (out / "result.json").exists():
        raise FileExistsError(f"/runs/{name} is a finished run; choose a new name")
    if out.exists():  # an attempt that died before scoring: start it over
        import shutil
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    for part, text in data.items():
        (out / "data" / f"{part}.jsonl").write_text(text)
    base, revision = BASES[size]
    recipe = {**RECIPES[size], **({"epochs": epochs} if epochs else {}), **({"lr": lr} if lr else {})}
    checkpoint = out / "checkpoint"
    init = None
    if init_from:
        init = Path(RUNS) / init_from / "checkpoint"
        result = json.loads((Path(RUNS) / init_from / "result.json").read_text())
        if not (init / "head.pt").exists() or result.get("base") != base:
            raise ValueError(f"--init-from must be a Rev run on this volume with base {base}: {init_from}")
    cmd = [sys.executable, "-m", "kev.train", "--data", str(out / "data/train.jsonl"),
           "--base", base, "--base_revision", revision, "--epochs", recipe["epochs"], "--lr", recipe["lr"],
           "--batch", recipe["batch"], "--accum", recipe["accum"], "--checkpointing", recipe["checkpointing"],
           "--dtype", "bf16", "--device", "cuda", "--seed", seed, "--out", checkpoint]
    if init:
        cmd += ["--init_from", init]
    cmd = [str(c) for c in cmd]
    stage("training: " + " ".join(cmd[2:]))
    started = time.time()
    with (out / "train.log").open("w") as log, subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=KEV_ROOT) as proc:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("ep", "saved", "device", "dropped")) or "Error" in line or "Traceback" in line:
                print(line.rstrip(), flush=True)
    runs.commit()
    hf_cache.commit()
    if proc.returncode:
        raise RuntimeError("kev.train failed; see train.log")

    calibration = load_records(out / "data/calibration.jsonl")
    development = load_records(out / "data/development.jsonl")
    stage(f"calibrating on {len(calibration)} and scoring {len(development)} development records")
    predictor = LocalPredictor(str(checkpoint), "cuda", LoadOptions(temperature=1.0))
    _, calibration_rows = evaluate_records(calibration, predictor, out / "calibration")
    temperature = fit_temperature(calibration_rows, aggregation="micro")
    report, rows = evaluate_records(development, predictor, out / "development", temperature)
    meta = read_meta(str(checkpoint))
    meta.temperature = temperature
    meta.extra["temperature_fit"] = {"rows": "calibration.jsonl", "n": len(calibration), "value": temperature,
                                     "method": "min NLL, micro, kev.metrics.fit_temperature"}
    write_meta(str(checkpoint), meta)
    clean = [r for r in rows if r["variant"] == "clean"]
    result = {"name": name, "model": f"rev-{size}", "base": base, "base_revision": revision, "init_from": init_from or None,
              "kev_ref": KEV_REF, "recipe": recipe, "seed": seed,
              "records": {part: text.count("\n") for part, text in data.items()},
              "temperature": temperature, "development": {
                  "raw": report["clean"], "calibrated": report["calibrated_clean"],
                  "per_question": grouped_metrics(clean, "question", temperature)},
              "wall_seconds": round(time.time() - started), "gpu": torch.cuda.get_device_name(0)}
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    runs.commit()
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={RUNS: runs})
def pull_remote(name: str) -> bytes:
    """The run directory (checkpoint, reports, logs) without the uploaded data, as a tar."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted((Path(RUNS) / name).iterdir()):
            if path.name != "data":
                tar.add(path, arcname=path.name)
    return buffer.getvalue()


@app.local_entrypoint()
def train(data: str, name: str, size: str = "0.8b", epochs: int = 0, seed: int = 0, init_from: str = "",
          lr: float = 0.0):
    if size not in BASES:
        raise SystemExit(f"--size must be one of {sorted(BASES)}")
    local = ROOT / "runs/rev" / name
    if local.exists():
        raise SystemExit(f"{local} exists; choose a new name")
    texts = {part: (Path(data) / f"{part}.jsonl").read_text() for part in PARTITIONS}
    result = train_remote.remote(name, texts, size, epochs, seed, init_from, lr)
    local.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(pull_remote.remote(name)), mode="r:gz") as tar:
        tar.extractall(local, filter="data")
    (local / "data_source.txt").write_text(str(Path(data).resolve()) + "\n")
    dev = result["development"]
    print(f"\n{result['model']} ({name}) from {result['base']}, temperature {result['temperature']:.3f}")
    print(f"development: accuracy {dev['calibrated']['acc']:.3f}  brier {dev['calibrated']['brier']:.3f}  "
          f"ece {dev['calibrated']['ece']:.3f}")
    for question, metrics in dev["per_question"].items():
        print(f"  {question:15s} accuracy {metrics['acc']:.3f}  brier {metrics['brier']:.3f}  ece {metrics['ece']:.3f}")
    print(f"pulled to {local}")
