"""Train Rev on Modal: a Qwen base model with a fresh LoRA and pointer head, from scratch.

Rev is our Jev-style decision model for riding (records from rev_dataset.py). It uses
the open-source Kev code (github.com/jaredpalmer/kev, pinned at KEV_REF) as its
architecture, trainer and scorer, but its weights start from the plain Qwen base: no
Kev checkpoint (--init_from) and no Kev replay data. The recipe is the from-scratch
first stage Kev documents for its 0.8B size (2 epochs, lr 1e-4, batch 8, LoRA r16,
bf16 autocast over fp32 masters).

DAgger rounds continue from the previous round's Rev (--init-from NAME) for
--epochs 1 at a lower learning rate. --upload-init verifies and uploads a completed
local Rev parent to the volume without replacing any existing checkpoint files.

After training, one temperature is fitted on calibration.jsonl (min NLL on raw logits,
kev.metrics.fit_temperature) and written into the checkpoint, and development.jsonl is
scored at that temperature. --preserve-action-temperature instead keeps the parent's
temperature 1.0 for continuous action decoding and reports the fit as diagnostic.
Everything is pulled to runs/rev/NAME/.

  uvx --from modal==1.5.5 modal run --detach training/modal_rev.py::train \\
      --data runs/rev/data-v1 --name rev-0.8b-v1
  uvx --from modal==1.5.5 modal run training/modal_rev.py::train \\
      --data runs/rev/data-r1 --name rev-0.8b-r1 --init-from rev-0.8b-v1 --epochs 1 --lr 5e-5
"""

import io
import hashlib
import json
import math
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


def run_name(value: str) -> str:
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("use a Rev run name, without a directory path")
    return value


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_parent(parent: Path, size: str) -> tuple[dict, dict]:
    """Validate a completed Rev result and actual trainer output without loading a GPU."""
    result = json.loads((parent / "result.json").read_text())
    checkpoint = parent / "checkpoint"
    metrics = json.loads((checkpoint / "training_metrics.json").read_text())
    config = json.loads((checkpoint / "training_config.json").read_text())
    base, revision = BASES[size]
    if (result.get("model") != f"rev-{size}" or result.get("base") != base
            or result.get("base_revision") != revision or result.get("kev_ref") != KEV_REF
            or config["args"].get("base") != base or config.get("base_revision") != revision
            or not (checkpoint / "head.pt").is_file()):
        raise ValueError(f"parent must be a completed Rev {size} run with the pinned base and Kev: {parent}")
    steps = metrics.get("optimizer_steps")
    if not isinstance(steps, int) or steps < 1 or result.get("optimizer_steps", steps) != steps:
        raise ValueError(f"parent has inconsistent or absent actual optimizer updates: {parent}")
    if result.get("training_metrics", metrics).get("optimizer_steps") != steps:
        raise ValueError(f"parent training metrics differ from the actual trainer output: {parent}")
    if (checkpoint / "adapter_config.json").exists():
        weights = (checkpoint / "adapter_model.safetensors").is_file()
    else:
        weights = (checkpoint / "config.json").is_file() and any(checkpoint.glob("model*.safetensors"))
    if not weights or not (checkpoint / "tokenizer.json").is_file():
        raise ValueError(f"parent checkpoint is missing weights or tokenizer: {parent}")
    return result, metrics


def upload_parent(name: str, size: str):
    """Upload immutable local checkpoint files; verify and reuse identical remote files."""
    parent = ROOT / "runs/rev" / run_name(name)
    _, metrics = validate_parent(parent, size)
    files = {"result.json": parent / "result.json"}
    for path in sorted((parent / "checkpoint").rglob("*")):
        if path.is_symlink():
            raise ValueError(f"parent checkpoint must not contain symlinks: {path}")
        if path.is_file():
            files[path.relative_to(parent).as_posix()] = path
    try:
        existing = runs.listdir(name, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        existing = []
    remote_files = {str(Path(entry.path.lstrip("/")).relative_to(name)) for entry in existing
                    if entry.type == modal.volume.FileEntryType.FILE
                    and (Path(entry.path.lstrip("/")).relative_to(name).parts[0] == "checkpoint"
                         or Path(entry.path).name == "result.json")}
    if remote_files - files.keys():
        raise ValueError(f"remote parent contains unexpected immutable files: {sorted(remote_files - files.keys())}")
    missing = []
    for relative, path in files.items():
        if relative not in remote_files:
            missing.append((relative, path))
            continue
        digest = hashlib.sha256()
        for chunk in runs.read_file(f"{name}/{relative}"):
            digest.update(chunk)
        if digest.hexdigest() != file_digest(path):
            raise ValueError(f"remote parent differs from local checkpoint: {name}/{relative}")
    if missing:
        with runs.batch_upload(force=False) as batch:
            for relative, path in missing:
                batch.put_file(str(path), f"{name}/{relative}")
    stage(f"verified parent {name}: {metrics['optimizer_steps']} actual optimizer updates, {len(files)} immutable files")


# 12 h: Rev-4B takes ~4x Rev-0.8B's hour per 60k records and epoch.
@app.function(image=image, gpu="H100", cpu=4, memory=(32768, 131072), timeout=12 * 3600, retries=0,
              volumes={RUNS: runs, HF: hf_cache})
def train_remote(name: str, data: dict, size: str, epochs: int, seed: int, init_from: str = "",
                 lr: float = 0.0, source_commit: str = "", preserve_action_temperature: bool = False,
                 shared_prefix: int = 0) -> dict:
    try:
        return _train_remote(name, data, size, epochs, seed, init_from, lr, source_commit,
                             preserve_action_temperature, shared_prefix)
    finally:
        # Preserve logs and partial checkpoints even if training or scoring fails.
        try:
            runs.commit()
        finally:
            hf_cache.commit()


def _train_remote(name: str, data: dict, size: str, epochs: int, seed: int, init_from: str,
                  lr: float, source_commit: str, preserve_action_temperature: bool,
                  shared_prefix: int) -> dict:
    import torch
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions, read_meta, write_meta
    from kev.data import load_records
    from kev.metrics import fit_temperature, grouped_metrics
    from kev.predictors import LocalPredictor

    runs.reload()
    name = run_name(name)
    out = Path(RUNS) / name
    if (out / "result.json").exists():
        raise FileExistsError(f"/runs/{name} is a finished run; choose a new name")
    if out.exists():
        out.rename(out.with_name(f"{name}.incomplete-{time.time_ns()}"))
    (out / "data").mkdir(parents=True)
    for part, text in data.items():
        (out / "data" / f"{part}.jsonl").write_text(text)
    base, revision = BASES[size]
    recipe = {**RECIPES[size], **({"epochs": epochs} if epochs else {}), **({"lr": lr} if lr else {}),
              "shared_prefix": shared_prefix}
    checkpoint = out / "checkpoint"
    init = None
    inherited_temperature = 1.0
    if init_from:
        parent = Path(RUNS) / run_name(init_from)
        result, _ = validate_parent(parent, size)
        init = parent / "checkpoint"
        parent_meta = read_meta(str(init))
        if (parent_meta.base, parent_meta.base_revision) != (base, revision):
            raise ValueError("parent head and result base pins differ")
        inherited_temperature = parent_meta.temperature
        if inherited_temperature != result["temperature"]:
            raise ValueError("parent head and result temperatures differ")
    if preserve_action_temperature and inherited_temperature != 1.0:
        raise ValueError("preserving action temperature requires a parent temperature of 1.0")
    cmd = [sys.executable, "-m", "kev.train", "--data", str(out / "data/train.jsonl"),
           "--base", base, "--base_revision", revision, "--epochs", recipe["epochs"], "--lr", recipe["lr"],
           "--batch", recipe["batch"], "--accum", recipe["accum"], "--checkpointing", recipe["checkpointing"],
           "--shared_prefix", recipe["shared_prefix"],
           "--dtype", "bf16", "--device", "cuda", "--seed", seed, "--out", checkpoint]
    if init:
        cmd += ["--init_from", init]
    cmd = [str(c) for c in cmd]
    data_sha256 = {part: hashlib.sha256(text.encode("utf-8")).hexdigest() for part, text in data.items()}
    config = {"name": name, "init_from": init_from or None, "base": base, "base_revision": revision,
              "kev_ref": KEV_REF, "recipe": recipe, "seed": seed, "source_commit": source_commit,
              "data_sha256": data_sha256, "command": cmd, "training_backend": "modal",
              "preserve_action_temperature": preserve_action_temperature}
    if init:
        config["parent_result_sha256"] = file_digest(parent / "result.json")
    (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")
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
    training_metrics = json.loads((checkpoint / "training_metrics.json").read_text())
    if training_metrics.get("optimizer_steps", 0) < 1:
        raise RuntimeError("kev.train saved no actual optimizer updates")

    calibration = load_records(out / "data/calibration.jsonl")
    development = load_records(out / "data/development.jsonl")
    stage(f"calibrating on {len(calibration)} and scoring {len(development)} development records")
    predictor = LocalPredictor(str(checkpoint), "cuda", LoadOptions(temperature=1.0))
    _, calibration_rows = evaluate_records(calibration, predictor, out / "calibration")
    fitted = fit_temperature(calibration_rows, aggregation="micro")
    report, rows = evaluate_records(development, predictor, out / "development", fitted)
    temperature = inherited_temperature if preserve_action_temperature else fitted
    meta = read_meta(str(checkpoint))
    meta.temperature = temperature
    fit = {"rows": "calibration.jsonl", "n": len(calibration), "value": fitted,
           "method": "min NLL, micro, kev.metrics.fit_temperature", "applied": not preserve_action_temperature}
    if preserve_action_temperature:
        fit["reason"] = "Rev decodes soft action probabilities into continuous controls; preserve temperature 1.0"
    meta.extra["temperature_fit"] = fit
    write_meta(str(checkpoint), meta)
    clean = [r for r in rows if r["variant"] == "clean"]
    result = {"name": name, "model": f"rev-{size}", "base": base, "base_revision": revision, "init_from": init_from or None,
              "kev_ref": KEV_REF, "recipe": recipe, "seed": seed,
              "records": {part: sum(bool(line.strip()) for line in text.splitlines()) for part, text in data.items()},
              "temperature": temperature, "calibration_temperature": fitted, "temperature_fit": fit,
              "optimizer_steps": training_metrics["optimizer_steps"], "training_metrics": training_metrics,
              "source_commit": source_commit, "data_sha256": data_sha256, "training_backend": "modal",
              "development": {
                  "raw": report["clean"], "calibrated": report["calibrated_clean"],
                  "per_question": grouped_metrics(clean, "question", temperature),
                  "raw_per_question": grouped_metrics(clean, "question", 1.0),
                  "calibrated_per_question": grouped_metrics(clean, "question", fitted)},
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
          lr: float = 0.0, upload_init: bool = False, preserve_action_temperature: bool = False,
          shared_prefix: int = 0):
    if size not in BASES:
        raise SystemExit(f"--size must be one of {sorted(BASES)}")
    name = run_name(name)
    if init_from:
        init_from = run_name(init_from)
    if shared_prefix not in (0, 1) or epochs < 0 or not math.isfinite(lr) or lr < 0:
        raise SystemExit("shared-prefix must be 0 or 1; epochs and lr must be nonnegative and finite")
    if upload_init and not init_from:
        raise SystemExit("--upload-init requires --init-from")
    if name == init_from:
        raise SystemExit("new run name must differ from its parent")
    local = ROOT / "runs/rev" / name
    if local.exists():
        if (local / "result.json").exists():
            raise SystemExit(f"{local} is complete; choose a new name")
        local.rename(local.with_name(f"{name}.incomplete-{time.time_ns()}"))
    texts = {part: (Path(data) / f"{part}.jsonl").read_text() for part in PARTITIONS}
    if any(not text.strip() for text in texts.values()):
        raise SystemExit("all data partitions must contain records")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if upload_init:
        upload_parent(init_from, size)
    result = train_remote.remote(name, texts, size, epochs, seed, init_from, lr, source_commit,
                                 preserve_action_temperature, shared_prefix)
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
