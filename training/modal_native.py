"""Run the pinned NVIDIA workspace on a bounded pair of Modal GPUs."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/opt/thunderhill"
UPSTREAM = "/opt/alpagym"
REVISION = "972d160eed0e23d388497851504a3a233fec5879"
GODOT = Path.home() / ".local/share/thunderhill-tools/godot-4.7.2/Godot_v4.7.2-stable_linux.x86_64"
app = modal.App("thunderhill-native-validation")
cache = modal.Volume.from_name("thunderhill-huggingface-v2")
runs = modal.Volume.from_name("thunderhill-runs-v2")
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "curl", "git", "git-lfs", "ninja-build",
        "redis-server", "libnccl-dev=2.26.2-1+cuda12.8", "libnccl2=2.26.2-1+cuda12.8",
        "xvfb", "xauth", "libgl1-mesa-dri", "libglx-mesa0", "libvulkan1",
        "mesa-vulkan-drivers", "libxcursor1", "libxinerama1", "libxi6", "libxrandr2",
        "libfontconfig1", "ffmpeg",
    )
    .pip_install("uv==0.11.2")
    .env({
        "MAX_JOBS": "2", "UV_CONCURRENT_BUILDS": "1", "NVTE_BUILD_MAX_JOBS": "2",
        "NVTE_BUILD_THREADS_PER_JOB": "1", "TORCH_CUDA_ARCH_LIST": "9.0",
        "PYTHONUNBUFFERED": "1",
    })
    .run_commands(
        "git lfs install --system --skip-smudge",
        f"git clone https://github.com/NVlabs/alpagym.git {UPSTREAM}",
        f"git -C {UPSTREAM} checkout --detach {REVISION}",
        f"uv sync --frozen --all-packages --project {UPSTREAM}",
        f"uv pip check --python {UPSTREAM}/.venv/bin/python",
    )
    .add_local_file(str(GODOT), "/usr/local/bin/godot", copy=True)
    .add_local_dir(str(ROOT / "godot"), REMOTE + "/godot", copy=True,
                   ignore=[".godot", "builds", "**/.DS_Store"])
    .run_commands("chmod 755 /usr/local/bin/godot",
                  f"godot --headless --path {REMOTE}/godot --editor --import")
    .env({"HF_HOME": "/model-cache", "HF_HUB_OFFLINE": "1",
          "PYTHONPATH": f"{REMOTE}:{REMOTE}/training:{REMOTE}/tools",
          "THUNDERHILL_AGENT_OFFSCREEN": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT / "training"), REMOTE + "/training",
                   ignore=["alpamayo", "__pycache__", "results"])
    .add_local_dir(str(ROOT / "tools"), REMOTE + "/tools", ignore=["__pycache__"])
)


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/model-cache": cache}, include_source=False)
def prepare_checkpoint():
    """Convert cached release with NVIDIA's converter before GPU allocation."""
    import subprocess
    from pathlib import Path

    checkpoint = Path("/model-cache/alpagym-converted-1.5")
    release = Path("/model-cache/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/7aba8293c09993f2e125c6819df05d7fa3e873ea")
    processor = Path("/model-cache/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9")
    if not checkpoint.exists():
        subprocess.run([
            UPSTREAM + "/.venv/bin/python",
            UPSTREAM + "/packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py",
            "--input", str(release), "--output", str(checkpoint),
            "--vlm-name-or-path", str(processor),
        ], check=True)
    cache.commit()
    return str(checkpoint)


@app.local_entrypoint()
def main():
    print(prepare_checkpoint.remote())
