"""Verify the pinned Cosmos launcher and its opt-in fixed-job supervision patch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

REVISION = "d2a2c57c4bd6496482bc42da19a59b4432705eda"
PATCH = Path(__file__).resolve().parents[1] / "tools/patches/cosmos-static-failfast.patch"


def patch_manifest() -> dict:
    manifest = json.loads(PATCH.with_suffix(".json").read_text())
    if manifest["upstream_revision"] != REVISION:
        raise ValueError("Cosmos launcher patch has an unreviewed upstream revision")
    if hashlib.sha256(PATCH.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Cosmos launcher patch differs from its reviewed manifest")
    return manifest


def verify_launcher(root: Path, *, apply_patch: bool = False) -> dict:
    """Verify exact source before applying the reviewed patch to an owned runtime."""
    manifest = patch_manifest()
    states = []
    for name, digests in manifest["files"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest == digests["original_sha256"]:
            states.append("original")
        elif digest == digests["patched_sha256"]:
            states.append("patched")
        else:
            raise ValueError(f"Unreviewed Cosmos launcher source: {name}")
    if states != ["patched"] * len(states):
        if not apply_patch or states != ["original"] * len(states):
            raise ValueError("Cosmos launcher requires the complete reviewed supervision patch")
        command = ["git", "apply"]
        subprocess.run([*command, "--check", str(PATCH)], cwd=root, check=True)
        subprocess.run([*command, str(PATCH)], cwd=root, check=True)
        return verify_launcher(root)
    return manifest


def verify_installed(*, apply_patch: bool = False) -> dict:
    distribution = importlib.metadata.distribution("cosmos-rl")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError("Installed Cosmos does not match NVIDIA's pinned git revision")
    if direct_url.get("url") != "https://github.com/nvidia-cosmos/cosmos-rl.git":
        raise ValueError("Installed Cosmos has an unreviewed source repository")
    root = Path(distribution.locate_file(""))
    return verify_launcher(root, apply_patch=apply_patch)


if __name__ == "__main__":
    print(json.dumps(verify_installed(apply_patch=True), indent=2))
