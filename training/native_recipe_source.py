"""Verify the pinned NVIDIA recipe and complete prompt padding mask patch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

REVISION = "1d99fc50370c96637157455da386323a44663c9d"
PATCH = (
    Path(__file__).resolve().parents[1] / "tools/patches/alpamayo-padding-mask.patch"
)


def patch_manifest() -> dict:
    manifest = json.loads(PATCH.with_suffix(".json").read_text())
    if manifest["upstream_revision"] != REVISION:
        raise ValueError("Recipe padding patch has an unreviewed revision")
    if hashlib.sha256(PATCH.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Recipe padding patch differs from its reviewed manifest")
    return manifest


def verify_recipe(root: Path, *, apply_patch: bool = False) -> dict:
    manifest = patch_manifest()
    states = []
    for name, digests in manifest["files"].items():
        path = root / name
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if digest == digests["original_sha256"]:
            states.append("original")
        elif digest == digests["patched_sha256"]:
            states.append("patched")
        else:
            raise ValueError(f"Unreviewed native recipe source: {name}")
    if states != ["patched"] * len(states):
        if not apply_patch or states != ["original"] * len(states):
            raise ValueError(
                "Native recipe requires the complete reviewed padding mask patch"
            )
        # A venv can live below a git checkout. Applying from that subdirectory
        # otherwise silently skips package-relative paths outside git's prefix.
        destination = root.resolve()
        command = [
            "git",
            "apply",
            "--directory",
            str(destination.relative_to(destination.anchor)),
        ]
        subprocess.run(
            [*command, "--check", str(PATCH)], cwd=destination.anchor, check=True
        )
        subprocess.run([*command, str(PATCH)], cwd=destination.anchor, check=True)
        return verify_recipe(root)
    return manifest


def verify_installed(*, apply_patch: bool = False) -> dict:
    distribution = importlib.metadata.distribution("alpamayo1-x-rl")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (
        origin.get("vcs_info", {}).get("commit_id") != REVISION
        or origin.get("url") != "https://github.com/NVlabs/alpamayo-recipes.git"
        or origin.get("subdirectory") != "recipes/alpamayo1_x_rl"
    ):
        raise ValueError("Installed native recipe differs from NVIDIA's pinned source")
    return verify_recipe(Path(distribution.locate_file("")), apply_patch=apply_patch)


if __name__ == "__main__":
    print(json.dumps(verify_installed(apply_patch=True), indent=2))
