"""Verify the pinned Cosmos launcher and its opt-in fixed-job supervision patch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

REVISION = "d2a2c57c4bd6496482bc42da19a59b4432705eda"
PATCH = (
    Path(__file__).resolve().parents[1] / "tools/patches/cosmos-static-failfast.patch"
)


def _manifest(patch: Path) -> dict:
    manifest = json.loads(patch.with_suffix(".json").read_text())
    if manifest["upstream_revision"] != REVISION:
        raise ValueError("Cosmos launcher patch has an unreviewed upstream revision")
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Cosmos launcher patch differs from its reviewed manifest")
    return manifest


LIFETIME_PATCH = PATCH.with_name("cosmos-receive-lifetime.patch")


def patch_manifest() -> dict:
    return {
        "version": "reviewed_cosmos_runtime_v2",
        "upstream_revision": REVISION,
        "patches": [_manifest(PATCH), _manifest(LIFETIME_PATCH)],
    }


def verify_launcher(root: Path, *, apply_patch: bool = False) -> dict:
    for patch in (PATCH, LIFETIME_PATCH):
        _verify_patch(root, patch, apply_patch=apply_patch)
    return patch_manifest()


def _verify_patch(root: Path, patch: Path, *, apply_patch: bool = False) -> dict:
    """Verify exact source before applying the reviewed patch to an owned runtime."""
    manifest = _manifest(patch)
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
            raise ValueError(
                "Cosmos launcher requires the complete reviewed supervision patch"
            )
        # Installed packages may live in a venv below a git worktree. Apply
        # from the filesystem anchor with an explicit destination, not git's
        # current-subdirectory prefix (which silently skips these paths).
        destination = root.resolve()
        command = [
            "git",
            "apply",
            "--directory",
            str(destination.relative_to(destination.anchor)),
        ]
        subprocess.run(
            [*command, "--check", str(patch)], cwd=destination.anchor, check=True
        )
        subprocess.run([*command, str(patch)], cwd=destination.anchor, check=True)
        return _verify_patch(root, patch)
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
