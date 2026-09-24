"""Verify the pinned NVIDIA source plus our explicit navigation transport patch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ALPAGYM_REVISION = "972d160eed0e23d388497851504a3a233fec5879"
PATCH = (
    Path(__file__).resolve().parents[1]
    / "tools/patches/alpagym-native-navigation.patch"
)
MANIFEST = PATCH.with_suffix(".json")


def verify_source(source: Path, *, apply_patch: bool = False) -> dict:
    """Accept only the pinned base with the exact reviewed patch, never arbitrary dirt."""
    source = source.resolve()
    manifest = json.loads(MANIFEST.read_text())
    if hashlib.sha256(PATCH.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Navigation patch does not match its manifest")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != ALPAGYM_REVISION:
        raise ValueError(f"Expected AlpaGym {ALPAGYM_REVISION}, found {head}")
    changed = set(
        subprocess.check_output(
            ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True
        ).splitlines()
    )
    expected = set(manifest["files"])
    if not changed and apply_patch:
        subprocess.run(
            ["git", "-C", str(source), "apply", "--check", str(PATCH)], check=True
        )
        subprocess.run(["git", "-C", str(source), "apply", str(PATCH)], check=True)
        changed = expected
    untracked = set(
        subprocess.check_output(
            ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"],
            text=True,
        ).splitlines()
    )
    if untracked - set(manifest.get("added_files", [])):
        raise ValueError("NVIDIA checkout contains unreviewed untracked files")
    changed.update(untracked)
    if changed != expected:
        raise ValueError(
            "NVIDIA checkout must contain exactly the reviewed navigation patch; run tools/setup_alpagym.py"
        )
    for name, digest in manifest["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unreviewed NVIDIA source modification: {name}")
    return manifest


def validate_navigation_checkpoint(checkpoint: Path) -> dict:
    """Reject checkpoints without verified Alpamayo 1.5 conversion provenance."""
    marker = checkpoint / "navigation_provenance.json"
    if not marker.is_file():
        raise ValueError(
            "Navigation requires verified Alpamayo 1.5 provenance; run tools/check_native_navigation.py with the release config"
        )
    provenance = json.loads(marker.read_text())
    if provenance.get("source_model_type") != "alpamayo1_5" or provenance.get(
        "source_architectures"
    ) != ["Alpamayo1_5"]:
        raise ValueError(
            "Navigation is supported only for the verified Alpamayo 1.5 source model"
        )
    if (
        provenance.get("source_config_sha256")
        != "824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e"
    ):
        raise ValueError(
            "Navigation source config is not the reviewed Alpamayo 1.5 release"
        )
    if provenance.get("navigation_format") != "native_navigation_text_v1":
        raise ValueError("Checkpoint navigation provenance uses an unsupported format")
    actual = hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()
    if actual != provenance.get("converted_config_sha256"):
        raise ValueError(
            "Checkpoint config changed after navigation provenance verification"
        )
    return provenance


def record_navigation_checkpoint(checkpoint: Path, release_config: Path) -> dict:
    """Prove the checkpoint config is NVIDIA's conversion of the reviewed 1.5 release.

    This records architecture provenance, not a claim that arbitrary weight files
    have been authenticated. Training exports must run this check again.
    """
    import runpy
    import alpagym_alpamayo_r1

    release_bytes = release_config.read_bytes()
    release = json.loads(release_bytes)
    actual = json.loads((checkpoint / "config.json").read_text())
    converter_path = (
        Path(alpagym_alpamayo_r1.__file__).resolve().parents[2]
        / "scripts/convert_release_to_alpagym_checkpoint.py"
    )
    converter = runpy.run_path(str(converter_path))
    expected = converter["build_expert_config"](release, actual["vlm_name_or_path"])
    comparison = "exact_native_conversion"
    if actual != expected:
        # Official save_pretrained materializes default configuration fields and
        # changes path/version metadata. Canonicalize with NVIDIA's own config
        # class rather than allowing an open ended list of extra model fields.
        from alpamayo1_x_rl.models.expert_model.config import ExpertModelConfig

        expected_canonical = ExpertModelConfig(**expected).to_dict()
        actual_canonical = ExpertModelConfig(**actual).to_dict()
        for metadata in ("_name_or_path", "transformers_version"):
            expected_canonical.pop(metadata, None)
            actual_canonical.pop(metadata, None)
        if actual_canonical != expected_canonical:
            differences = sorted(
                key
                for key in set(actual_canonical) | set(expected_canonical)
                if actual_canonical.get(key) != expected_canonical.get(key)
            )
            raise ValueError(
                f"Checkpoint architecture differs from the reviewed 1.5 release: {differences}"
            )
        comparison = "native_config_canonicalization"

    provenance = dict(
        source_model_type=release.get("model_type"),
        source_architectures=release.get("architectures"),
        source_config_sha256=hashlib.sha256(release_bytes).hexdigest(),
        converted_config_sha256=hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        source_config=str(release_config.resolve()),
        navigation_format="native_navigation_text_v1",
        comparison=comparison,
    )
    if provenance["source_model_type"] != "alpamayo1_5" or provenance[
        "source_architectures"
    ] != ["Alpamayo1_5"]:
        raise ValueError("The supplied release is not Alpamayo 1.5")
    if (
        provenance["source_config_sha256"]
        != "824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e"
    ):
        raise ValueError(
            "The supplied release config is not the reviewed Alpamayo 1.5 revision"
        )
    (checkpoint / "navigation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return validate_navigation_checkpoint(checkpoint)
