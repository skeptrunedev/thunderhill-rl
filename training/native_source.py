"""Verify pinned NVIDIA source plus individually reviewed, versioned patches."""

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
PATCHES = (
    PATCH,
    PATCH.with_name("alpagym-padding-skip.patch"),
    PATCH.with_name("alpagym-host-replay.patch"),
)


def verify_source(source: Path, *, apply_patch: bool = False) -> dict:
    """Accept only the pinned base with the exact reviewed patch, never arbitrary dirt."""
    source = source.resolve()
    manifests = [json.loads(patch.with_suffix(".json").read_text()) for patch in PATCHES]
    for patch, manifest in zip(PATCHES, manifests):
        if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
            raise ValueError(f"Native patch does not match its manifest: {patch.name}")
        if manifest["upstream_revision"] != ALPAGYM_REVISION:
            raise ValueError(f"Native patch targets a different revision: {patch.name}")
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
    expected = set().union(*(set(item["files"]) for item in manifests))
    untracked = set(
        subprocess.check_output(
            ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"],
            text=True,
        ).splitlines()
    )
    allowed_added = set().union(*(set(item.get("added_files", [])) for item in manifests))
    if untracked - allowed_added:
        raise ValueError("NVIDIA checkout contains unreviewed untracked files")
    changed.update(untracked)
    if changed - expected:
        raise ValueError("NVIDIA checkout contains unreviewed modifications")
    for patch, manifest in zip(PATCHES, manifests):
        patch_files = set(manifest["files"])
        present = changed & patch_files
        if not present and apply_patch:
            subprocess.run(
                ["git", "-C", str(source), "apply", "--check", str(patch)], check=True
            )
            subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
            changed.update(patch_files)
        for name, digest in manifest["files"].items():
            if not (source / name).is_file() or hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Unreviewed or missing NVIDIA source modification: {name}")
    if changed != expected:
        raise ValueError(
            "NVIDIA checkout must contain exactly the reviewed patches; run tools/setup_alpagym.py"
        )
    from training.native_cosmos_source import patch_manifest

    return {
        "version": "reviewed_native_adaptations_v1",
        "upstream_revision": head,
        "patches": manifests,
        "cosmos_launcher": patch_manifest(),
    }


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
    attention_normalization = None
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
        if (
            expected_canonical.get("attn_implementation") == "flash_attention_2"
            and "attn_implementation" in actual_canonical
            and actual_canonical["attn_implementation"] is None
        ):
            # The pinned Transformers from_pretrained path materializes null
            # when no attention override is supplied during native export.
            # NVIDIA's inference loader explicitly supplies sdpa for both the
            # base and exported checkpoint. Prove the complete loaded configs
            # agree under that exact native override, rather than ignoring a
            # performance setting or accepting arbitrary backend changes.
            expected_loaded = ExpertModelConfig.from_dict(
                expected, attn_implementation="sdpa"
            ).to_dict()
            actual_loaded = ExpertModelConfig.from_dict(
                actual, attn_implementation="sdpa"
            ).to_dict()
            for metadata in ("_name_or_path", "transformers_version"):
                expected_loaded.pop(metadata, None)
                actual_loaded.pop(metadata, None)
            if expected_loaded == actual_loaded:
                attention_normalization = {
                    "source_serialized": "flash_attention_2",
                    "export_serialized": None,
                    "native_inference_override": "sdpa",
                    "complete_loaded_configs_equal": True,
                }
                expected_canonical = expected_loaded
                actual_canonical = actual_loaded
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
        source_config=str(release_config.absolute()),
        navigation_format="native_navigation_text_v1",
        comparison=comparison,
        attention_normalization=attention_normalization,
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
