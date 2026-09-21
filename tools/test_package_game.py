"""Packaging must reject stale source geometry and stale baked scene bytes."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import package_game


class BakeValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "godot"
        root_patch = patch.object(package_game, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        for name in ("scenery", "landmarks"):
            source = self.project / f"scripts/{name}.gd"
            scene = self.project / f"assets/generated/{name}.scn"
            source.parent.mkdir(parents=True, exist_ok=True)
            scene.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"original source")
            scene.write_bytes(b"original baked scene")
            self.write_manifest(name, {
                "schema_version": 1,
                "sources": {
                    f"scripts/{name}.gd": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
                "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            })

    def write_manifest(self, name, manifest):
        path = self.project / f"data/{name}-bake.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))

    def test_valid_scenes_and_existing_scenery_entrypoint(self):
        package_game.validate_scenery_bake()
        package_game.validate_bakes()

    def test_source_mismatch_rejected_for_each_bake(self):
        for name in ("scenery", "landmarks"):
            with self.subTest(name=name):
                source = self.project / f"scripts/{name}.gd"
                source.write_bytes(b"changed source")
                with self.assertRaisesRegex(RuntimeError, f"Stale {name} bake: scripts/{name}.gd"):
                    package_game.validate_bakes()
                source.write_bytes(b"original source")

    def test_scene_mismatch_rejected_for_each_bake(self):
        for name in ("scenery", "landmarks"):
            with self.subTest(name=name):
                scene = self.project / f"assets/generated/{name}.scn"
                scene.write_bytes(b"changed scene")
                with self.assertRaisesRegex(RuntimeError, f"Stale {name} bake: assets/generated/{name}.scn"):
                    package_game.validate_bakes()
                scene.write_bytes(b"original baked scene")

    def test_missing_scene_rejected(self):
        (self.project / "assets/generated/landmarks.scn").unlink()
        with self.assertRaisesRegex(RuntimeError, "Stale landmarks bake"):
            package_game.validate_bakes()

    def test_malformed_manifests_rejected(self):
        for manifest in ([], {}, {"schema_version": 1, "sources": []},
                         {"schema_version": 1, "sources": {}},
                         {"schema_version": 2, "sources": {"source": "digest"}},
                         {"schema_version": 1, "sources": {"source": "digest"}}):
            with self.subTest(manifest=manifest):
                self.write_manifest("landmarks", manifest)
                with self.assertRaisesRegex(RuntimeError, "Invalid landmarks bake manifest"):
                    package_game.validate_bakes()

    def test_missing_and_invalid_json_manifest_rejected(self):
        path = self.project / "data/landmarks-bake.json"
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Invalid landmarks bake manifest"):
            package_game.validate_bakes()
        path.write_text("{")
        with self.assertRaisesRegex(RuntimeError, "Invalid landmarks bake manifest"):
            package_game.validate_bakes()


class PavementToneValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "godot"
        root_patch = patch.object(package_game, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.sources = {
            "assets/materials/pavement_tone.png": b"source pixels",
            "assets/materials/pavement_tone.res": b"runtime resource",
            "tools/bake_pavement_tone.gd": b"numeric mip builder",
            "data/track.json": b"track source",
        }
        hashes = {}
        for name, content in self.sources.items():
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        self.assets = self.project / "assets/materials"
        (self.assets / "pavement_tone.json").write_text(json.dumps({
            "output_sha256": hashes["assets/materials/pavement_tone.png"],
            "track_sha256": hashes["data/track.json"],
        }))
        (self.assets / "pavement_tone_runtime.json").write_text(json.dumps({
            "source_sha256": hashes["assets/materials/pavement_tone.png"],
            "output_sha256": hashes["assets/materials/pavement_tone.res"],
            "builder_sha256": hashes["tools/bake_pavement_tone.gd"],
        }))

    def test_valid_bake(self):
        package_game.validate_pavement_tone()

    def test_changed_input_resource_builder_and_track_rejected(self):
        for name, content in self.sources.items():
            with self.subTest(name=name):
                path = self.project / name
                path.write_bytes(b"modified bytes")
                with self.assertRaisesRegex(RuntimeError, "Invalid pavement tone bake"):
                    package_game.validate_pavement_tone()
                path.write_bytes(content)

    def test_missing_and_malformed_runtime_manifest(self):
        path = self.assets / "pavement_tone_runtime.json"
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Invalid pavement tone bake"):
            package_game.validate_pavement_tone()
        for content in ["{", "[]", "{}"]:
            path.write_text(content)
            with self.assertRaisesRegex(RuntimeError, "Invalid pavement tone bake"):
                package_game.validate_pavement_tone()


if __name__ == "__main__":
    unittest.main()
