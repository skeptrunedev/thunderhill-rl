"""Offline path validation for resumed Modal experiments."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import modal_app
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    modal_app = None


@unittest.skipUnless(modal_app, "Modal SDK is not installed in this interpreter")
class SourceCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runs = Path(self.temporary.name)
        self.patch = patch.object(modal_app, "RUNS", self.runs)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def adapter(self, checkpoint):
        directory = self.runs / "source" / "experiment" / checkpoint
        directory.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "adapter_config.json", "model_spec.json"):
            (directory / name).write_text("test")
        return directory

    def test_native_source_and_later_campaign_paths(self):
        source = self.adapter("checkpoint-0003")
        self.assertEqual(modal_app.source_adapter("native-tools-validation", "source", "checkpoint-0003"), source)
        migrated = self.adapter("native-migration/adapter")
        self.assertEqual(modal_app.source_adapter("full-lap", "source", "native-migration/adapter"), migrated)

    def test_legacy_default_retained(self):
        expected = self.adapter("grpo/adapter")
        self.assertEqual(modal_app.source_adapter("full-lap", "source"), expected)
        self.assertIsNone(modal_app.source_adapter("warmstart-rl", "source"))

    def test_missing_native_selection_and_unsafe_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            modal_app.validate_checkpoint("native-tools-validation", "")
        for value in ("../other", "/runs/other", "a/../../other", "a//b", "a/", "a;echo"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "safe relative"):
                modal_app.validate_checkpoint("native-tools-validation", value)
        with self.assertRaisesRegex(ValueError, "campaign stage"):
            modal_app.validate_checkpoint("warmstart-rl", "checkpoint-0003")

    def test_missing_files_and_symlink_escape_rejected(self):
        with self.assertRaises(FileNotFoundError):
            modal_app.source_adapter("full-lap", "source", "missing")
        outside = self.runs / "outside"
        outside.mkdir()
        experiment = self.runs / "source" / "experiment"
        experiment.mkdir(parents=True)
        (experiment / "escaped").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            modal_app.source_adapter("full-lap", "source", "escaped")


if __name__ == "__main__":
    unittest.main()
