"""Archive publication uses the real CLI contract, with no cloud side effects."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from archive_modal_run import archive_run


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "artifacts" / "modal-trial-archive"
        self.status = {"run_id": "trial", "ok": False, "error": "Training stopped"}
        self.calls = []
        self.pending = False
        self.wrong_identity = False

    def execute(self, command, **kwargs):
        self.calls.append(command)
        if command[:3] == ["modal", "volume", "ls"]:
            if self.pending:
                self.pending = False
                data = []
            elif command[4] == "/":
                data = [{"filename": "trial", "type": "dir"}]
            else:
                data = [{"filename": "trial/status.json", "type": "file"}]
            return subprocess.CompletedProcess(command, 0, json.dumps(data))
        if command[4] == "trial/status.json":
            destination = Path(command[-1])
            self.assertTrue(destination.parent.is_dir())
            self.assertFalse(destination.exists())
            destination.write_text(json.dumps(self.status))
            return subprocess.CompletedProcess(command, 0, "✓ Finished downloading files to local!\n")
        if command[:3] == ["modal", "volume", "get"]:
            destination = Path(command[-1])
            self.assertTrue(destination.is_dir())
            self.assertFalse(self.out.exists())
            (destination / "trial").mkdir()
            (destination / "trial/status.json").write_text(json.dumps(self.status))
            (destination / "trial/launch.json").write_text(json.dumps({"run_id": "wrong" if self.wrong_identity else "trial"}))
        else:
            self.assertIn("render_video_queue.py", command[1])
            self.assertTrue((self.out / "trial/status.json").is_file())
        return subprocess.CompletedProcess(command, 0)

    def run_archive(self, **kwargs):
        return archive_run(run_id="trial", staging_root=self.root / "staging", output=self.out,
                           godot="godot", ffmpeg="ffmpeg", execute=self.execute, **kwargs)

    def test_failed_training_archives_and_renders_without_overwrite(self):
        self.assertEqual(self.run_archive(), self.out)
        self.assertFalse(json.loads((self.out / "trial/status.json").read_text())["ok"])
        with self.assertRaises(FileExistsError):
            self.run_archive()

    def test_waits_with_bounded_poll(self):
        self.pending = True
        delays = []
        self.run_archive(watch=True, sleep=delays.append)
        self.assertEqual(delays, [30])

    def test_wrong_identity_never_published(self):
        self.wrong_identity = True
        with self.assertRaises(ValueError):
            self.run_archive()
        self.assertFalse(self.out.exists())
        self.assertEqual(len(list((self.root / "staging").glob("trial-*"))), 1)

    def test_staging_inside_watched_output_rejected(self):
        with self.assertRaises(ValueError):
            archive_run(run_id="trial", staging_root=self.out.parent / "stage", output=self.out,
                        godot="godot", ffmpeg="ffmpeg", execute=self.execute)

    def test_renderer_failure_preserves_published_archive(self):
        def fail_render(command, **kwargs):
            if command[0] != "modal":
                raise subprocess.CalledProcessError(1, command)
            return self.execute(command, **kwargs)
        with self.assertRaises(subprocess.CalledProcessError):
            archive_run(run_id="trial", staging_root=self.root / "staging", output=self.out,
                        godot="godot", ffmpeg="ffmpeg", execute=fail_render)
        self.assertTrue((self.out / "trial/status.json").is_file())

    def test_cli_failure_is_not_pending(self):
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])
        with self.assertRaises(subprocess.CalledProcessError):
            archive_run(run_id="trial", staging_root=self.root / "staging", output=self.out,
                        godot="godot", ffmpeg="ffmpeg", execute=fail, watch=True)


if __name__ == "__main__":
    unittest.main()
