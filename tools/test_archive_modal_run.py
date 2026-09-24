"""Archive publication uses the real CLI contract, with no cloud side effects."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import threading
import time
from types import SimpleNamespace

from archive_modal_run import archive_run, download_volume_run


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
        self.training_started = True

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
            if self.training_started:
                (destination / 'trial/experiment').mkdir()
            (destination / "trial/status.json").write_text(json.dumps(self.status))
            (destination / "trial/launch.json").write_text(json.dumps({"run_id": "wrong" if self.wrong_identity else "trial"}))
        else:
            self.assertIn("render_video_queue.py", command[1])
            self.assertIn("--recover-interrupted", command)
            self.assertTrue((self.out / "trial/status.json").is_file())
        return subprocess.CompletedProcess(command, 0)

    def download(self, run_id, stage):
        self.execute(["modal", "volume", "get", "unused", run_id, str(stage)])

    def run_archive(self, **kwargs):
        return archive_run(run_id="trial", staging_root=self.root / "staging", output=self.out,
                           godot="godot", ffmpeg="ffmpeg", execute=self.execute, downloader=self.download, **kwargs)

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

    def test_failed_preflight_archives_without_requesting_nonexistent_videos(self):
        self.training_started = False
        self.assertEqual(self.run_archive(), self.out)
        self.assertTrue(all(command[0] == 'modal' for command in self.calls))

    def test_success_cannot_skip_renderer_even_without_experiment(self):
        self.training_started = False
        self.status['ok'] = True
        self.run_archive()
        self.assertTrue(any(command[0] != 'modal' for command in self.calls))

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
                        godot="godot", ffmpeg="ffmpeg", execute=fail_render, downloader=self.download)
        self.assertTrue((self.out / "trial/status.json").is_file())

    def test_cli_failure_is_not_pending(self):
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])
        with self.assertRaises(subprocess.CalledProcessError):
            archive_run(run_id="trial", staging_root=self.root / "staging", output=self.out,
                        godot="godot", ffmpeg="ffmpeg", execute=fail, watch=True)


class VolumeDownloadTests(unittest.TestCase):
    def test_inventory_finishes_before_bounded_reads(self):
        listed = False
        active = maximum = 0
        lock = threading.Lock()
        def inventory(*args, **kwargs):
            nonlocal listed
            for index in range(12):
                yield SimpleNamespace(path=f'trial/{index}', type=1, size=3)
            listed = True
        def read(path, output):
            nonlocal active, maximum
            self.assertTrue(listed)
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.01)
            output.write(b'abc')
            with lock:
                active -= 1
            return 3
        volume = SimpleNamespace(iterdir=inventory, read_file_into_fileobj=read)
        with tempfile.TemporaryDirectory() as root:
            download_volume_run(volume, 'trial', root, workers=4)
            self.assertEqual(len(list((Path(root) / 'trial').iterdir())), 12)
            self.assertLessEqual(maximum, 4)
            self.assertGreater(maximum, 1)

    def test_partial_failure_is_never_published(self):
        volume = SimpleNamespace(
            iterdir=lambda *args, **kwargs: [SimpleNamespace(path='trial/status.json', type=1, size=10)],
            read_file_into_fileobj=lambda path, output: output.write(b'short'))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                download_volume_run(volume, 'trial', root)
            self.assertFalse((Path(root) / 'trial/status.json').exists())
            self.assertEqual(len(list((Path(root) / 'trial').glob('*.partial'))), 1)

    def test_download_exception_never_publishes_archive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            def fail(run_id, stage):
                (stage / 'partial-evidence').write_text('preserve')
                raise ConnectionError('network lost')
            from unittest.mock import patch
            with patch('archive_modal_run.finished_status', return_value={'run_id': 'trial', 'ok': True}):
                with self.assertRaises(ConnectionError):
                    archive_run(run_id='trial', staging_root=root / 'staging', output=root / 'out/archive',
                                godot='godot', ffmpeg='ffmpeg', downloader=fail)
            self.assertFalse((root / 'out/archive').exists())
            self.assertEqual(len(list((root / 'staging').glob('trial-*/partial-evidence'))), 1)

    def test_unsafe_inventory_is_rejected_before_any_reads(self):
        from unittest.mock import Mock
        for path, kind in [('trial/../../escape', 1), ('other/file', 1), ('trial/link', 3)]:
            read = Mock()
            volume = SimpleNamespace(iterdir=lambda *args, **kwargs: [SimpleNamespace(path=path, type=kind, size=1)],
                                     read_file_into_fileobj=read)
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    download_volume_run(volume, 'trial', root)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
