"""Local subprocess and download ordering tests. No Modal remote calls."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

try:
    import modal  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    raise unittest.SkipTest("Launcher tests require the separate Modal CLI environment") from error

from modal_alpamayo_rl import (
    MODEL, PROCESSOR, download_model_files, run_logged_process,
)


class DownloadOrderTests(unittest.TestCase):
    def test_both_config_gates_precede_all_snapshots(self):
        calls = []
        def config(model, *args, **kwargs):
            calls.append(('config', model))
        def snapshot(model, **kwargs):
            calls.append(('snapshot', model))
            return '/cache/' + model
        hub = types.SimpleNamespace(hf_hub_download=config, snapshot_download=snapshot)
        with patch.dict(sys.modules, {'huggingface_hub': hub}):
            result = download_model_files('/cache')
        self.assertEqual(calls, [('config', MODEL), ('config', PROCESSOR),
                                ('snapshot', PROCESSOR), ('snapshot', MODEL)])
        self.assertEqual(result['checkpoint'], '/cache/' + MODEL)

    def test_missing_processor_access_prevents_weight_download(self):
        config = Mock(side_effect=['/model/config.json', PermissionError('gated processor')])
        snapshot = Mock()
        hub = types.SimpleNamespace(hf_hub_download=config, snapshot_download=snapshot)
        with patch.dict(sys.modules, {'huggingface_hub': hub}):
            with self.assertRaisesRegex(PermissionError, 'gated processor'):
                download_model_files('/cache')
        snapshot.assert_not_called()


@unittest.skipUnless(sys.platform == 'linux', 'Process group tests require Linux')
class ProcessCleanupTests(unittest.TestCase):
    def run_child(self, code, root, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_logged_process(
                [sys.executable, '-u', '-c', code], cwd=root,
                log_path=Path(root) / 'run.log', heartbeat=lambda: None,
                timeout_seconds=kwargs.pop('timeout_seconds', 3),
                grace_seconds=kwargs.pop('grace_seconds', .2), **kwargs,
            )

    def assert_terminated(self, pid):
        # An orphan killed after wrapper exit can briefly remain a zombie until
        # the container's init reaps it. It must not remain a runnable process.
        path = Path(f'/proc/{pid}/stat')
        deadline = time.monotonic() + 2
        while path.exists() and time.monotonic() < deadline:
            try:
                state = path.read_text().split(') ', 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                return
            if state == 'Z':
                return
            time.sleep(.01)
        self.assertFalse(path.exists(), f'Process {pid} survived supervisor cleanup')

    def test_success_keeps_final_logs(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.run_child("print('first'); print('last')", root)
            self.assertEqual(result, 0)
            self.assertEqual((Path(root) / 'run.log').read_text(), 'first\nlast\n')

    def test_timeout_drains_interrupt_cleanup_before_closing_log(self):
        code = """
        import signal,time
        def stop(*args):
            print('cleanup receipt', flush=True)
            raise SystemExit(0)
        signal.signal(signal.SIGINT, stop)
        print('started', flush=True)
        time.sleep(100)
        """
        import textwrap
        code = textwrap.dedent(code)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(TimeoutError):
                self.run_child(code, root, timeout_seconds=.3)
            self.assertIn('cleanup receipt', (Path(root) / 'run.log').read_text())

    def test_timeout_kills_process_ignoring_interrupt(self):
        code = "import os,signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(100)"
        with tempfile.TemporaryDirectory() as root:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                self.run_child(code, root, timeout_seconds=.3)
            self.assertLess(time.monotonic() - started, 3)
            self.assert_terminated(int((Path(root) / 'run.log').read_text().strip()))

    def test_wrapper_exit_kills_descendant_holding_stdout(self):
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(100)']); print(p.pid,flush=True)"
        with tempfile.TemporaryDirectory() as root:
            self.run_child(code, root)
            pid = int((Path(root) / 'run.log').read_text().strip())
            self.assert_terminated(pid)


if __name__ == '__main__':
    unittest.main()
