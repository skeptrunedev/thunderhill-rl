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

import modal_alpamayo_rl as launcher

from modal_alpamayo_rl import (
    MODEL, PROCESSOR, download_model_files, run_logged_process,
)


class SourcePackagingTests(unittest.TestCase):
    def test_both_remote_functions_have_uploadable_entrypoint_modules(self):
        # Inspect actual SDK dependency mounts, not source text or mocked calls.
        # include_source=False functions import their module by name remotely.
        for image, function in ((launcher.download_image, launcher.prepare_weights),
                                (launcher.image, launcher.run_generation)):
            info = function._get_info()
            with self.subTest(function=info.function_name):
                self.assertFalse(info.is_serialized())
                expected_remote = (Path(launcher.REMOTE) / 'training' /
                                   (info.module_name.replace('.', '/') + '.py'))
                uploads = []
                for dependency in image.deps():
                    for entry in getattr(dependency, 'entries', ()):
                        uploads.extend(entry.get_files_to_upload())
                matching = [(local, remote) for local, remote in uploads
                            if str(remote) == str(expected_remote)]
                self.assertEqual(len(matching), 1)
                local, _ = matching[0]
                self.assertEqual(local.resolve(), Path(launcher.__file__).resolve())
                self.assertEqual(local.read_bytes(), Path(launcher.__file__).read_bytes())


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


class DiagnosticCompletionTests(unittest.TestCase):
    def test_baseline_stop_is_valid_but_not_completed_training(self):
        value = dict(diagnostic_complete=True, complete=False, generations=[],
                     evaluations=[{}] * 4, baseline_gate={'passed': False},
                     stop_reason='baseline_gate_failed')
        launcher.validate_diagnostic_result(value)
        for changed in (dict(complete=True), dict(generations=[{}]), dict(evaluations=[])):
            with self.assertRaises(RuntimeError):
                launcher.validate_diagnostic_result({**value, **changed})

    def test_success_requires_all_generations_and_evaluations(self):
        value = dict(diagnostic_complete=True, complete=True, generations=[{}] * 3,
                     evaluations=[{}] * 16, baseline_gate={'passed': False}, training_readiness={'passed': True},
                     stop_reason='generations_completed')
        launcher.validate_diagnostic_result(value)
        for changed in (dict(generations=[{}]), dict(evaluations=[{}] * 4),
                        dict(training_readiness={'passed': False}), dict(diagnostic_complete=False)):
            with self.assertRaises(RuntimeError):
                launcher.validate_diagnostic_result({**value, **changed})


class ResumeInvocationTests(unittest.TestCase):
    def prepare(self, root, **changes):
        args = dict(run_id='test', launch_id='first', source={'git_commit': 'new'},
                    diagnostic=True, generations=5, resume=False, now=100)
        args.update(changes)
        return launcher.prepare_invocation(root, **args)

    def test_preemption_keeps_deadline_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'test'
            launch, first = self.prepare(root)
            (first / 'run.log').write_text('original logs')
            again, second = self.prepare(root, now=200)
            self.assertEqual(launch, again)
            self.assertEqual(again['deadline_at'], 5500)
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'run.log').read_text(), 'original logs')

    def test_changed_source_requires_explicit_resume_and_retains_lineage(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'test'
            original, _ = self.prepare(root)
            with self.assertRaises(FileExistsError):
                self.prepare(root, launch_id='second')
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                self.prepare(root, source={'git_commit': 'other'})
            resumed, invocation = self.prepare(root, launch_id='second', resume=True,
                source={'git_commit': 'other'}, now=300)
            self.assertEqual(resumed['resumed_from_source'], original['source'])
            self.assertEqual(json.loads((root / 'launch.json').read_text()), original)
            self.assertEqual(resumed['source']['git_commit'], 'other')
            self.assertEqual(resumed['deadline_at'], 5700)

    def test_resume_cannot_create_missing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self.prepare(Path(directory) / 'test', resume=True)

    def test_previous_terminal_status_is_preserved_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'test'
            self.prepare(root)
            (root / 'status.json').write_text('{"ok": false}')
            _, invocation = self.prepare(root, launch_id='second', resume=True)
            self.assertFalse((root / 'status.json').exists())
            self.assertEqual((invocation / 'previous-status.json').read_text(), '{"ok": false}')

    def test_five_generation_completion_cannot_accept_three(self):
        value = dict(diagnostic_complete=True, complete=True, generations=[{}] * 5,
                     evaluations=[{}] * 24, training_readiness={'passed': True},
                     stop_reason='generations_completed')
        launcher.validate_diagnostic_result(value, 5)
        with self.assertRaises(RuntimeError):
            launcher.validate_diagnostic_result({**value, 'generations': [{}] * 3}, 5)


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
