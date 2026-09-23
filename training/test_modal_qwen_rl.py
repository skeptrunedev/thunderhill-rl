"""Validate the isolated cloud command, credential handling, and shutdown audit."""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    import modal_qwen_rl as runner
except ModuleNotFoundError as error:
    if error.name != 'modal':
        raise
    runner = None


@unittest.skipUnless(runner, 'Modal SDK is not installed in this interpreter')
class QwenLauncherTests(unittest.TestCase):
    def test_only_fresh_gameplay_training_is_launched(self):
        command = runner.training_command('/runs/test/experiment')
        self.assertIn('--qwen27b', command)
        self.assertNotIn('--adapter', command)
        self.assertNotIn('--resume', command)
        for flag, value in {'--generations': '1', '--initial-generation': '0',
                            '--time-budget-seconds': '30', '--batch-candidates': '3',
                            '--rollouts-per-generation': '4', '--evaluation-rollouts': '2',
                            '--temperature': '0.6', '--wandb-mode': 'online'}.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('qwen_tools.py', runner.TRAINING_FILES)
        for filename in runner.TRAINING_FILES:
            self.assertFalse(any(word in filename for word in ('sft', 'dataset', 'warmstart', 'modal_')))

    def test_run_id_cannot_escape_volume(self):
        for value in ('', '../outside', '/absolute', 'a/b', 'a;echo', 'x' * 81):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_run_id(value)
        runner.validate_run_id('qwen27b-rl-01')

    def test_credentials_forwarded_only_as_secret(self):
        with patch.object(runner.netrc, 'netrc') as netrc, patch.object(runner.modal.Secret, 'from_dict') as secret:
            netrc.return_value.authenticators.return_value = ('user', None, 'test-private-token')
            runner.wandb_secret()
            secret.assert_called_once_with({'WANDB_API_KEY': 'test-private-token'})
            netrc.return_value.authenticators.return_value = None
            with self.assertRaisesRegex(RuntimeError, 'authentication'):
                runner.wandb_secret()

    def run_mock(self, waits, clock=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        process = MagicMock()
        process.stdout = io.StringIO('real child output\n')
        process.wait.side_effect = waits
        process.returncode = 0
        process.poll.return_value = 0
        volume = MagicMock()
        patches = [patch.object(runner, 'RUNS', root), patch.object(runner, 'artifacts', volume),
                   patch.object(runner, 'cache'), patch.object(runner.subprocess, 'Popen', return_value=process),
                   patch.dict(os.environ, {'WANDB_API_KEY': 'test-private-token'})]
        if clock is not None:
            patches.append(patch.object(runner.time, 'monotonic', side_effect=clock))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return root, process, volume

    def test_success_preserves_manifest_log_status_and_periodic_snapshot(self):
        root, process, volume = self.run_mock([subprocess.TimeoutExpired('test', 60), 0])
        status = runner.execute_run('test', {'git_commit': 'abc'})
        self.assertTrue(status['ok'])
        self.assertGreaterEqual(volume.commit.call_count, 3)
        manifest = json.loads((root / 'test/launch.json').read_text())
        self.assertIsNone(manifest['source_checkpoint'])
        self.assertFalse(manifest['supervised_training_performed'])
        self.assertEqual((root / 'test/run.log').read_text(), 'real child output\n')
        self.assertTrue(json.loads((root / 'test/status.json').read_text())['ok'])
        self.assertNotIn('test-private-token', (root / 'test/launch.json').read_text())
        self.assertTrue(runner.subprocess.Popen.call_args.kwargs['start_new_session'])
        with self.assertRaises(FileExistsError):
            runner.execute_run('test', {})

    def test_deadline_interrupts_trainer_before_group_kill(self):
        root, process, _ = self.run_mock([subprocess.TimeoutExpired('test', 120), 0],
                                       clock=[0, 0, runner.CHILD_TIMEOUT + 1, runner.CHILD_TIMEOUT + 2])
        with patch.object(runner.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'Training exited'):
                runner.execute_run('timeout', {})
            process.send_signal.assert_called_once_with(signal.SIGINT)
            kill.assert_called_once_with(process.pid, signal.SIGKILL)
        status = json.loads((root / 'timeout/status.json').read_text())
        self.assertTrue(status['timeout'])
        self.assertFalse(status['ok'])

    def test_missing_secret_stops_before_reserving_run(self):
        root, _, _ = self.run_mock([0])
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'Missing W&B'):
            runner.execute_run('missing-secret', {})
        self.assertFalse((root / 'missing-secret').exists())


if __name__ == '__main__':
    unittest.main()
