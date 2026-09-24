"""CPU integration checks against the pinned NVIDIA config implementation."""

import json
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import tomllib
import unittest

from training.alpagym_metrics import REWARD_SCALES, REWARD_VERSION
from training.nvidia_alpagym import (
    DEFAULT_SOURCE,
    cosmos_command,
    prepare,
    owned_subreaper,
    stop_process_tree,
    wait_process,
    write_status,
)


@unittest.skipUnless(
    DEFAULT_SOURCE.is_dir(), "Pinned AlpaGym source checkout is required"
)
class ConfigurationTests(unittest.TestCase):
    def test_actual_upstream_round_trip_and_training_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = prepare(
                DEFAULT_SOURCE,
                Path(directory),
                Path(directory) / "model",
                godot="godot",
                max_steps=3,
                rollouts=4,
                episode_seconds=12,
                concurrency=2,
            )
            from alpagym_host.config import load_run_config

            config = load_run_config(run_dir / "resolved_config.yaml")
            cosmos = tomllib.loads((run_dir / "cosmos_config.toml").read_text())
            self.assertEqual(
                cosmos["train"]["train_policy"]["trainer_type"], "alpagym_grpo"
            )
            self.assertEqual(cosmos["train"]["train_policy"]["type"], "grpo")
            self.assertEqual(cosmos["train"]["max_num_steps"], 3)
            self.assertEqual(cosmos["train"]["epoch"], 3)
            self.assertEqual(cosmos["rollout"]["n_generation"], 4)
            self.assertEqual(cosmos["train"]["train_batch_per_replica"], 4)
            self.assertEqual(
                cosmos["rollout"]["backend"], "thunderhill_alpagym_rollout"
            )
            self.assertFalse(cosmos["rollout"]["prefetch_rollout"])
            self.assertEqual(
                config.policy.model.use_cameras, ["camera_front_wide_120fov"]
            )
            self.assertEqual(config.expected_valid_steps, 60)
            self.assertEqual(config.alpasim.wizard_args.control_timestep_us, 200000)
            self.assertEqual(config.alpasim.wizard_args.force_gt_duration_us, 0)
            self.assertEqual(config.transport.kind, "nccl")
            self.assertTrue(all(term.kind == "metric" for term in config.reward.terms))
            self.assertEqual(
                {term.metric_name for term in config.reward.terms},
                {"progress", "collision_any", "offroad", "fall_without_collision"},
            )
            self.assertEqual({term.metric_name: term.scale for term in config.reward.terms}, REWARD_SCALES)
            self.assertEqual(config.cosmos.mode, "disaggregated")
            command = cosmos_command(DEFAULT_SOURCE, config)
            self.assertEqual(command[-1], "training.alpagym_worker")
            self.assertIn("cosmos_rl.launcher.launch_all", command)
            manifest = json.loads((run_dir / "launch_manifest.json").read_text())
            self.assertFalse(manifest["gpu_training_verified"])
            self.assertEqual(manifest["reward_version"], REWARD_VERSION)
            self.assertFalse((run_dir / "topology").exists())
            game = json.loads((run_dir / "game_config.json").read_text())
            self.assertTrue((Path(game["project_path"]) / "project.godot").is_file())
            self.assertEqual(game["concurrency"], 2)

    def test_invalid_budget_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for options in (
                {"max_steps": 0},
                {"rollouts": 1},
                {"concurrency": 0},
                {"episode_seconds": float("nan")},
                {"episode_seconds": 0},
                {"episode_seconds": 0.15},
                {"max_wall_seconds": 0},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    prepare(
                        DEFAULT_SOURCE, root, root / "model", godot="godot", **options
                    )
            self.assertEqual(list(root.iterdir()), [])


class ProcessLifecycleTests(unittest.TestCase):
    def test_timeout_reaps_child_that_creates_own_session(self):
        import psutil

        with tempfile.TemporaryDirectory() as directory:
            child_file = Path(directory) / "child.pid"
            script = (
                "import subprocess,sys,time,pathlib; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
            )
            parent = subprocess.Popen(
                [sys.executable, "-c", script, str(child_file)], start_new_session=True
            )
            try:
                end = time.monotonic() + 5
                while not child_file.exists() and time.monotonic() < end:
                    time.sleep(0.02)
                child_pid = int(child_file.read_text())
                with self.assertRaises(TimeoutError):
                    wait_process(parent, time.monotonic())
            finally:
                stop_process_tree(parent)
            self.assertIsNotNone(parent.poll())
            self.assertTrue(
                not psutil.pid_exists(child_pid)
                or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
            )

    def test_subreaper_collects_double_fork_daemon_and_preserves_existing_child(self):
        import ctypes
        import os
        import psutil

        existing = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        libc = ctypes.CDLL(None)
        before = ctypes.c_int()
        libc.prctl(37, ctypes.byref(before), 0, 0, 0)
        try:
            with tempfile.TemporaryDirectory() as directory:
                pidfile = Path(directory) / "daemon.pid"
                script = """import os, pathlib, sys, time
if os.fork():
    os._exit(0)
os.setsid()
if os.fork():
    os._exit(0)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
"""
                with owned_subreaper():
                    bootstrap = subprocess.Popen(
                        [sys.executable, "-c", script, str(pidfile)]
                    )
                    bootstrap.wait(timeout=5)
                    end = time.monotonic() + 5
                    while not pidfile.exists() and time.monotonic() < end:
                        time.sleep(0.02)
                    daemon_pid = int(pidfile.read_text())
                    daemon = psutil.Process(daemon_pid)
                    self.assertEqual(daemon.ppid(), os.getpid())
                    self.assertNotEqual(os.getpgid(daemon_pid), os.getpgrp())
                self.assertFalse(psutil.pid_exists(daemon_pid))
                self.assertIsNone(existing.poll())
                after = ctypes.c_int()
                libc.prctl(37, ctypes.byref(after), 0, 0, 0)
                self.assertEqual(before.value, after.value)
        finally:
            existing.terminate()
            existing.wait(timeout=5)

    def test_success_status_does_not_claim_optimizer_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_status(root, "completed", launcher_exit_code=0)
            status = json.loads((root / "run_status.json").read_text())
            self.assertEqual(status["state"], "completed")
            self.assertFalse(status["optimizer_updates_verified"])


if __name__ == "__main__":
    unittest.main()
