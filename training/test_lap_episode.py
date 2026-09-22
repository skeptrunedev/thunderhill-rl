"""Full episode reward boundaries and real simulator collector lifecycle."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lap_episode import LapEpisode, StallConfig, StallMonitor, episode_reward
from lap_policy import RoadTelemetry


class EpisodeRewardTests(unittest.TestCase):
    def reward(self, **kwargs):
        return episode_reward(**dict({"legal_progress_m": 3000, "track_length_m": 4800,
            "sim_seconds": 500, "time_budget_seconds": 900, "success": False}, **kwargs))

    def test_shorter_valid_completion_is_better(self):
        self.assertGreater(self.reward(success=True, sim_seconds=200)["total"],
                           self.reward(success=True, sim_seconds=800)["total"])

    def test_incomplete_never_beats_completed(self):
        self.assertLess(self.reward(legal_progress_m=1e9)["total"],
                        self.reward(success=True, sim_seconds=900)["total"])

    def test_net_progress_and_failure_penalties(self):
        self.assertGreater(self.reward()["total"], self.reward(legal_progress_m=500)["total"])
        self.assertEqual(self.reward(legal_progress_m=-100)["total"], -1)
        self.assertEqual(self.reward(failed=True)["total"], self.reward()["total"] - 0.2)
        self.assertEqual(self.reward(invalid_syntax=True)["total"], self.reward()["total"] - 1)
        self.assertEqual(self.reward(failed=True)["completion_bonus"], 0)

    def test_exploration_beats_inactivity_but_clean_progress_is_better(self):
        idle = self.reward(legal_progress_m=0)["total"]
        crashed = self.reward(legal_progress_m=40, failed=True)["total"]
        clean = self.reward(legal_progress_m=40)["total"]
        self.assertEqual(idle, 0)
        self.assertAlmostEqual(crashed, 0.2)
        self.assertAlmostEqual(clean, 0.4)
        self.assertGreater(crashed, idle)
        self.assertGreater(clean, crashed)
        self.assertAlmostEqual(self.reward(legal_progress_m=57)["total"], 0.57)

    def test_early_crash_cannot_escape_penalty_or_gain_time_bonus(self):
        early = self.reward(legal_progress_m=0, failed=True, sim_seconds=1)
        late = self.reward(legal_progress_m=0, failed=True, sim_seconds=500)
        self.assertEqual(early["total"], late["total"])
        self.assertLess(early["total"], self.reward(legal_progress_m=0)["total"])
        self.assertEqual(early["speed_bonus"], 0)

    def test_signed_progress_does_not_reward_retracing(self):
        self.assertEqual(self.reward(legal_progress_m=sum([40, -40]))["total"], 0)
        self.assertLess(self.reward(legal_progress_m=-40)["total"], 0)
        self.assertAlmostEqual(self.reward(legal_progress_m=40)["progress_fraction"], 40 / 4800)

    def test_reject_invalid_success_or_nonfinite(self):
        for kwargs in ({"success": True, "failed": True}, {"success": True, "invalid_syntax": True},
                       {"success": True, "sim_seconds": 901}, {"legal_progress_m": float("nan")}):
            with self.assertRaises(ValueError):
                self.reward(**kwargs)



class StallMonitorTests(unittest.TestCase):
    def run_monitor(self, distance, last_tick=1200):
        monitor = StallMonitor()
        stopped = None
        for tick in range(0, last_tick + 1, 12):
            if monitor.observe(tick, distance(tick)):
                stopped = tick
                break
        return monitor, stopped

    def test_idle_stops_at_ten_seconds_not_during_grace(self):
        monitor, stopped = self.run_monitor(lambda tick: 0)
        self.assertEqual(stopped, 1200)
        self.assertEqual(monitor.diagnostic["start_tick"], 600)
        self.assertEqual(monitor.diagnostic["window_ticks"], 600)
        self.assertEqual(monitor.diagnostic["legal_progress_m"], 0)

    def test_launch_during_grace_and_slow_legal_progress_survive(self):
        for distance in (lambda tick: max(0, tick - 480) / 300,
                         lambda tick: tick / 600):
            with self.subTest(distance=distance):
                monitor, stopped = self.run_monitor(distance, 3600)
                self.assertIsNone(stopped)
                self.assertGreaterEqual(monitor.diagnostic["legal_progress_m"], 1)

    def test_signed_reverse_and_retracing_cannot_avoid_stop(self):
        for distance in (lambda tick: -tick / 100,
                         lambda tick: (tick % 600) / 100):
            with self.subTest(distance=distance):
                monitor, stopped = self.run_monitor(distance)
                self.assertEqual(stopped, 1200)
                self.assertLessEqual(monitor.diagnostic["legal_progress_m"], 0)

    def test_late_stall_uses_rolling_progress_not_total_distance(self):
        monitor, stopped = self.run_monitor(lambda tick: min(tick, 1800) / 120, 3000)
        self.assertEqual(stopped, 2292)
        self.assertEqual(monitor.diagnostic["window_ticks"], 600)
        self.assertGreater(monitor.diagnostic["end_legal_distance_m"], 1)
        self.assertLess(monitor.diagnostic["legal_progress_m"], 1)

    def test_exact_meter_boundary_tolerates_float_roundoff(self):
        _, stopped = self.run_monitor(lambda tick: tick / 600 - (1e-12 if tick >= 1200 else 0))
        self.assertIsNone(stopped)
        _, stopped = self.run_monitor(lambda tick: tick * 0.999 / 600)
        self.assertEqual(stopped, 1200)

    def test_simulator_endings_take_precedence_at_stall_boundary(self):
        endings = (
            (True, False, "crash", "", False, "crash"),
            (True, False, "lap_completed", "", True, "lap_completed"),
            (False, True, "", "episode_tick_limit", True, "episode_tick_limit"),
            (False, False, "", "", False, "track_limits"),
        )
        for terminated, truncated, termination, truncation, valid, expected in endings:
            with self.subTest(reason=expected):
                road = SimpleNamespace(prompt_features=lambda features: "observed road")
                episode = LapEpisode(godot="unused", output="unused", road=road,
                    adapter_sha256="a" * 64, model="test", revision="test", generation=1, rollout=1)
                episode.episode_id = "test"
                episode.decisions = io.StringIO()
                episode.view = {"road": {}, "observation_token": "receipt"}
                episode.env = SimpleNamespace(_observation={"tick": 1188})
                episode.stall_monitor.observe(600, 0)
                episode.stall_monitor.observe(1188, 0)

                def control_bike(*args, **kwargs):
                    episode.env._observation = {
                        "tick": 1200, "terminated": terminated, "truncated": truncated,
                        "termination_reason": termination, "truncation_reason": truncation,
                        "track": {"lap_valid": valid, "legal_distance": 0}}
                    return json.dumps({"road": {}, "observation_token": "next"})

                episode.env.control_bike = control_bike
                episode.apply("control_bike 0 0 100 0", [2], [1])
                self.assertEqual(episode.reason, expected)
                self.assertNotIn("stall_diagnostic", episode.records[-1])

    def test_observations_and_configuration_fail_on_invalid_inputs(self):
        for kwargs in ({"grace_ticks": -1}, {"window_ticks": 0}, {"window_ticks": 1.5},
                       {"minimum_progress_m": float("nan")}):
            with self.assertRaises(ValueError):
                StallConfig(**kwargs)
        monitor = StallMonitor()
        monitor.observe(0, 0)
        for tick, distance in ((0, 0), (1.5, 0), (12, float("nan"))):
            with self.assertRaises(ValueError):
                monitor.observe(tick, distance)

@unittest.skipUnless(os.environ.get("THUNDERHILL_GODOT"), "Set THUNDERHILL_GODOT for native collector verification")
class NativeEpisodeTests(unittest.TestCase):
    def episode(self, root, name):
        return LapEpisode(godot=os.environ["THUNDERHILL_GODOT"], output=root / name,
                          road=RoadTelemetry(), adapter_sha256="a" * 64,
                          model="collector-test", revision="test", generation=1, rollout=1,
                          time_budget_seconds=0.2)

    def test_stalled_idle_is_archived_and_training_eligible(self):
        root = Path(tempfile.mkdtemp(prefix="stall-collector-native-",
            dir=Path(__file__).resolve().parents[1] / "artifacts"))
        with LapEpisode(godot=os.environ["THUNDERHILL_GODOT"], output=root / "idle",
                        road=RoadTelemetry(), adapter_sha256="c" * 64,
                        model="stall-lifecycle-test", revision="test", generation=1, rollout=1,
                        time_budget_seconds=30) as episode:
            while not episode.done:
                episode.apply("control_bike 0 0 100 0", [2], [1])
            self.assertEqual(episode.reason, "stalled")
            self.assertEqual(episode.observation["tick"], 1200)
            self.assertFalse(episode.observation["state"]["crashed"])
        summary = episode.summary
        self.assertEqual(summary["recorded_transitions"], 1200)
        self.assertTrue(summary["training_eligible"])
        self.assertTrue(summary["recording_provenance_verified"])
        self.assertFalse(summary["success"])
        self.assertEqual(summary["reward_components"]["failure_penalty"], 0)
        self.assertAlmostEqual(summary["reward_components"]["total"],
                               summary["reward_components"]["legal_progress_m"] / 100)
        self.assertEqual(summary["stall_diagnostic"]["window_ticks"], 600)
        self.assertEqual(episode.records[-1]["stall_diagnostic"], summary["stall_diagnostic"])
        video = json.loads((episode.output / summary["video_job"]).read_text())
        self.assertEqual(video["metadata"]["stop_reason"], "stalled")
        self.assertEqual(video["metadata"]["stall_diagnostic"], summary["stall_diagnostic"])
        self.assertTrue((episode.output / video["source"]).is_file())

    def test_time_budget_takes_precedence_over_stall(self):
        root = Path(tempfile.mkdtemp(prefix="stall-budget-native-",
            dir=Path(__file__).resolve().parents[1] / "artifacts"))
        with LapEpisode(godot=os.environ["THUNDERHILL_GODOT"], output=root / "budget",
                        road=RoadTelemetry(), adapter_sha256="d" * 64,
                        model="stall-budget-test", revision="test", generation=1, rollout=1,
                        time_budget_seconds=10) as episode:
            while not episode.done:
                episode.apply("control_bike 0 0 100 0", [2], [1])
            self.assertEqual(episode.reason, "episode_tick_limit")
            self.assertEqual(episode.observation["tick"], 1200)
        self.assertTrue(episode.summary["training_eligible"])

    def test_native_tool_feedback_uses_latest_actual_result(self):
        from transformers import AutoTokenizer
        from model_runtime import GEMMA4_NATIVE_SPEC, PolicyRoadTelemetry
        tokenizer = AutoTokenizer.from_pretrained(GEMMA4_NATIVE_SPEC.model,
            revision=GEMMA4_NATIVE_SPEC.revision, local_files_only=True)
        road = PolicyRoadTelemetry(GEMMA4_NATIVE_SPEC, tokenizer)
        root = Path(tempfile.mkdtemp(prefix="native-tool-collector-",
            dir=Path(__file__).resolve().parents[1] / "artifacts"))
        completion = road.native_tools.completion_from_controls(
            {"steer": 0, "throttle": 0.3, "front_brake": 0, "rear_brake": 0})
        ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        with LapEpisode(godot=os.environ["THUNDERHILL_GODOT"], output=root / "episode",
                        road=road, adapter_sha256="b" * 64, model="native-collector-test",
                        revision="test", generation=1, rollout=1, time_budget_seconds=0.3) as episode:
            self.assertNotIn("<|tool_response>", episode.prompt())
            for index in range(3):
                prompt = episode.prompt()
                if index:
                    self.assertEqual(prompt.count("<|tool_call>"), 1)
                    self.assertEqual(prompt.count("<|tool_response>"), 1)
                    self.assertIn(f"tick:{index * 12}", prompt)
                    self.assertNotIn("observation_token", prompt)
                    self.assertTrue(prompt.endswith("<tool_response|>"))
                episode.apply(completion, ids, tokenizer(prompt, add_special_tokens=False)["input_ids"])
            self.assertTrue(episode.done)
        self.assertEqual(episode.summary["recorded_transitions"], 36)
        self.assertTrue(episode.summary["training_eligible"])
        self.assertEqual(len(list((root / "episode" / "video_jobs").glob("*.json"))), 1)

    def test_complete_budget_and_malformed_and_exception_are_archived(self):
        root = Path(tempfile.mkdtemp(prefix="lap-collector-native-", dir=Path(__file__).resolve().parents[1] / "artifacts"))
        with self.episode(root, "budget") as episode:
            for _ in range(2):
                episode.apply("control_bike 0 30 0 0", [2], [1], behavior_logprobs=[-0.1])
            self.assertTrue(episode.done)
        self.assertEqual(episode.summary["recorded_transitions"], 24)
        self.assertTrue(episode.summary["training_eligible"])
        self.assertFalse(episode.summary["success"])
        self.assertEqual(episode.summary["reason"], "episode_tick_limit")
        with self.episode(root, "syntax") as malformed:
            malformed.apply("bad call", [2], [1])
        self.assertEqual(malformed.summary["reward_components"]["syntax_penalty"], -1)
        self.assertEqual(malformed.summary["recorded_transitions"], 0)
        with self.assertRaisesRegex(RuntimeError, "inference broke"):
            with self.episode(root, "failure") as failed:
                failed.apply("control_bike 0 30 0 0", [2], [1])
                raise RuntimeError("inference broke")
        self.assertFalse(failed.summary["training_eligible"])
        self.assertNotIn("reward_components", failed.summary)
        for name in ("budget", "syntax", "failure"):
            jobs = list((root / name / "video_jobs").glob("*.json"))
            self.assertEqual(len(jobs), 1)
            job = json.loads(jobs[0].read_text())
            self.assertTrue((root / name / job["source"]).is_file())
            self.assertEqual(job["policy_display"]["rollout_number"], 1)


if __name__ == "__main__":
    unittest.main()
