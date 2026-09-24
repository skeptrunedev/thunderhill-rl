"""Real simulator collector lifecycle checks."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from lap_episode import LapEpisode
from lap_policy import RoadTelemetry


@unittest.skipUnless(os.environ.get("THUNDERHILL_GODOT"), "Set THUNDERHILL_GODOT for native collector verification")
class EpisodeLifecycleTests(unittest.TestCase):
    def test_moving_reset_validates_speed_and_preserves_default(self):
        root = Path(tempfile.mkdtemp(prefix='moving-reset-native-',
                                    dir=Path(__file__).resolve().parents[1] / 'artifacts'))
        with LapEpisode(godot=os.environ['THUNDERHILL_GODOT'], output=root / 'episode',
                        road=RoadTelemetry(), adapter_sha256='e' * 64,
                        model='moving reset fixture', revision='test', generation=0,
                        rollout=1, initial_speed_m_s=5, time_budget_seconds=.1) as episode:
            self.assertEqual(episode.observation['state']['speed'], 5)
            self.assertEqual(episode.observation['state']['longitudinal_velocity'], 5)
            for speed in (-1, 11, True, '5'):
                response = episode.client.request(dict(op='reset', initial_speed_m_s=speed))
                self.assertIn('error', response)
            episode.apply('control_bike 0 0 0 0', [], [])
        self.assertTrue(episode.summary['recording_provenance_verified'])
        self.assertEqual(episode.summary['initial_speed_m_s'], 5)

    def test_continuous_controls_survive_protocol_and_recording(self):
        from driving_trajectory import encode_controller_controls, decode_controller_controls
        root = Path(tempfile.mkdtemp(prefix='continuous-collector-native-',
                                    dir=Path(__file__).resolve().parents[1] / 'artifacts'))
        controls = dict(steer=-1.234567891234567e-8, throttle=0.00089031472971384,
                        front_brake=0.000136217883843274, rear_brake=0.0, shift=0)
        with LapEpisode(godot=os.environ['THUNDERHILL_GODOT'], output=root / 'episode',
                        road=RoadTelemetry(), adapter_sha256='e' * 64,
                        model='continuous control plumbing fixture', revision='test',
                        generation=0, rollout=1, time_budget_seconds=.1,
                        action_parser=decode_controller_controls) as episode:
            row = episode.apply(encode_controller_controls(controls), [], [])
            self.assertEqual(row['controls'], controls)
        self.assertTrue(episode.summary['recording_provenance_verified'])
        self.assertEqual(episode.summary['recorded_transitions'], 12)
        from lap_audit import recorded_controls_match
        transitions = [json.loads(line) for path in episode.summary['recordings']
                       for line in Path(path).read_text().splitlines()
                       if json.loads(line).get('type') == 'transition']
        self.assertEqual(len(transitions), 12)
        for transition in transitions:
            self.assertTrue(recorded_controls_match(transition['requested_controls'], controls))
            for name in ('steer', 'throttle', 'front_brake'):
                self.assertNotEqual(transition['requested_controls'][name], 0.0)

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
