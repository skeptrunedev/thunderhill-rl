"""Full episode reward boundaries and real simulator collector lifecycle."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from lap_episode import LapEpisode, episode_reward
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
        self.assertEqual(self.reward(legal_progress_m=-100)["total"], 0)
        self.assertEqual(self.reward(failed=True)["total"], self.reward()["total"] - 1)
        self.assertEqual(self.reward(invalid_syntax=True)["total"], self.reward()["total"] - 1)
        self.assertEqual(self.reward(failed=True)["completion_bonus"], 0)

    def test_reject_invalid_success_or_nonfinite(self):
        for kwargs in ({"success": True, "failed": True}, {"success": True, "invalid_syntax": True},
                       {"success": True, "sim_seconds": 901}, {"legal_progress_m": float("nan")}):
            with self.assertRaises(ValueError):
                self.reward(**kwargs)


@unittest.skipUnless(os.environ.get("THUNDERHILL_GODOT"), "Set THUNDERHILL_GODOT for native collector verification")
class NativeEpisodeTests(unittest.TestCase):
    def episode(self, root, name):
        return LapEpisode(godot=os.environ["THUNDERHILL_GODOT"], output=root / name,
                          road=RoadTelemetry(), adapter_sha256="a" * 64,
                          model="collector-test", revision="test", generation=1, rollout=1,
                          time_budget_seconds=0.2)

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
