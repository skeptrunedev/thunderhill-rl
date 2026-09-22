"""Metric semantics and an actual offline SDK round trip."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_tracking import collection_metrics, episode_metrics, import_campaign


def episode(reason="crash", success=False, actions=4):
    return dict(reason=reason, success=success, recording_provenance_verified=True,
                final_observation={"state": {"crashed": reason == "crash"}, "sim_time": 12.5},
                actions=actions, offtrack_ticks=0, gates=[1],
                reward_components={"total": -0.8, "legal_progress_m": 100, "progress_fraction": .02})


def collection(rollouts):
    return dict(generation=0, elapsed_seconds=30, adapter_sha256="a" * 64,
                evaluation=episode(), rollouts=rollouts)


class TrackingTests(unittest.TestCase):
    def test_stall_is_distinct_from_crash_and_keeps_earned_reward(self):
        stalled = episode("stalled")
        stalled["reward_components"]["total"] = 0.12
        metrics = collection_metrics(collection([stalled, episode("crash")]))
        self.assertEqual(metrics["rollout/stall_rate"], 0.5)
        self.assertEqual(metrics["rollout/crash_rate"], 0.5)
        self.assertEqual(episode_metrics(stalled)["reward"], 0.12)
        self.assertNotIn("lap_seconds", episode_metrics(stalled))

    def test_failed_duration_is_not_lap_time(self):
        row = episode_metrics(episode())
        self.assertNotIn("lap_seconds", row)
        self.assertEqual(row["episode_seconds"], 12.5)
        valid = episode("lap_completed", True)
        self.assertEqual(episode_metrics(valid)["lap_seconds"], 12.5)
        valid["recording_provenance_verified"] = False
        self.assertNotIn("lap_seconds", episode_metrics(valid))

    def test_invalid_call_denominator_includes_rejected_action(self):
        row = collection_metrics(collection([episode("invalid_model_action"), episode("lap_completed", True)]))
        self.assertEqual(row["rollout/tool_call_count"], 9)
        self.assertEqual(row["rollout/invalid_call_rate"], 1 / 9)
        self.assertEqual(row["rollout/invalid_episode_rate"], .5)
        self.assertEqual(row["rollout/completion_rate"], .5)
        self.assertEqual(row["rollout/mean_lap_seconds"], 12.5)
        self.assertNotIn("eval/lap_seconds", row)

    def test_eval_only_has_no_rollout_rate(self):
        row = collection_metrics(collection([]))
        self.assertEqual(row["rollout/count"], 0)
        self.assertNotIn("rollout/completion_rate", row)

    def test_interrupted_update_preserves_collected_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign.json"
            campaign.write_text(json.dumps({"generations": [], "complete": False}))
            generation = root / "generation-0000"
            generation.mkdir()
            row = collection([episode("invalid_model_action")])
            (generation / "collection.json").write_text(json.dumps(row))
            with patch("experiment_tracking.ExperimentTracker") as factory:
                tracker = factory.return_value.__enter__.return_value
                import_campaign(campaign, root / "tracking")
                tracker.collection.assert_called_once_with(row)
                tracker.update.assert_not_called()

    def test_real_offline_sdk_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign.json"
            manifest = dict(model="test-model", revision="test-revision", secret="must-not-log",
                            generations=[dict(collection=collection([episode()]),
                                              update={"generation": 1, "loss": .2, "trained_tokens": 15},
                                              verification={"adapter_sha256": "b" * 64,
                                                            "reloaded_logits_match": True})],
                            complete=True, final_evaluation=dict(collection([]), generation=1))
            campaign.write_text(json.dumps(manifest))
            run_id = import_campaign(campaign, root / "tracking", mode="offline")
            self.assertTrue(run_id)
            rows = [json.loads(line) for line in (root / "tracking/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["update/trained_tokens"], 15)
            self.assertFalse(any("lap_seconds" in key for row in rows for key in row))
            data = list((root / "tracking/wandb").glob("offline-run-*/*.wandb"))
            self.assertEqual(len(data), 1)
            self.assertGreater(data[0].stat().st_size, 0)
            self.assertNotIn(b"must-not-log", data[0].read_bytes())


if __name__ == "__main__":
    unittest.main()
