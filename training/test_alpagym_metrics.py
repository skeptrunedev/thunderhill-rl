"""Regression checks using NVIDIA's actual reward dispatcher."""

import unittest
from types import SimpleNamespace

from alpagym_host.config import RewardConfig, RewardTermConfig
from alpagym_runtime.rewards.compute import compute_reward
from alpagym_runtime.types import EpisodeOutput, EpisodeMetrics as NativeMetrics
from training.alpagym_metrics import (
    EpisodeMetrics,
    REWARD_SCALES,
    validate_reward_terms,
)


def transition(distance, *, on_track=True, crashed=False, contact=None):
    return dict(
        track=dict(legal_distance=distance, on_track=on_track, lap_valid=False),
        state=dict(crashed=crashed, collision_contact=contact or {}),
    )


def reward(metrics):
    episode = EpisodeOutput(
        scene_id="test",
        session_uuid="test",
        num_steps=1,
        policy_outputs=(),
        metrics=NativeMetrics(metrics.values()),
    )
    config = RewardConfig(
        terms=[
            RewardTermConfig(kind="metric", metric_name=name, scale=scale)
            for name, scale in REWARD_SCALES.items()
        ]
    )
    return compute_reward(episode, None, config).total


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.metrics = EpisodeMetrics(track_length_m=5000, start_legal_distance_m=20)

    def test_normalization_and_actual_nvidia_dispatch(self):
        self.metrics.observe(transition(70))
        self.assertAlmostEqual(reward(self.metrics), 0.01)
        self.metrics.observe(transition(70, crashed=True, contact={"object": "wall"}))
        self.assertAlmostEqual(reward(self.metrics), -9.99)
        self.assertEqual(self.metrics.values()["offroad"], 0)
        self.assertEqual(self.metrics.values()["fall_without_collision"], 0)

    def test_fall_is_distinct_from_collision_and_invalid_lap(self):
        self.metrics.observe(transition(70, crashed=True))
        self.assertAlmostEqual(reward(self.metrics), -9.99)
        self.assertEqual(self.metrics.values()["collision_any"], 0)
        self.assertEqual(self.metrics.values()["offroad"], 0)
        self.assertEqual(self.metrics.values()["fall_without_collision"], 1)

    def test_transient_offroad_is_retained_and_events_combine(self):
        self.metrics.observe(transition(40, on_track=False))
        self.metrics.observe(transition(70))
        self.assertAlmostEqual(reward(self.metrics), -4.99)
        self.metrics.observe(transition(70, crashed=True, contact={"object": "wall"}))
        self.assertAlmostEqual(reward(self.metrics), -14.99)

    def test_signed_progress_and_bounds(self):
        for distance, expected in [
            (5020, 1),
            (10020, 1),
            (2520, 0.5),
            (10, 0),
            (20, 0),
        ]:
            self.metrics.observe(transition(distance))
            self.assertEqual(self.metrics.values()["progress"], expected)
            self.assertEqual(self.metrics.report()["legal_progress_m"], distance - 20)
        self.assertFalse(self.metrics.report()["reference_trajectory_penalty"])

    def test_invalid_inputs_fail(self):
        for length in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                EpisodeMetrics(track_length_m=length, start_legal_distance_m=0)
        with self.assertRaises(ValueError):
            self.metrics.observe(transition(float("nan")))
        with self.assertRaises(KeyError):
            self.metrics.observe({})

    def test_stale_or_reference_reward_rejected(self):
        terms = [
            SimpleNamespace(kind="metric", metric_name=name, scale=scale)
            for name, scale in REWARD_SCALES.items()
        ]
        validate_reward_terms(terms)
        for invalid in [
            terms[:3],
            terms + [terms[0]],
            terms + [SimpleNamespace(kind="distance_to_gt")],
        ]:
            with self.assertRaises(ValueError):
                validate_reward_terms(invalid)


if __name__ == "__main__":
    unittest.main()
