"""Reward accounting and independent rollout provenance adversarial checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from lap_policy import parse_action
from lap_rollout import audit_rollout, physical_snapshot, rollout_reward


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "episode.jsonl"
        controls = parse_action("control_bike 0 20 0 0")
        self.header = {
            "type": "episode",
            "episode_id": "episode",
            "policy_id": "policy",
            "track_sha256": "track",
            "start_station": 0,
            "initial_state": {"tick": 0, "crashed": False},
        }
        self.rows = [
            {
                "type": "transition",
                "episode_id": "episode",
                "policy_id": "policy",
                "previous_tick": tick - 1,
                "tick": tick,
                "requested_controls": dict(controls),
                "state": {"tick": tick, "crashed": False},
                "track": {"on_track": True, "lap_valid": True},
                "reward_components": {"legal_progress_m": 100 if tick <= 12 else 1},
                "terminated": False,
                "truncated": False,
                "termination_reason": "",
                "truncation_reason": "",
            }
            for tick in range(1, 25)
        ]
        self.record = {
            "episode_id": "episode",
            "policy_id": "policy",
            "prefix_tick": 12,
            "prefix_adapter_sha256": "old",
            "current_adapter_sha256": "new",
            "reset_snapshot": {
                "tick": 0,
                "state": copy.deepcopy(self.header["initial_state"]),
                "track": {},
            },
            "branch_snapshot": physical_snapshot(copy.deepcopy(self.rows[11])),
            "final_observation": {
                **copy.deepcopy(self.rows[-1]),
                "rollout_valid": True,
            },
            "decisions": [
                {
                    "phase": phase,
                    "before_tick": i * 12,
                    "after_tick": (i + 1) * 12,
                    "completion": "control_bike 0 20 0 0",
                    "controls": dict(controls),
                    "adapter_sha256": "old" if i == 0 else "new",
                }
                for i, phase in enumerate(("prefix", "sampled"))
            ],
            "reward_components": rollout_reward(self.rows, 12),
        }

    def audit(self):
        self.path.write_text(
            "".join(json.dumps(row) + "\n" for row in [self.header, *self.rows])
        )
        return audit_rollout([self.path], self.record, "track")

    def test_prefix_progress_excluded(self):
        self.assertEqual(self.record["reward_components"]["total"], 12)
        self.assertTrue(self.audit()["reward_verified"])

    def test_failure_and_syntax_penalties_are_separate_and_once(self):
        self.rows[14]["track"]["lap_valid"] = False
        self.rows[15]["state"]["crashed"] = True
        reward = rollout_reward(self.rows, 12, True)
        self.assertEqual(reward["failure_penalty"], -25)
        self.assertEqual(reward["syntax_penalty"], -10)
        self.assertEqual(reward["total"], -23)

    def test_invalid_first_action_preserves_prefix_without_rewarding_it(self):
        self.rows = self.rows[:12]
        self.record["decisions"][1] = {
            "phase": "sampled",
            "before_tick": 12,
            "completion": "nonsense",
            "adapter_sha256": "new",
            "error": "Invalid action",
        }
        self.record["final_observation"] = {
            **copy.deepcopy(self.rows[-1]),
            "rollout_valid": True,
        }
        self.record["reward_components"] = rollout_reward(self.rows, 12, True)
        self.assertEqual(self.record["reward_components"]["total"], -10)
        self.assertTrue(self.audit()["reward_verified"])

    def test_corrupt_control_reward_or_checkpoint_rejected(self):
        cases = [
            (self.rows[13]["requested_controls"], "steer", 1),
            (self.record["reward_components"], "total", 999),
            (self.record["decisions"][1], "adapter_sha256", "old"),
            (self.rows[13], "previous_tick", 0),
            (self.header, "track_sha256", "wrong"),
            (self.record["branch_snapshot"]["state"], "tick", 0),
            (self.record["final_observation"]["state"], "tick", 100),
        ]
        for target, key, value in cases:
            with self.subTest(key=key, value=value):
                original = target[key]
                target[key] = value
                with self.assertRaises(ValueError):
                    self.audit()
                target[key] = original

    def test_nonfinite_reward_is_infrastructure_error(self):
        self.rows[-1]["reward_components"]["legal_progress_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            rollout_reward(self.rows, 12)


if __name__ == "__main__":
    unittest.main()
