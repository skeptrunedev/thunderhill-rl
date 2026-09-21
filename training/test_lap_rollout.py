"""Reward accounting and independent rollout provenance adversarial checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from lap_policy import parse_action
from lap_prefix import load_prefix
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
        return audit_rollout(
            [self.path, *getattr(self, "extra_paths", [])], self.record, "track"
        )

    def make_snapshot_episode(self):
        snapshot = {
            "version": "worker-snapshot-v1",
            "snapshot_id": "a" * 32,
            "source_episode_id": "source",
            "source_policy_id": "policy",
            "source_tick": 12,
        }
        source_header = {**self.header, "episode_id": "source"}
        source_rows = [{**row, "episode_id": "source"} for row in self.rows[:12]]
        source_path = self.path.parent / "source.jsonl"
        source_path.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in [
                    source_header,
                    *source_rows,
                    {"type": "snapshot", "provenance": snapshot},
                ]
            )
        )
        self.extra_paths = [source_path]
        self.header["initial_state"] = copy.deepcopy(source_rows[-1]["state"])
        self.header["start_station"] = 100
        self.header["snapshot"] = snapshot
        self.header["initial_track"] = {"lap_valid": True}
        self.rows = self.rows[12:]
        self.record["decisions"] = self.record["decisions"][1:]
        self.record["snapshot_provenance"] = copy.deepcopy(snapshot)
        self.record["prefix_source_provenance"] = {
            "all_source_controls_and_states_verified": True
        }
        self.record["reset_snapshot"] = copy.deepcopy(self.record["branch_snapshot"])
        self.record["reward_components"] = rollout_reward(self.rows, 12)

    def test_snapshot_branch_preserves_absolute_tick_and_excludes_prefix(self):
        self.make_snapshot_episode()
        self.assertTrue(self.audit()["reward_verified"])
        self.assertEqual(self.record["reward_components"]["recorded_prefix_ticks"], 0)
        self.assertEqual(self.record["reward_components"]["total"], 12)

    def test_snapshot_creation_and_header_provenance_required(self):
        self.make_snapshot_episode()
        self.header["snapshot"]["snapshot_id"] = "b" * 32
        with self.assertRaisesRegex(ValueError, "Snapshot reset"):
            self.audit()
        self.header["snapshot"] = copy.deepcopy(self.record["snapshot_provenance"])
        source_path = self.extra_paths[0]
        lines = source_path.read_text().splitlines()
        source_path.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Snapshot creation"):
            self.audit()

    def test_invalid_first_action_from_snapshot_has_no_physics_ticks(self):
        self.make_snapshot_episode()
        self.rows = []
        self.record["decisions"][0] = {
            "phase": "sampled",
            "before_tick": 12,
            "completion": "nonsense",
            "adapter_sha256": "new",
            "error": "Invalid action",
        }
        self.record["final_observation"]["tick"] = 12
        self.record["final_observation"]["state"] = copy.deepcopy(
            self.header["initial_state"]
        )
        self.record["reward_components"] = rollout_reward([], 12, True)
        self.assertEqual(self.audit()["transitions"], 0)

    def test_saved_prefix_bound_to_recorded_model_controls_and_prompts(self):
        self.audit()
        decision_path = self.path.parent / "decisions.jsonl"
        decision = {
            "action_index": 0,
            "episode_id": "episode",
            "adapter_sha256": "old",
            "tick": 12,
            "prompt": "tick=0",
            "completion": "control_bike 0 20 0 0",
            "completion_ids": [1],
            "controls": parse_action("control_bike 0 20 0 0"),
        }
        decision_path.write_text(json.dumps(decision) + "\n")
        (self.path.parent / "summary.json").write_text(
            json.dumps(
                {
                    "adapter_sha256": "old",
                    "teacher_used_at_inference": False,
                    "recordings": [str(self.path)],
                }
            )
        )

        class Road:
            track_sha256 = "track"

            def prompt(self, observation):
                return f"tick={observation['state']['tick']}"

        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return "control_bike 0 20 0 0"

        prefix, provenance = load_prefix(decision_path, 1, "old", Road(), Tokenizer())
        self.assertEqual(prefix[0]["source_after_state"]["tick"], 12)
        self.assertTrue(provenance["all_source_controls_and_states_verified"])
        decision["controls"]["steer"] = 1
        decision_path.write_text(json.dumps(decision) + "\n")
        with self.assertRaisesRegex(ValueError, "controls differ"):
            load_prefix(decision_path, 1, "old", Road(), Tokenizer())

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
