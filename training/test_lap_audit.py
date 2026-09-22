"""Reject altered provenance even when lap completion fields look successful."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from lap_audit import audit_lap
from lap_policy import parse_action


class LapAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "episode.jsonl"
        self.controls = parse_action("control_bike 5 20 0 0")
        self.header = {
            "type": "episode",
            "episode_id": "episode",
            "policy_id": "model",
            "track_sha256": "track",
            "initial_state": {"tick": 0, "crashed": False},
            "start_station": 0,
        }
        self.rows = []
        self.decisions = []
        for tick in range(1, 384):
            done = tick == 383
            self.rows.append(
                {
                    "type": "transition",
                    "episode_id": "episode",
                    "policy_id": "model",
                    "previous_tick": tick - 1,
                    "tick": tick,
                    "requested_controls": dict(self.controls),
                    "state": {"tick": tick, "crashed": False},
                    "track": {"on_track": True, "lap_valid": True},
                    "events": [{"type": "gate", "gate": 0 if done else tick // 12}]
                    if tick % 12 == 0 or done
                    else [],
                    "terminated": done,
                    "truncated": False,
                    "termination_reason": "lap_completed" if done else "",
                    "truncation_reason": "",
                }
            )
            if tick % 12 == 0 or done:
                self.decisions.append(
                    {
                        "action_index": len(self.decisions),
                        "episode_id": "episode",
                        "adapter_sha256": "a" * 64,
                        "tick": tick,
                        "completion": "control_bike 5 20 0 0",
                        "controls": dict(self.controls),
                    }
                )
        last = self.rows[-1]
        self.final = {
            key: copy.deepcopy(last[key])
            for key in (
                "episode_id",
                "policy_id",
                "tick",
                "state",
                "track",
                "terminated",
                "truncated",
                "termination_reason",
                "truncation_reason",
            )
        }
        self.final["track"]["completed_laps"] = 1
        self.final["rollout_valid"] = True

    def audit(self, paths=None):
        self.path.write_text(
            "".join(json.dumps(row) + "\n" for row in [self.header, *self.rows])
        )
        return audit_lap(
            paths or [self.path],
            episode_id="episode",
            policy_id="model",
            track_sha256="track",
            final_observation=self.final,
            decisions=self.decisions,
        )

    def test_complete_lap_and_partial_final_action(self):
        result = self.audit()
        self.assertTrue(result["success"])
        self.assertEqual(result["recorded_transitions"], 383)
        self.assertEqual(result["actions"], 32)

    def test_metadata_keeps_full_lap_audit_strict(self):
        self.rows.insert(
            0, {"type": "model_decision", "tick": 0, "controls": self.controls}
        )
        self.rows.insert(7, {"type": "camera_observation", "tick": 6})
        self.rows.insert(14, {"type": "snapshot", "provenance": {}})
        result = self.audit()
        self.assertTrue(result["success"])
        self.assertEqual(result["recorded_transitions"], 383)
        transition = next(row for row in self.rows if row.get("tick") == 24)
        transition["requested_controls"]["steer"] = 0.7
        with self.assertRaisesRegex(ValueError, "recorded controls"):
            self.audit()

    def test_unknown_or_failure_record_rejected(self):
        for kind in ("environment_failure", "unknown"):
            with self.subTest(kind=kind):
                self.rows.insert(0, {"type": kind})
                with self.assertRaisesRegex(ValueError, "Unexpected simulator"):
                    self.audit()
                self.rows.pop(0)

    def test_recorded_controls_must_match_generated_text(self):
        self.rows[23]["requested_controls"]["steer"] = 0.7
        with self.assertRaisesRegex(ValueError, "recorded controls"):
            self.audit()

    def test_submitted_controls_cannot_override_model(self):
        self.decisions[0]["controls"]["steer"] = 0.7
        with self.assertRaisesRegex(ValueError, "generated and submitted"):
            self.audit()

    def test_identity_and_tick_corruption_rejected(self):
        for target, key, value in (
            (self.header, "track_sha256", "wrong"),
            (self.header, "policy_id", "teacher"),
            (self.rows[8], "episode_id", "other"),
            (self.rows[8], "policy_id", "teacher"),
            (self.rows[8], "previous_tick", 0),
            (self.final, "tick", 384),
        ):
            with self.subTest(key=key, value=value):
                original = target[key]
                target[key] = value
                with self.assertRaises(ValueError):
                    self.audit()
                target[key] = original

    def test_missing_tail_and_state_mutation_rejected(self):
        tail = self.rows.pop()
        with self.assertRaisesRegex(ValueError, "reach final tick"):
            self.audit()
        self.rows.append(tail)
        self.final["state"]["crashed"] = True
        with self.assertRaisesRegex(ValueError, "final state"):
            self.audit()

    def test_duplicate_episode_files_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.audit([self.path, self.path])

    def test_gate_skip_and_excursion_cannot_pass(self):
        self.rows[11]["events"] = []
        self.assertFalse(self.audit()["success"])
        self.rows[11]["events"] = [{"type": "gate", "gate": 1}]
        self.rows[20]["track"]["on_track"] = False
        self.assertFalse(self.audit()["success"])

    def test_honest_action_budget_failure_is_valid_audit(self):
        self.rows = self.rows[:12]
        self.decisions = self.decisions[:1]
        last = self.rows[-1]
        for key in ("tick", "state", "track", "terminated", "termination_reason"):
            self.final[key] = copy.deepcopy(last[key])
        self.final["track"]["completed_laps"] = 0
        self.assertFalse(self.audit()["success"])


if __name__ == "__main__":
    unittest.main()
