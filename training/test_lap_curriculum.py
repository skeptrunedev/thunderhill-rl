"""Exercise orchestration with fake children and independently audited recordings."""

import json
import tempfile
import unittest
from pathlib import Path

from lap_audit import audit_lap
from lap_policy import RoadTelemetry, parse_action
from train_lap_curriculum import Config, checkpoint, digest, run


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = self.root / "initial"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"initial")
        self.config = Config(
            self.adapter, "fake-godot", self.root / "run", max_cycles=1
        )
        self.calls = []
        self.outcomes = [False, True]
        self.training_exit = 0
        self.eval_exit = None
        self.corrupt = False

    def runner(self, command, log):
        self.calls.append(command)
        log.write_text("fake child stdout\n")

        def arg(name):
            return command[command.index(name) + 1]

        directory = Path(arg("--output"))
        directory.mkdir()
        adapter_hash = checkpoint(arg("--adapter"))
        if command[2].endswith("train_lap_grpo.py"):
            adapter = directory / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"trained")
            decisions = Path(arg("--prefix-decisions"))
            summary = {
                "ok": True,
                "all_recording_audits_passed": True,
                "reloaded_logits_match": True,
                "optimizer_steps": self.config.steps,
                "initial_adapter_sha256": adapter_hash,
                "prefix_adapter_sha256": adapter_hash,
                "current_adapter_sha256": checkpoint(adapter),
                "prefix_and_continuation_tokens_trained": 0,
                "prefix_source_provenance": {
                    "all_source_controls_and_states_verified": True,
                    "actions": int(arg("--prefix-actions")),
                    "decision_sha256": digest(decisions),
                    "adapter_sha256": adapter_hash,
                },
            }
            code = self.training_exit
        else:
            success = self.outcomes.pop(0)
            policy = "interactive-step-lap-eval-" + adapter_hash[:12]
            controls = parse_action("control_bike 0 20 0 0")
            header = {
                "type": "episode",
                "episode_id": "episode",
                "policy_id": policy,
                "track_sha256": RoadTelemetry().track_sha256,
                "start_station": 0,
                "initial_state": {"tick": 0, "crashed": False},
            }
            rows = []
            for tick in range(1, 361):
                done = tick == 360
                rows.append(
                    {
                        "type": "transition",
                        "episode_id": "episode",
                        "policy_id": policy,
                        "previous_tick": tick - 1,
                        "tick": tick,
                        "requested_controls": controls,
                        "state": {"tick": tick, "crashed": False},
                        "track": {
                            "on_track": success or not done,
                            "lap_valid": success or not done,
                            "completed_laps": int(success and done),
                        },
                        "terminated": success and done,
                        "termination_reason": "lap_completed"
                        if success and done
                        else "",
                        "truncated": False,
                        "truncation_reason": "",
                        "events": [
                            {"type": "gate", "gate": gate}
                            for gate in [*range(1, 32), 0]
                        ]
                        if success and done
                        else [],
                    }
                )
            recording = directory / "recording.jsonl"
            recording.write_text(
                "".join(json.dumps(row) + "\n" for row in [header, *rows])
            )
            decisions = [
                {
                    "action_index": i,
                    "episode_id": "episode",
                    "adapter_sha256": adapter_hash,
                    "tick": (i + 1) * 12,
                    "controls": controls,
                    "completion": "control_bike 0 20 0 0",
                }
                for i in range(30)
            ]
            (directory / "decisions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in decisions)
            )
            final = {**rows[-1], "rollout_valid": True}
            audit = audit_lap(
                [recording],
                episode_id="episode",
                policy_id=policy,
                track_sha256=header["track_sha256"],
                final_observation=final,
                decisions=decisions,
            )
            summary = {
                **audit,
                "teacher_used_at_inference": False,
                "compiled_inference": True,
                "final_observation": final,
                "reason": "lap_completed" if success else "track_limits",
            }
            if self.corrupt:
                rows[-1]["requested_controls"] = parse_action("control_bike 1 20 0 0")
                recording.write_text(
                    "".join(json.dumps(row) + "\n" for row in [header, *rows])
                )
            code = (
                self.eval_exit if self.eval_exit is not None else (0 if success else 1)
            )
        (directory / "summary.json").write_text(json.dumps(summary))
        return code

    def test_failure_training_success_and_lineage(self):
        result = run(self.config, self.runner)
        self.assertTrue(result["success"])
        self.assertEqual(len(self.calls), 3)
        self.assertIn("--compile", self.calls[0])
        training = self.calls[1]
        self.assertEqual(training[training.index("--prefix-actions") + 1], "5")
        self.assertEqual(
            result["lineage"][1]["parent_sha256"], checkpoint(self.adapter)
        )
        self.assertEqual(
            self.calls[2][self.calls[2].index("--adapter") + 1], result["final_adapter"]
        )

    def test_success_does_not_train(self):
        self.outcomes = [True]
        self.assertTrue(run(self.config, self.runner)["success"])
        self.assertEqual(len(self.calls), 1)

    def test_final_cycle_still_evaluated_and_bounded(self):
        self.outcomes = [False, False]
        result = run(self.config, self.runner)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "maximum_training_cycles_reached")
        self.assertEqual(len(self.calls), 3)

    def test_corrupt_recording_never_trains(self):
        self.corrupt = True
        with self.assertRaisesRegex(ValueError, "recorded controls"):
            run(self.config, self.runner)
        self.assertEqual(len(self.calls), 1)

    def test_child_infrastructure_exit_never_trains(self):
        self.eval_exit = 2
        with self.assertRaisesRegex(RuntimeError, "exit 2"):
            run(self.config, self.runner)
        self.assertEqual(len(self.calls), 1)

    def test_training_error_aborts_without_evaluating_adapter(self):
        self.training_exit = 1
        with self.assertRaisesRegex(RuntimeError, "exit 1"):
            run(self.config, self.runner)
        self.assertEqual(len(self.calls), 2)

    def test_missing_summary_is_not_policy_failure(self):
        with self.assertRaises(FileNotFoundError):
            run(self.config, lambda command, log: 1)

    def test_interrupted_child_stops_curriculum(self):
        def interrupted(command, log):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            run(self.config, interrupted)
        summary = json.loads((self.config.output / "summary.json").read_text())
        self.assertEqual(summary["reason"], "operator_interrupted")


if __name__ == "__main__":
    unittest.main()
