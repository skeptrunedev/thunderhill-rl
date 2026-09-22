"""Comparison must be supported by raw recordings, identical physics and learning."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compare_laps import compare, digest, main
import test_lap_audit


class CompareLapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = {}
        for name, ticks in (("baseline", 383), ("candidate", 382)):
            directory = self.root / name
            directory.mkdir()
            adapter = directory / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(name.encode())
            adapter_hash = digest(adapter / "adapter_model.safetensors")
            fixture = test_lap_audit.LapAuditTests()
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            fixture.path = directory / "recording.jsonl"
            if ticks == 382:
                fixture.rows.pop(-2)
                fixture.rows[-1]["tick"] = ticks
                fixture.rows[-1]["previous_tick"] = ticks - 1
                fixture.rows[-1]["state"]["tick"] = ticks
                fixture.final["tick"] = ticks
                fixture.final["state"]["tick"] = ticks
                fixture.decisions[-1]["tick"] = ticks
            fixture.header.update(
                {
                    "physics_dt": 1 / 120,
                    "physics_version": "fixture-v1",
                    "parameters": {"mass": 200},
                    "terrain_sha256": "b" * 64,
                    "surface_sha256": "c" * 64,
                    "obstacle_collision": {"enabled": True},
                }
            )
            fixture.header["initial_state"].update(speed=0, elapsed=0)
            for row in fixture.rows:
                row["state"]["elapsed"] = row["tick"] / 120
            fixture.final["state"]["elapsed"] = ticks / 120
            fixture.final["sim_time"] = ticks / 120
            # The low level fixture uses arbitrary IDs; make real evaluator identities.
            result = fixture.audit()
            contents = fixture.path.read_text().replace(
                '"model"', '"interactive-step-lap-eval-' + adapter_hash[:12] + '"'
            )
            rows = [json.loads(line) for line in contents.splitlines()]
            rows[0]["track_sha256"] = "a" * 64
            fixture.path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            fixture.final["policy_id"] = (
                "interactive-step-lap-eval-" + adapter_hash[:12]
            )
            for decision in fixture.decisions:
                decision["adapter_sha256"] = adapter_hash
            (directory / "decisions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in fixture.decisions)
            )
            summary = {
                **result,
                "model": "fixture-model",
                "adapter_sha256": adapter_hash,
                "teacher_used_at_inference": False,
                "sim_seconds": ticks / 120,
                "final_observation": fixture.final,
            }
            (directory / "summary.json").write_text(json.dumps(summary))
            self.args[name] = directory
            self.args[name + "_adapter"] = adapter
        training = self.root / "training.json"
        training.write_text(
            json.dumps(
                {
                    "model": "fixture-model",
                    "ok": True,
                    "reloaded_logits_match": True,
                    "all_recording_audits_passed": True,
                    "optimizer_steps": 1,
                    "max_adapter_delta": 0.01,
                    "initial_adapter_sha256": digest(
                        self.args["baseline_adapter"] / "adapter_model.safetensors"
                    ),
                    "current_adapter_sha256": digest(
                        self.args["candidate_adapter"] / "adapter_model.safetensors"
                    ),
                }
            )
        )
        self.args["training_summary"] = [training]

    def mutate_summary(self, key, value):
        path = self.args["candidate"] / "summary.json"
        data = json.loads(path.read_text())
        data[key] = value
        path.write_text(json.dumps(data))

    def test_faster_lap(self):
        result = compare(**self.args)
        self.assertTrue(result["improved"])
        self.assertAlmostEqual(result["seconds_saved"], 1 / 120)

    def test_summary_lies_rejected(self):
        for key, value in (
            ("model", "other-model"),
            ("sim_seconds", 0.1),
            ("actions", 1),
            ("teacher_used_at_inference", None),
            ("adapter_sha256", "f" * 64),
        ):
            with self.subTest(key=key):
                path = self.args["candidate"] / "summary.json"
                original = path.read_text()
                self.mutate_summary(key, value)
                with self.assertRaises(ValueError):
                    compare(**self.args)
                path.write_text(original)

    def test_changed_physics_or_start_rejected(self):
        path = self.args["candidate"] / "recording.jsonl"
        original = path.read_text()
        for key, value in (
            ("parameters", {"mass": 100}),
            ("start_station", 1),
            ("surface_sha256", "d" * 64),
            ("initial_state", {"tick": 0, "elapsed": 0, "speed": 1, "crashed": False}),
        ):
            with self.subTest(key=key):
                rows = [json.loads(line) for line in original.splitlines()]
                rows[0][key] = value
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    compare(**self.args)
        path.write_text(original)

    def test_failed_lap_cannot_be_claimed_successful(self):
        path = self.args["candidate"] / "recording.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[21]["track"]["on_track"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Summary disagrees"):
            compare(**self.args)
        self.mutate_summary("success", False)
        self.mutate_summary("offtrack_ticks", 1)
        with self.assertRaisesRegex(ValueError, "complete legal lap"):
            compare(**self.args)

    def test_slower_candidate_not_improvement(self):
        for key in ("baseline", "baseline_adapter"):
            other = key.replace("baseline", "candidate")
            self.args[key], self.args[other] = self.args[other], self.args[key]
        path = self.args["training_summary"][0]
        row = json.loads(path.read_text())
        row["initial_adapter_sha256"], row["current_adapter_sha256"] = (
            row["current_adapter_sha256"],
            row["initial_adapter_sha256"],
        )
        path.write_text(json.dumps(row))
        self.assertFalse(compare(**self.args)["improved"])

    def test_learning_chain_and_zero_update_rejected(self):
        path = self.args["training_summary"][0]
        row = json.loads(path.read_text())
        final = row["current_adapter_sha256"]
        row["current_adapter_sha256"] = "d" * 64
        path.write_text(json.dumps(row))
        second = self.root / "second.json"
        row.update(initial_adapter_sha256="d" * 64, current_adapter_sha256=final)
        second.write_text(json.dumps(row))
        self.args["training_summary"].append(second)
        self.assertTrue(compare(**self.args)["improved"])
        for key, value in (
            ("max_adapter_delta", float("nan")),
            ("optimizer_steps", 0),
            ("initial_adapter_sha256", "e" * 64),
        ):
            with self.subTest(key=key):
                second.write_text(json.dumps({**row, key: value}))
                with self.assertRaises(ValueError):
                    compare(**self.args)

    def test_output_never_overwritten(self):
        output = self.root / "comparison.json"
        output.write_text("preserved")
        argv = ["compare_laps.py", "--output", str(output)]
        for key, value in self.args.items():
            for item in value if isinstance(value, list) else [value]:
                argv.extend(["--" + key.replace("_", "-"), str(item)])
        with patch("sys.argv", argv), self.assertRaises(FileExistsError):
            main()
        self.assertEqual(output.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
