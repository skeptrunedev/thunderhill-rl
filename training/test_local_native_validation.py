"""Local training uses gameplay RL, with no supervised or lap completion gates."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import local_native_validation as runner
from model_runtime import FUNCTIONGEMMA_SPEC


def campaign():
    failed = {"reason": "crash", "success": False}
    return {"complete": True, "smoke_only": False, "model": FUNCTIONGEMMA_SPEC.model,
            "revision": FUNCTIONGEMMA_SPEC.revision, "prompt_style": FUNCTIONGEMMA_SPEC.prompt_style,
            "constrained_sampling_and_training": True,
            "generations": [{"generation": i + 1,
                "collection": {"evaluation": failed, "rollouts": [failed, failed, failed]},
                "update": {"optimizer_steps": 1, "actions": 40},
                "verification": {"reloaded_logits_match": True}} for i in range(3)],
            "final_evaluation": {"evaluation": failed, "rollouts": []}}


class LocalValidationTests(unittest.TestCase):
    def test_failed_driving_does_not_prevent_valid_rl(self):
        runner.validate_campaign(campaign(), 3)

    def test_requires_actual_updates_and_native_model(self):
        for key, value in (("complete", False), ("smoke_only", True), ("model", "other"),
                           ("revision", "other"), ("prompt_style", "custom"),
                           ("constrained_sampling_and_training", False), ("generations", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_campaign({**campaign(), key: value}, 3)
        for key in ("optimizer_steps", "actions"):
            row = copy.deepcopy(campaign())
            row["generations"][0]["update"][key] = 0
            with self.assertRaisesRegex(ValueError, "optimizer update"):
                runner.validate_campaign(row, 3)

    def test_checkpoint_reload_required(self):
        row = campaign()
        row["generations"][0]["verification"]["reloaded_logits_match"] = False
        with self.assertRaisesRegex(ValueError, "reload"):
            runner.validate_campaign(row, 3)

    def exercise(self, returncode=0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        args = runner.parser().parse_args(["--godot", "/fake/godot", "--output", str(root / "output")])
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir()
            (destination / "campaign.json").write_text(json.dumps(campaign()))
            queue = destination / "video_jobs"
            queue.mkdir()
            (queue / "failed-rollout.json").write_text('{"recording": "preserved"}')
            return SimpleNamespace(returncode=returncode)

        with patch.object(runner.subprocess, "run", side_effect=execute):
            if returncode:
                with self.assertRaises(RuntimeError):
                    runner.run(args)
                result = json.loads((args.output / "diagnostic.json").read_text())
            else:
                result = runner.run(args)
        return result, calls, args.output

    def test_direct_gameplay_rl_only_with_requested_defaults(self):
        result, calls, _ = self.exercise()
        self.assertTrue(result["complete"])
        self.assertFalse(result["supervised_imitation_used"])
        self.assertFalse(result["lap_completion_required"])
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertTrue(command[2].endswith("train_full_lap_grpo.py"))
        self.assertIn("--functiongemma", command)
        for key, value in (("--generations", "3"), ("--initial-generation", "0"),
                           ("--time-budget-seconds", "30"), ("--batch-candidates", "4"),
                           ("--wandb-mode", "online"), ("--wandb-entity", "skeptrune-org")):
            self.assertEqual(command[command.index(key) + 1], value)
        for forbidden in ("--adapter", "--dataset", "--smoke", "--context-dataset"):
            self.assertNotIn(forbidden, command)
        self.assertNotIn("modal", " ".join(command))
        self.assertEqual(result["video_jobs"], 1)

    def test_failed_subprocess_preserves_recordings_and_report(self):
        result, calls, output = self.exercise(returncode=1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["complete"])
        self.assertIn("RuntimeError", result["error"])
        self.assertTrue((output / "rl/video_jobs/failed-rollout.json").is_file())
        self.assertTrue((output / "rl/campaign.json").is_file())
        self.assertEqual(result["video_jobs"], 1)

    def test_cannot_overwrite_previous_run(self):
        _, _, output = self.exercise()
        args = runner.parser().parse_args(["--godot", "/fake/godot", "--output", str(output)])
        with self.assertRaises(FileExistsError):
            runner.run(args)


if __name__ == "__main__":
    unittest.main()
