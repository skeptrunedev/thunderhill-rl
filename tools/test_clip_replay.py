import json
import tempfile
import unittest
from pathlib import Path

from clip_replay import clip_replay


class ReplayClipTest(unittest.TestCase):
    def test_preserves_states_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.jsonl", Path(folder) / "clip.jsonl"
            rows = [{"type": "episode", "physics_dt": 0.25, "policy_id": "generation-7",
                     "track_sha256": "track", "initial_state": {"tick": 0, "elapsed": 0}}]
            rows += [{"type": "transition", "tick": i, "previous_tick": i - 1,
                      "state": {"tick": i, "elapsed": i * 0.25, "position": [i, 0, 0]}}
                     for i in range(1, 9)]
            source.write_text("\n".join(map(json.dumps, rows)))
            report = clip_replay(source, output, 0.75, 1.5)
            clipped = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(clipped[0]["initial_state"], rows[2]["state"])
            self.assertEqual(clipped[0]["policy_id"], "generation-7")
            self.assertEqual(clipped[0]["track_sha256"], "track")
            self.assertEqual(clipped[1:], rows[3:7])
            self.assertEqual(report["transitions"], 4)
            with self.assertRaises(FileExistsError):
                clip_replay(source, output, 0.75, 1.5)
            with self.assertRaises(ValueError):
                clip_replay(source, Path(folder) / "too-long.jsonl", 1, 4)
            self.assertFalse((Path(folder) / "too-long.jsonl").exists())
            with self.assertRaises(ValueError):
                clip_replay(source, Path(folder) / "short-tail.jsonl", 1, 2.1)
            with self.assertRaises(ValueError):
                clip_replay(output, Path(folder) / "missing-start.jsonl", 0, 1)
            between = Path(folder) / "between-ticks.jsonl"
            clip_replay(source, between, 0.8, 1.6)
            selected = [json.loads(line) for line in between.read_text().splitlines()]
            self.assertEqual(selected[0]["initial_state"], rows[3]["state"])
            self.assertEqual(selected[1:], rows[4:7])
            rows[4]["state"]["tick"] = 8
            source.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaises(ValueError):
                clip_replay(source, Path(folder) / "invalid.jsonl", 0, 2)


class DecisionClipTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.source = Path(self.folder.name) / "episode.jsonl"
        self.sidecar = Path(self.folder.name) / "decisions.jsonl"
        self.output = Path(self.folder.name) / "clip.jsonl"
        self.controls = {"steer": 0, "throttle": 0.5}
        self.rows = [{"type": "episode", "episode_id": "e",
                      "initial_state": {"tick": 0, "elapsed": 0}}]
        self.rows += [{"type": "transition", "episode_id": "e", "tick": i,
                       "previous_tick": i - 1, "requested_controls": self.controls,
                       "state": {"tick": i, "elapsed": i * 0.25}}
                      for i in range(1, 9)]
        self.calls = [{"episode_id": "e", "action_index": i,
                       "tick": (i + 1) * 2, "controls": self.controls,
                       "completion": "control_bike 0 50 0 0"}
                      for i in range(4)]
        self.write()

    def write(self):
        self.source.write_text("\n".join(map(json.dumps, self.rows)))
        self.sidecar.write_text("\n".join(map(json.dumps, self.calls)))

    def read(self, path=None):
        return [json.loads(line) for line in (path or self.output).read_text().splitlines()]

    def test_call_start_and_identical_calls_preserved(self):
        result = clip_replay(self.source, self.output, 0, 1.5, self.sidecar)
        rows = self.read()
        events = [r for r in rows if r["type"] == "model_decision"]
        self.assertEqual([e["tick"] for e in events], [0, 2, 4])
        self.assertEqual([e["action_index"] for e in events], [0, 1, 2])
        self.assertEqual(result["model_decisions"], 3)
        for i, row in enumerate(rows):
            if row["type"] == "model_decision":
                self.assertEqual(rows[i + 1]["previous_tick"], row["tick"])
                self.assertEqual(row["text"], self.calls[0]["completion"])
        # The call at the final state tick has not yet controlled any clip frame.
        self.assertNotIn(3, [e["action_index"] for e in events])

    def test_inside_call_retains_active_event_and_reclip(self):
        clip_replay(self.source, self.output, 1, 2, self.sidecar)
        rows = self.read()
        self.assertEqual(rows[0]["initial_state"]["tick"], 3)
        self.assertEqual(rows[1]["tick"], 2)
        self.assertEqual(rows[1]["elapsed"], 0.5)
        second = self.output.with_name("second.jsonl")
        clip_replay(self.output, second, 1.5, 2)
        events = [r for r in self.read(second) if r["type"] == "model_decision"]
        self.assertEqual([r["tick"] for r in events], [4, 6])
        self.assertEqual([r["action_index"] for r in events], [2, 3])

    def test_exact_call_boundary_excludes_previous_call(self):
        clip_replay(self.source, self.output, 1.25, 2, self.sidecar)
        events = [r for r in self.read() if r["type"] == "model_decision"]
        self.assertEqual([r["tick"] for r in events], [4, 6])

    def test_rejects_wrong_episode_controls_order_and_coverage(self):
        variants = [lambda: self.calls[1].update(episode_id="other"),
                    lambda: self.calls[1].update(controls={"steer": 1}),
                    lambda: self.calls[1].update(action_index=0),
                    lambda: self.calls[1].update(tick=2),
                    lambda: self.calls.pop()]
        original = json.dumps(self.calls)
        for mutate in variants:
            with self.subTest(mutate=mutate):
                self.calls = json.loads(original)
                mutate()
                self.write()
                with self.assertRaises(ValueError):
                    clip_replay(self.source, self.output, 0, 0.5, self.sidecar)
                self.assertFalse(self.output.exists())

    def test_rejects_future_embedded_event(self):
        self.rows.insert(1, {"type": "model_decision", "tick": 2,
                             "text": "future"})
        self.write()
        with self.assertRaises(ValueError):
            clip_replay(self.source, self.output, 0, 2)

    def test_rejected_final_completion_is_not_an_applied_call(self):
        self.calls.append({"action_index": 4, "episode_id": "e",
                           "completion": "invalid", "error": "invalid controls"})
        self.write()
        result = clip_replay(self.source, self.output, 0, 2, self.sidecar)
        self.assertEqual(result["model_decisions"], 4)


    def test_explicit_display_labels_and_preservation(self):
        label = {"model_name": "Gemma 3 270M", "generation": 0}
        clip_replay(self.source, self.output, 0, 2, self.sidecar, **label)
        self.assertEqual(self.read()[0]["policy_display"], label)
        second = self.output.with_name("generation-999.jsonl")
        clip_replay(self.output, second, 1, 2)
        self.assertEqual(self.read(second)[0]["policy_display"], label)
        override = self.output.with_name("override.jsonl")
        clip_replay(second, override, 1.5, 2, model_name="Other model", generation=2)
        self.assertEqual(self.read(override)[0]["policy_display"],
                         {"model_name": "Other model", "generation": 2})

    def test_absent_display_labels_are_not_inferred(self):
        clip_replay(self.source, self.output, 0, 2)
        self.assertNotIn("policy_display", self.read()[0])

    def test_display_labels_require_complete_valid_pair(self):
        cases = [{"model_name": "Gemma"}, {"generation": 0},
                 {"model_name": "", "generation": 0},
                 {"model_name": "  ", "generation": 0},
                 {"model_name": "Gemma", "generation": -1},
                 {"model_name": "Gemma", "generation": 1.5},
                 {"model_name": "Gemma", "generation": True},
                 {"model_name": 3, "generation": 0}]
        for label in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    clip_replay(self.source, self.output, 0, 2, **label)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
