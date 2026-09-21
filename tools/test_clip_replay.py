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


if __name__ == "__main__":
    unittest.main()
