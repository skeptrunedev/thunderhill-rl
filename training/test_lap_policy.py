"""Strict controls and causal dataset pairing for privileged telemetry pilot."""

import json
import tempfile
import unittest
from pathlib import Path

from lap_policy import RoadTelemetry, build_dataset, decode_action, encode_action


class LapPolicyTests(unittest.TestCase):
    def test_codec_rejects_malformed_or_unbounded_controls(self):
        for text in (
            "control_bike 1001 0 0 0",
            "control_bike 0 -1 0 0",
            "control_bike 0 0 0 0 extra",
            "control_bike 0 0.1 0 0",
        ):
            with self.assertRaises(ValueError):
                decode_action(text)
        for text in ("control_bike 0 20 0 0", "control_bike -750 0 65 12"):
            self.assertEqual(encode_action(decode_action(text)), text)

    def test_interpolation_wraps(self):
        road = RoadTelemetry()
        self.assertEqual(road.point(0), road.samples[0]["p"])
        self.assertEqual(road.point(0), road.point(road.length))
        self.assertEqual(road.point(-1), road.point(road.length - 1))
        midpoint = (road.stations[0] + road.stations[1]) / 2
        self.assertEqual(
            road.point(midpoint),
            [(a + b) / 2 for a, b in zip(road.samples[0]["p"], road.samples[1]["p"])],
        )

    def test_dataset_uses_preaction_state_and_keeps_holdout_groups_separate(self):
        road = RoadTelemetry()
        state = {
            "speed": 0,
            "lean": 0,
            "heading": 0,
            "position": road.point(0),
            "tick": 0,
        }
        header = {
            "type": "episode",
            "track_sha256": road.track_sha256,
            "initial_state": state,
            "start_station": 0,
            "episode_id": "fixture",
        }
        rows = []
        for index in range(100):
            rows.append(
                {
                    "request": {
                        "episode_id": "fixture",
                        "expected_tick": index * 12,
                        "controls": {
                            "steer": 0,
                            "throttle": 0.2,
                            "front_brake": 0,
                            "rear_brake": 0,
                        },
                    },
                    "observation": {
                        "state": {**state, "speed": index + 1},
                        "tick": (index + 1) * 12,
                        "track": {"progress": 0, "lateral_m": 0},
                    },
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "episode.jsonl").write_text(json.dumps(header) + "\n")
            (folder / "driver.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            build_dataset(
                folder / "driver.jsonl", folder / "episode.jsonl", folder / "dataset"
            )
            train = [
                json.loads(line)
                for line in (folder / "dataset/train.jsonl").read_text().splitlines()
            ]
            evaluation = [
                json.loads(line)
                for line in (folder / "dataset/eval.jsonl").read_text().splitlines()
            ]
            for row in train + evaluation:
                features = json.loads(row["prompt"].split("\n")[-2])
                self.assertEqual(features["speed"], row["source_row"])
                self.assertNotIn("tick", features)
                self.assertNotIn("station", features)
            self.assertFalse(
                {row["source_row"] for row in train}
                & {row["source_row"] for row in evaluation}
            )
            self.assertEqual(
                [row["source_row"] for row in evaluation], list(range(45, 55))
            )


if __name__ == "__main__":
    unittest.main()
