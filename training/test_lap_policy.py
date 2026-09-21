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
                features = json.loads(row["prompt"].split("\n")[1])
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

            manifest = build_dataset(
                folder / "driver.jsonl",
                folder / "episode.jsonl",
                folder / "recovery",
                recovery_grid=True,
            )
            recovered = [
                json.loads(line)
                for line in (folder / "recovery/train.jsonl").read_text().splitlines()
            ]
            recovered_eval = [
                json.loads(line)
                for line in (folder / "recovery/eval.jsonl").read_text().splitlines()
            ]
            self.assertEqual(recovered_eval, evaluation)
            self.assertEqual([row for row in recovered if "recovery" not in row], train)
            extra = [row for row in recovered if "recovery" in row]
            self.assertEqual(manifest["recovery_grid"]["examples"], 72)
            self.assertEqual(len(extra), 72)
            self.assertEqual({row["source_row"] for row in extra}, {0, 20, 80})
            self.assertTrue(all(row["split"] == "train" for row in extra))
            speed_counts = {speed: 0 for speed in (0, 2, 4, 6, 8, 10)}
            for row in extra:
                features = json.loads(row["prompt"].split("\n")[1])
                config = row["recovery"]
                speed_counts[config["speed_bin_m_s"]] += 1
                self.assertGreaterEqual(features["speed"], 0)
                self.assertLessEqual(
                    abs(features["speed"] - config["speed_bin_m_s"]), 0.901
                )
                if config["centered_exact"]:
                    self.assertEqual(features["speed"], config["speed_bin_m_s"])
                    self.assertEqual(config["heading_offset_rad"], 0)
                    self.assertEqual(config["lateral_offset_m"], 0)
                self.assertEqual(features["speed"], row["recovery"]["speed_m_s"])
                self.assertNotIn("station", features)
                self.assertNotIn("tick", features)
                controls = decode_action(row["completion"])
                if features["speed"] == 0:
                    self.assertGreater(controls["throttle"], 0)
                    self.assertEqual(controls["front_brake"], 0)
                if features["speed"] == 10:
                    self.assertEqual(controls["throttle"], 0)
                    self.assertGreater(controls["front_brake"], 0)
            self.assertEqual(set(speed_counts.values()), {12})
            self.assertEqual(
                sum(row["recovery"]["centered_exact"] for row in extra), 18
            )
            self.assertTrue(
                any(
                    row["recovery"]["speed_m_s"] != row["recovery"]["speed_bin_m_s"]
                    for row in extra
                )
            )
            self.assertEqual(manifest["recovery_grid"]["speed_jitter"]["seed"], 71)
            build_dataset(
                folder / "driver.jsonl",
                folder / "episode.jsonl",
                folder / "recovery_repeat",
                recovery_grid=True,
            )
            self.assertEqual(
                (folder / "recovery/train.jsonl").read_bytes(),
                (folder / "recovery_repeat/train.jsonl").read_bytes(),
            )
            self.assertEqual(
                {row["recovery"]["heading_offset_rad"] for row in extra},
                {-0.12, 0, 0.12},
            )
            self.assertEqual(
                {row["recovery"]["lateral_offset_m"] for row in extra}, {-1, 0, 1}
            )


if __name__ == "__main__":
    unittest.main()
