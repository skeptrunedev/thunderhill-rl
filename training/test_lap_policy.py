"""Strict control serialization and simulator road interpolation."""

import unittest

from lap_policy import RoadTelemetry, decode_action, encode_action


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


if __name__ == "__main__":
    unittest.main()
