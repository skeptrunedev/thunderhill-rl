"""Independent checks for the experimental ribbon and serialized derivatives.
Run: uv run --with numpy==2.4.3 --with scipy==1.17.1 python -m unittest discover -s tools -p test_road_surface.py
"""

import unittest
import json
from pathlib import Path
import numpy as np
from build_road_surface import build, evaluate, FIELDS


class RoadSurfaceTests(unittest.TestCase):
    def test_analytic_parabolic_grade_and_varying_bank(self):
        # C(s)=(s, .1*s², 0), L=(0,0,-1), b(s)=.02*s².
        data = {
            "period_m": 10.0,
            "breaks": [0.0, 10.0],
            "road_lift_m": 0.04,
            "coefficients": [
                [
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0.1, 0, 0],
                    [0] * 6,
                    [0, 0, 0, 0.02, 0, 0],
                ]
            ],
        }
        s, u = 2.0, 3.0
        r = evaluate(data, s, u)
        expected = {
            "R": [s, 0.1 * s * s + u * 0.02 * s * s + 0.04, -u],
            "Rs": [1, 0.2 * s + u * 0.04 * s, 0],
            "Ru": [0, 0.02 * s * s, -1],
            "Rss": [0, 0.2 + u * 0.04, 0],
            "Rsu": [0, 0.04 * s, 0],
            "Ruu": [0, 0, 0],
        }
        for key in FIELDS:
            np.testing.assert_allclose(r[key], expected[key], atol=1e-12)
        normal = np.array([-(0.2 * s + u * 0.04 * s), 1, 0.02 * s * s])
        np.testing.assert_allclose(
            r["normal"], normal / np.linalg.norm(normal), atol=1e-12
        )

    def setUp(self):
        angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        self.track = {
            "length_m": 2 * np.pi,
            "samples": [
                {
                    "s": float(t),
                    "p": [60 * np.cos(t), 2 * np.sin(2 * t), 60 * np.sin(t)],
                    "bank": 0.06 * np.sin(3 * t),
                    "width": 12,
                }
                for t in angles
            ],
        }
        self.data = build(self.track)

    def test_independent_circle_positions_and_derivatives(self):
        for s in np.linspace(0.01, 6.27, 19):
            r = evaluate(self.data, s, 0)
            np.testing.assert_allclose(
                r["R"],
                [60 * np.cos(s), 2 * np.sin(2 * s) + 0.04, 60 * np.sin(s)],
                atol=1e-7,
            )
            np.testing.assert_allclose(
                r["Rs"], [-60 * np.sin(s), 4 * np.cos(2 * s), 60 * np.cos(s)], atol=2e-7
            )
            np.testing.assert_allclose(
                r["Rss"],
                [-60 * np.cos(s), -8 * np.sin(2 * s), -60 * np.sin(s)],
                atol=4e-6,
            )

    def test_all_knots_and_period_seam(self):
        for s in self.data["breaks"]:
            for u in [-6.0, 0.0, 6.0]:
                before, after = (
                    evaluate(self.data, s - 1e-9, u),
                    evaluate(self.data, s + 1e-9, u),
                )
                for key in FIELDS + ("normal",):
                    np.testing.assert_allclose(before[key], after[key], atol=2e-6)
        for key in FIELDS:
            np.testing.assert_allclose(
                evaluate(self.data, -0.3, 4)[key],
                evaluate(self.data, 2 * np.pi - 0.3, 4)[key],
                atol=1e-12,
            )

    def test_historical_knots_full_width_continuity(self):
        path = Path(__file__).resolve().parents[1] / "godot/data/track.json"
        track = json.loads(path.read_text())
        data = build(track)
        for row in track["samples"]:
            for u in [-row["width"] / 2, 0.0, row["width"] / 2]:
                before = evaluate(data, row["s"] - 1e-8, u)
                after = evaluate(data, row["s"] + 1e-8, u)
                for key in FIELDS + ("normal",):
                    np.testing.assert_allclose(
                        before[key], after[key], rtol=0, atol=5e-6
                    )

    def test_derivatives_of_surface_off_center(self):
        h = 1e-4
        for s in [0.43, 1.13, 2.83, 4.15, 6.1]:
            for u in [-6.0, 2.0, 6.0]:
                r = evaluate(self.data, s, u)
                minus, plus = (
                    evaluate(self.data, s - h, u),
                    evaluate(self.data, s + h, u),
                )
                lateral_minus, lateral_plus = (
                    evaluate(self.data, s, u - h),
                    evaluate(self.data, s, u + h),
                )
                np.testing.assert_allclose(
                    (plus["R"] - minus["R"]) / (2 * h), r["Rs"], atol=2e-6
                )
                np.testing.assert_allclose(
                    (plus["Rs"] - minus["Rs"]) / (2 * h), r["Rss"], atol=2e-6
                )
                np.testing.assert_allclose(
                    (lateral_plus["Rs"] - lateral_minus["Rs"]) / (2 * h),
                    r["Rsu"],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    (lateral_plus["R"] - lateral_minus["R"]) / (2 * h),
                    r["Ru"],
                    atol=1e-8,
                )
                self.assertLess(abs(np.dot(r["normal"], r["Rs"])), 1e-12)
                self.assertLess(abs(np.dot(r["normal"], r["Ru"])), 1e-12)


if __name__ == "__main__":
    unittest.main()
