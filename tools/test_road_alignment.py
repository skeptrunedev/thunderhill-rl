# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "matplotlib==3.10.8"]
# ///
"""Independent analytical tests for the local lidar curvature audit."""

import unittest

import numpy as np
from audit_road_alignment import fit_height_graph, smooth_plan_candidate


class PlanStudyTests(unittest.TestCase):
    def fixture(self):
        # Nonuniform samples on a closed square with known metric station.
        stations = np.unique(np.r_[np.arange(0, 400, 3.0), [50, 100, 180, 200, 300]])
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]])
        xz = np.column_stack(
            [
                np.interp(stations, [0, 100, 200, 300, 400], corners[:, i])
                for i in range(2)
            ]
        )
        return (
            {
                "length_m": 400.0,
                "samples": [
                    {"s": float(s), "p": [float(p[0]), 7.0, float(p[1])]}
                    for s, p in zip(stations, xz)
                ],
            },
            stations,
            xz,
        )

    def test_local_curve_softens_corner_without_moving_outside(self):
        track, stations, original = self.fixture()
        before = repr(track)
        candidate, report = smooth_plan_candidate(track, 50, 180, 9)
        fixed = (stations <= 50) | (stations >= 180)
        np.testing.assert_array_equal(candidate[fixed], original[fixed])
        self.assertEqual(repr(track), before)
        self.assertAlmostEqual(report["maximum_baseline_heading_change_deg"], 90)
        self.assertLess(report["maximum_candidate_heading_change_deg"], 20)
        self.assertGreater(report["maximum_center_displacement_m"], 0)
        np.testing.assert_array_equal(
            report["stations_m"], stations[(stations >= 50) & (stations <= 180)]
        )

    def test_invalid_smoothing_or_interval_rejected(self):
        track, _, _ = self.fixture()
        for sigma in [0, -1, float("nan"), float("inf"), 21]:
            with self.assertRaises(ValueError):
                smooth_plan_candidate(track, 50, 180, sigma)
        for start, end in [(50, 100), (180, 50), (-1, 180), (50, 401)]:
            with self.assertRaises(ValueError):
                smooth_plan_candidate(track, start, end, 9)


class HeightGraphTests(unittest.TestCase):
    def test_tilted_quadratic_and_one_sided_support(self):
        for low in [-3.0, 0.25]:
            x, z = np.meshgrid(np.linspace(low, 3, 11), np.linspace(-3, 3, 13))
            heights = (
                7 + 0.2 * x - 0.1 * z + 0.01 * x * x + 0.003 * x * z - 0.02 * z * z
            )
            result = fit_height_graph(
                np.column_stack([x.ravel(), z.ravel()]), heights.ravel(), [1, 0]
            )
            self.assertAlmostEqual(result["height_m"], 7, places=10)
            np.testing.assert_allclose(result["gradient"], [0.2, -0.1], atol=1e-11)
            np.testing.assert_allclose(
                result["hessian"], [[0.02, 0.003], [0.003, -0.04]], atol=1e-11
            )
            expected = 0.02 / (np.sqrt(1.05) * 1.04)
            self.assertAlmostEqual(
                result["lidar_directional_normal_curvature_per_m"], expected, places=10
            )
            self.assertLess(result["rmse_m"], 1e-10)

    def test_plane_has_zero_curvature_in_any_direction(self):
        x, z = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        for direction in [[1, 0], [0, 4], [3, -2]]:
            result = fit_height_graph(
                np.column_stack([x.ravel(), z.ravel()]),
                (0.3 * x + 0.4 * z).ravel(),
                direction,
            )
            self.assertAlmostEqual(
                result["lidar_directional_normal_curvature_per_m"], 0, places=11
            )

    def test_collinear_data_is_not_a_surface_measurement(self):
        x = np.arange(30)
        with self.assertRaisesRegex(ValueError, "Rank deficient"):
            fit_height_graph(np.column_stack([x, x]), x * 0.2, [1, 0])


if __name__ == "__main__":
    unittest.main()
