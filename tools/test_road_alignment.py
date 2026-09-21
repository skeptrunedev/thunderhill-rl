# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "matplotlib==3.10.8"]
# ///
"""Independent analytical tests for the local lidar curvature audit."""

import unittest

import numpy as np
from audit_road_alignment import (
    fit_height_graph,
    smooth_plan_candidate,
    sample_aerial_profile,
    section_points,
)


class AerialSectionTests(unittest.TestCase):
    def test_north_up_pixels_rgb_and_neutral_score(self):
        image = np.array(
            [[[20, 30, 20], [90, 60, 30]], [[0, 0, 0], [100, 80, 50]]], dtype=np.uint8
        )
        extent = {"xmin": 100, "xmax": 101.2, "ymin": 200, "ymax": 201.2}
        result = sample_aerial_profile(image, extent, [[100.3, 200.9], [100.9, 200.3]])
        np.testing.assert_allclose(
            result["source_pixel_column_row"], [[0.5, 0.5], [1.5, 1.5]]
        )
        self.assertEqual(result["sampled_pixel_column_row"], [[0, 0], [1, 1]])
        self.assertEqual(result["rgb_255"], [[20, 30, 20], [100, 80, 50]])
        np.testing.assert_allclose(result["neutral_color_score"], [0, 1 / 3])
        floating = sample_aerial_profile(
            image.astype(float) / 255, extent, [[100.3, 200.9]]
        )
        np.testing.assert_allclose(floating["rgb_255"], [[20, 30, 20]])

    def test_outside_nonfinite_and_missing_pixels_rejected(self):
        image = np.ones((2, 2, 4))
        extent = {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2}
        for xy in [[[-0.01, 1]], [[2, 1]], [[1, 0]], [[1, 2.01]], [[np.nan, 1]]]:
            with self.assertRaises(ValueError):
                sample_aerial_profile(image, extent, xy)
        image[0, 0, 3] = 0
        with self.assertRaisesRegex(ValueError, "missing"):
            sample_aerial_profile(image, extent, [[0.5, 1.5]])
        image[0, 0] = np.nan
        with self.assertRaises(ValueError):
            sample_aerial_profile(image, extent, [[0.5, 1.5]])

    def test_positive_left_recovers_modeled_edge_with_bank(self):
        # Eastbound travel has north (negative local z) on its left.
        result = section_points([10, 2, 20], [10, 2.6, 14], 6, [-6, 0, 6])
        np.testing.assert_allclose(result, [[10, 1.4, 26], [10, 2, 20], [10, 2.6, 14]])
        for half_width in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                section_points([0, 0, 0], [0, 0, -6], half_width, [0])


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
