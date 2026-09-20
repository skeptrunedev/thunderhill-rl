"""Analytical checks for the experimental tensor surface.
Run with uv run --with numpy==2.4.3 --with scipy==1.17.1 python -m unittest discover -s tools -p test_cyclone_surface.py
"""

import unittest
import numpy as np
from fit_cyclone_surface import (
    knots,
    fit,
    normal_curvature,
    bending_matrix,
    validate_provenance,
)


class HeightSurfaceTests(unittest.TestCase):
    def test_invalid_inputs_and_stale_provenance(self):
        for spacing in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                knots([-1, 1], spacing)
        for strength in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                fit([], [], (), strength)
        track = {"metadata": {"lidar": {"corridor_sha256": "lidar"}}}
        road = {"metadata": {"track_sha256": "track"}}
        validate_provenance(track, road, {"track": "track", "lidar": "lidar"})
        for hashes in [
            {"track": "stale", "lidar": "lidar"},
            {"track": "track", "lidar": "stale"},
        ]:
            with self.assertRaises(ValueError):
                validate_provenance(track, road, hashes)

    def test_quadratic_height_derivatives_and_graph_curvature(self):
        grid = np.linspace(-5, 5, 31)
        xy = np.array(np.meshgrid(grid, grid, indexing="ij")).reshape(2, -1).T
        x, y = xy.T
        z = 90 + 0.2 * x - 0.1 * y + 0.03 * x * x + 0.02 * x * y - 0.04 * y * y
        t = (knots([-5, 5], 2), knots([-5, 5], 2))
        spline, _ = fit(xy, z, t, 0)
        q = np.array([[0.12, 0.71], [-2.3, 1.7], [4.2, -3.3]])
        x, y = q.T
        np.testing.assert_allclose(
            spline(q),
            90 + 0.2 * x - 0.1 * y + 0.03 * x * x + 0.02 * x * y - 0.04 * y * y,
            atol=1e-7,
        )
        for nu, expected in [
            ((1, 0), 0.2 + 0.06 * x + 0.02 * y),
            ((0, 1), -0.1 + 0.02 * x - 0.08 * y),
            ((2, 0), 0.06),
            ((1, 1), 0.02),
            ((0, 2), -0.08),
        ]:
            np.testing.assert_allclose(spline(q, nu=nu), expected, atol=1e-7)
        for point in q:
            grad = np.array(
                [
                    0.2 + 0.06 * point[0] + 0.02 * point[1],
                    -0.1 + 0.02 * point[0] - 0.08 * point[1],
                ]
            )
            d = np.array([0.6, 0.8])
            expected = (d @ np.array([[0.06, 0.02], [0.02, -0.08]]) @ d) / (
                np.sqrt(1 + grad @ grad) * (1 + (grad @ d) ** 2)
            )
            self.assertAlmostEqual(
                normal_curvature(spline, point, d), expected, places=7
            )
        # Exact area averaged Hessian energy, independent of knot spacing.
        energy = np.linalg.norm(bending_matrix(t) @ spline.c.ravel()) ** 2
        self.assertAlmostEqual(energy, 0.06**2 + 2 * 0.02**2 + 0.08**2, places=8)

    def test_regularizer_preserves_plane_and_no_extrapolation(self):
        grid = np.linspace(-4, 4, 25)
        xy = np.array(np.meshgrid(grid, grid, indexing="ij")).reshape(2, -1).T
        z = 92 + 0.3 * xy[:, 0] - 0.2 * xy[:, 1]
        t = (knots([-4, 4], 2), knots([-4, 4], 2))
        spline, _ = fit(xy, z, t, 10)
        np.testing.assert_allclose(spline(xy), z, atol=1e-7)
        self.assertTrue(np.isnan(spline([4.01, 0])))
        self.assertAlmostEqual(normal_curvature(spline, [0, 0], [1, 0]), 0, places=7)


if __name__ == "__main__":
    unittest.main()
