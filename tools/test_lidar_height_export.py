"""Verify source frame reflection using an independently known polynomial."""

import unittest
import numpy as np
from scipy.interpolate import NdBSpline
from fit_cyclone_surface import fit, knots
from export_lidar_height_surface import convert, fixture


def nonuniform_fixture():
    """Seeded control lattice exercises unequal knot intervals and all endpoints."""
    edges = [
        np.array([-4.0, -2.0, -0.125, 0.75, 3.0, 6.0]),
        np.array([-7.0, -3.0, -0.375, 1.5, 3.0]),
    ]
    t = [np.r_[np.repeat(a[0], 3), a, np.repeat(a[-1], 3)] for a in edges]
    candidate = {
        "degree": [3, 3],
        "knots": [a.tolist() for a in t],
        "coefficients_navd88_m": np.random.default_rng(62)
        .uniform(89, 95, (len(t[0]) - 4, len(t[1]) - 4))
        .tolist(),
    }
    origin = {"easting": 0.0, "northing": 0.0, "elevation_m": 90.0}
    surface = convert(candidate, [0.0, 0.0], origin)
    return fixture(candidate, [0.0, 0.0], origin, surface)


class ExportTests(unittest.TestCase):
    def test_north_reflection_height_gradient_and_mixed_curvature(self):
        x = np.linspace(-4, 6, 31)
        n = np.linspace(-7, 3, 31)
        q = np.array(np.meshgrid(x, n, indexing="ij")).reshape(2, -1).T
        xx, nn = q.T
        heights = (
            95 + 0.2 * xx - 0.3 * nn + 0.04 * xx * xx + 0.07 * xx * nn + 0.02 * nn * nn
        )
        spline, _ = fit(q, heights, (knots([-4, 6], 2), knots([-7, 3], 2)), 0)
        candidate = {
            "degree": [3, 3],
            "knots": [t.tolist() for t in spline.t],
            "coefficients_navd88_m": spline.c.tolist(),
        }
        origin = {"easting": 500000.0, "northing": 4000000.0, "elevation_m": 90.0}
        surface = convert(candidate, [500013.0, 3999979.0], origin)
        converted = NdBSpline(
            tuple(np.array(t) for t in surface["knots"]),
            np.array(surface["coefficients"]),
            3,
            extrapolate=False,
        )
        self.assertEqual(surface["origin"], [13.0, 21.0])
        for x, z in [[0, 0], [-4, -3], [6, 7], [2.3, -1.2]]:
            self.assertAlmostEqual(
                float(converted([x, z])),
                5 + 0.2 * x + 0.3 * z + 0.04 * x * x - 0.07 * x * z + 0.02 * z * z,
                places=7,
            )
            self.assertAlmostEqual(
                float(converted([x, z], nu=(1, 0))), 0.2 + 0.08 * x - 0.07 * z, places=7
            )
            self.assertAlmostEqual(
                float(converted([x, z], nu=(0, 1))), 0.3 - 0.07 * x + 0.04 * z, places=7
            )
            self.assertAlmostEqual(float(converted([x, z], nu=(1, 1))), -0.07, places=7)
        data = fixture(candidate, [500013.0, 3999979.0], origin, surface)
        for s in data["samples"]:
            p = [s["x"] - 13, s["z"] - 21]
            self.assertAlmostEqual(float(converted(p)), s["height"], places=10)
            for nu, expected in zip(
                [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)], s["gradient"] + s["hessian"]
            ):
                self.assertAlmostEqual(float(converted(p, nu=nu)), expected, places=10)


if __name__ == "__main__":
    unittest.main()
