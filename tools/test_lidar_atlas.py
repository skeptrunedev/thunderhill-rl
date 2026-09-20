"""Independent analytic and derivative checks for overlapping surface patches."""

import unittest

import numpy as np
from build_lidar_atlas import HeightAtlas, serialized_patch, weight_axis
from fit_cyclone_surface import knots
from scipy.interpolate import NdBSpline


def plane_patch(center, offset=0.0):
    t = knots([-16, 16], 2)
    g = np.array([np.mean(t[i + 1 : i + 4]) for i in range(len(t) - 4)])
    x, z = np.meshgrid(g + center[0], g + center[1], indexing="ij")
    c = 5 + 0.2 * x - 0.3 * z + offset
    return serialized_patch(
        NdBSpline((t, t), c, 3, extrapolate=False), np.array(center)
    )


def parity_fixture():
    data = {
        "schema_version": 1,
        "support_radius_m": 12.0,
        "patches": [
            plane_patch([-12.0, -12.0], 0.2),
            plane_patch([0.0, -12.0], -0.1),
            plane_patch([-12.0, 0.0], 0.05),
            plane_patch([0.0, 0.0], -0.2),
        ],
    }
    points = (
        np.array(
            np.meshgrid(np.linspace(-18, 6, 17), np.linspace(-18, 6, 17), indexing="ij")
        )
        .reshape(2, -1)
        .T
    )
    # Include both sides of internal support boundaries and spatial hash cells.
    seam = np.array(
        [
            [q, z]
            for q in [-12 - 1e-6, -12.0, -12 + 1e-6, -1e-6, 0.0, 1e-6]
            for z in [-6.0, 0.0, 3.0]
        ]
    )
    points = np.vstack([points, seam])
    result = HeightAtlas(data).evaluate(points)
    return {
        "atlas": data,
        "samples": [
            {
                "x": float(p[0]),
                "z": float(p[1]),
                "height": float(result["height"][i]),
                "gradient": result["gradient"][i].tolist(),
                "hessian": result["hessian"][i].tolist(),
            }
            for i, p in enumerate(points)
        ],
    }


class AtlasTests(unittest.TestCase):
    def test_shared_plane_and_vanishing_weight_derivatives(self):
        data = {
            "support_radius_m": 12.0,
            "patches": [plane_patch([0.0, 0.0]), plane_patch([12.0, 0.0])],
        }
        p = np.array(
            [[x, 1.2] for x in [-0.01, 0.0, 0.01, 3.0, 11.99, 12.0, 12.01, 20.0]]
        )
        r = HeightAtlas(data).evaluate(p)
        np.testing.assert_allclose(
            r["height"], 5 + 0.2 * p[:, 0] - 0.3 * p[:, 1], atol=1e-12
        )
        np.testing.assert_allclose(
            r["gradient"], np.tile([0.2, -0.3], (len(p), 1)), atol=1e-12
        )
        np.testing.assert_allclose(r["hessian"], 0, atol=1e-12)
        for q in [-13.0, -12.0, 12.0, 13.0]:
            np.testing.assert_equal(weight_axis(np.array([q]), 12.0), np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            HeightAtlas(data).evaluate([[24, 0]])

    def test_full_quotient_derivatives_and_support_seams(self):
        a = HeightAtlas(parity_fixture()["atlas"])
        h = 1e-5
        for p in np.array(
            [[-12.0, -6.0], [-6.0, -6.0], [0.0, -6.0], [-6.0, 0.0], [-3.5, -7.25]]
        ):
            r = a.evaluate([p])
            numeric_hessian = np.zeros((2, 2))
            for axis in range(2):
                d = np.eye(2)[axis] * h
                plus = a.evaluate([p + d])
                minus = a.evaluate([p - d])
                self.assertAlmostEqual(
                    float((plus["height"][0] - minus["height"][0]) / (2 * h)),
                    r["gradient"][0, axis],
                    places=7,
                )
                numeric_hessian[:, axis] = (
                    plus["gradient"][0] - minus["gradient"][0]
                ) / (2 * h)
            expected = np.array(
                [
                    [r["hessian"][0, 0], r["hessian"][0, 1]],
                    [r["hessian"][0, 1], r["hessian"][0, 2]],
                ]
            )
            np.testing.assert_allclose(numeric_hessian, expected, atol=1e-7)
            before = a.evaluate([p - [1e-8, 0]])
            after = a.evaluate([p + [1e-8, 0]])
            for key in ["height", "gradient", "hessian"]:
                np.testing.assert_allclose(before[key], after[key], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
