"""Independent analytic checks of footprint preservation and tessellation error."""

import unittest

import numpy as np
from build_atlas_road_mesh import build_mesh


class AnalyticSurface:
    def __init__(self, curvature=0):
        self.curvature = curvature

    def evaluate(self, points):
        x, z = np.asarray(points).T
        k = self.curvature
        return {
            "height": 3 + 0.2 * x - 0.3 * z + k * (x * x + z * z),
            "gradient": np.column_stack((0.2 + 2 * k * x, -0.3 + 2 * k * z)),
        }


def track_fixture():
    angle = np.arange(16) * 2 * np.pi / 16
    p = np.column_stack((20 * np.cos(angle), np.zeros(16), 20 * np.sin(angle)))
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    station = np.r_[0, np.cumsum(lengths)[:-1]]
    return {
        "length_m": float(sum(lengths)),
        "samples": [
            {
                "p": point.tolist(),
                "s": float(s),
                "width": float(3 + 0.2 * np.cos(a)),
                "bank": 0.04,
            }
            for point, s, a in zip(p, station, angle)
        ],
    }


class MeshTests(unittest.TestCase):
    def test_plane_footprint_normals_winding_and_seam(self):
        track = track_fixture()
        mesh = build_mesh(track, AnalyticSurface())
        v, n, uv, tris = [
            np.asarray(mesh[k]) for k in ("vertices", "normals", "uv", "triangles")
        ]
        np.testing.assert_allclose(
            v[:, 1], 3 + 0.2 * v[:, 0] - 0.3 * v[:, 2] + 0.04, atol=1e-13
        )
        normal = np.array([-0.2, 1, 0.3])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(n, np.tile(normal, (len(n), 1)), atol=1e-14)
        cross = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
        self.assertTrue(np.all(cross[:, 1] < 0))
        across = mesh["metadata"]["validation"]["across_intervals"] + 1
        np.testing.assert_array_equal(v[:across], v[-across:])
        np.testing.assert_array_equal(n[:across], n[-across:])
        np.testing.assert_array_equal(uv[:across, 0], uv[-across:, 0])
        np.testing.assert_array_equal(
            uv[-across:, 1], np.full(across, track["length_m"])
        )
        # Circle's centered planar left is exactly radial, independent of builder.
        edges = np.asarray(mesh["edge_vertices"])
        original = mesh["original_sample_rows"]
        for i, row in enumerate(track["samples"]):
            center = np.array(row["p"])[[0, 2]]
            radial = center / 20
            expected = np.array(
                [center - radial * row["width"] / 2, center + radial * row["width"] / 2]
            )
            np.testing.assert_allclose(
                edges[original[i]][:, [0, 2]], expected, atol=1e-13
            )
            end = original[i + 1] if i + 1 < len(original) else len(edges) - 1
            for j in range(original[i], end + 1):
                t = (j - original[i]) / (end - original[i])
                np.testing.assert_allclose(
                    edges[j][:, [0, 2]],
                    (1 - t) * edges[original[i]][:, [0, 2]] + t * edges[end][:, [0, 2]],
                    atol=1e-13,
                )
        self.assertLess(
            mesh["metadata"]["validation"]["sampled_vertical_error_m"]["max"], 1e-13
        )

    def test_quadratic_error_and_refinement(self):
        track = track_fixture()
        errors = []
        for spacing in [1, 0.5]:
            mesh = build_mesh(track, AnalyticSurface(0.02), spacing)
            v, tris = np.asarray(mesh["vertices"]), np.asarray(mesh["triangles"])
            p = v[tris][:, :, [0, 2]]
            # For k*|p|², centroid plane error is k/9 times sum of squared side lengths.
            centroid_error = (
                0.02 / 9 * np.sum((p - np.roll(p, 1, axis=1)) ** 2, axis=(1, 2))
            )
            edge_error = 0.02 / 4 * np.sum((p - np.roll(p, 1, axis=1)) ** 2, axis=2)
            expected_max = max(centroid_error.max(), edge_error.max())
            actual = mesh["metadata"]["validation"]["sampled_vertical_error_m"]["max"]
            self.assertAlmostEqual(actual, expected_max, places=12)
            errors.append(actual)
        self.assertLess(errors[1], errors[0] * 0.35)

    def test_invalid_inputs(self):
        for spacing in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                build_mesh(track_fixture(), AnalyticSurface(), spacing)
        track = track_fixture()
        track["samples"][1]["s"] = 0
        with self.assertRaises(ValueError):
            build_mesh(track, AnalyticSurface())


if __name__ == "__main__":
    unittest.main()
