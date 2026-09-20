# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3"]
# ///
"""Independent displacement and wrapped interval contracts."""

import hashlib
import json
import unittest

import numpy as np
from apply_pavement_envelope import apply_envelope


def fixture():
    angle = np.arange(16) * 2 * np.pi / 16
    track = {
        "length_m": 160,
        "samples": [
            {"s": i * 10, "p": [20 * np.cos(a), 0, 20 * np.sin(a)], "width": 4}
            for i, a in enumerate(angle)
        ],
    }
    manifest = {
        "baseline_track_sha256": hashlib.sha256(
            (json.dumps(track, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "coverage": {"start_station_m": 140, "end_station_m": 20},
        "anchors": [],
    }
    # Baseline edges of this counterclockwise circle have radial normals.
    for i in [14, 0, 2]:
        radial = np.array([np.cos(angle[i]), np.sin(angle[i])])
        shift = np.array([1.0, 2.0]) if i == 0 else np.zeros(2)
        manifest["anchors"].append(
            {
                "station_m": i * 10,
                "right_xz_m": (18 * radial + shift).tolist(),
                "left_xz_m": (22 * radial + shift).tolist(),
            }
        )
    return track, manifest


class EnvelopeTests(unittest.TestCase):
    def test_wrap_preserves_curvature_outside_and_interpolates_displacement(self):
        track, manifest = fixture()
        p, widths, report = apply_envelope(track, manifest)
        baseline = np.array([s["p"] for s in track["samples"]])[:, [0, 2]]
        expected = baseline.copy()
        expected[0] += [1, 2]
        expected[[15, 1]] += [0.5, 1]
        np.testing.assert_allclose(p, expected, atol=1e-14)
        np.testing.assert_allclose(widths, 4, atol=1e-14)
        self.assertEqual(report["changed_samples"], 3)

    def test_wrong_baseline_and_unjoined_or_duplicate_endpoints_rejected(self):
        track, manifest = fixture()
        manifest["baseline_track_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "baseline changed"):
            apply_envelope(track, manifest)
        track, manifest = fixture()
        manifest["anchors"][0]["left_xz_m"][0] += 1
        with self.assertRaisesRegex(ValueError, "endpoints must meet"):
            apply_envelope(track, manifest)
        track, manifest = fixture()
        manifest["anchors"].append(manifest["anchors"][0])
        with self.assertRaisesRegex(ValueError, "uniquely span"):
            apply_envelope(track, manifest)


if __name__ == "__main__":
    unittest.main()
