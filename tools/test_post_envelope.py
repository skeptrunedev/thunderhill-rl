# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "rasterio==1.4.4", "shapely==2.1.2"]
# ///
"""Protect precision, baseline pins and separate final edge diagnostics."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from build_track import apply_post_envelope, final_envelope_residuals
from test_pavement_envelope import fixture


class PostEnvelopeTests(unittest.TestCase):
    def test_preserves_unrounded_points_and_refits_without_mutating_baseline(self):
        track, envelope = fixture()
        original = copy.deepcopy(track)
        local = np.array([s["p"] for s in track["samples"]])[:, [0, 2]]
        unrounded = local * [1, -1] + [557000.0000123, 4377000.0000456]
        origin = np.array([557000, 4377000, 90])
        source_stations = np.arange(len(local)) * 9.1234
        expected = unrounded.copy()
        expected[0] += [1, -2]
        expected[[15, 1]] += [0.5, -1]
        # The mock isolates stage ordering and coordinates from the lidar source.
        with patch("build_track.geometry_samples", return_value=(track["samples"],)) as refit:
            points, widths, geometry, metadata = apply_post_envelope(
                track, envelope, unrounded, origin, source_stations
            )
        np.testing.assert_array_equal(points, expected)
        np.testing.assert_array_equal(refit.call_args.args[0], expected)
        np.testing.assert_array_equal(refit.call_args.args[1], widths)
        np.testing.assert_array_equal(refit.call_args.args[2], origin)
        np.testing.assert_array_equal(refit.call_args.args[3], source_stations)
        self.assertEqual(track, original)
        self.assertEqual(metadata["baseline_track_sha256"], envelope["baseline_track_sha256"])
        self.assertEqual(metadata["changed_samples"], 3)
        self.assertEqual(geometry[0], track["samples"])

    def test_stale_pin_rejected_before_refit(self):
        track, envelope = fixture()
        track["metadata"] = {"new_plan": True}
        with patch("build_track.geometry_samples") as refit:
            with self.assertRaisesRegex(ValueError, "baseline changed"):
                apply_post_envelope(track, envelope, None, None, None)
            refit.assert_not_called()

    def test_final_edges_use_baseline_identity_and_report_each_side(self):
        track, envelope = fixture()
        samples = copy.deepcopy(track["samples"])
        # Unchanged circular edges have an independently known radial position.
        for anchor in envelope["anchors"]:
            angle = anchor["station_m"] / 160 * 2 * np.pi
            radial = np.array([np.cos(angle), np.sin(angle)])
            anchor["right_xz_m"] = (18 * radial).tolist()
            anchor["left_xz_m"] = (22 * radial).tolist()
        envelope["anchors"][1]["right_xz_m"][0] += 0.3
        envelope["anchors"][1]["left_xz_m"][0] += 0.8
        for sample in samples:
            sample["s"] *= 1.2  # Final station labels cannot reinterpret anchors.
        report = final_envelope_residuals(track, samples, envelope)
        self.assertAlmostEqual(report["right"]["maximum_horizontal_residual_m"], 0.3)
        self.assertAlmostEqual(report["left"]["maximum_horizontal_residual_m"], 0.8)
        self.assertAlmostEqual(report["right"]["rms_horizontal_residual_m"], 0.3 / np.sqrt(3))
        self.assertAlmostEqual(report["left"]["rms_horizontal_residual_m"], 0.8 / np.sqrt(3))


if __name__ == "__main__":
    unittest.main()
