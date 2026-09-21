# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "rasterio==1.4.4", "shapely==2.1.2"]
# ///
"""Protect source identity and sample correspondence when applying plan studies."""

import copy
import hashlib
import json
import unittest

import numpy as np
from build_track import apply_plan_study


class PlanStudyApplicationTests(unittest.TestCase):
    def fixture(self):
        track = {
            "length_m": 120,
            "samples": [
                {"s": s, "p": p}
                for s, p in zip(
                    [0, 30, 60, 90],
                    [[0, 7, 0], [30, 8, 0], [30, 9, 30], [0, 6, 30]],
                )
            ],
        }
        encoded = json.dumps(track, separators=(",", ":")) + "\n"
        study = {
            "track_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "station_interval_m": [30, 60],
            "candidate": {
                "stations_m": [30, 60],
                "candidate_xz_m": [[29.5, 0], [29.5, 30]],
                "maximum_center_displacement_m": 0.5,
            },
        }
        return track, study

    def test_apply_only_selected_horizontal_samples(self):
        track, study = self.fixture()
        original = copy.deepcopy(track)
        displacement, metadata = apply_plan_study(track, study)
        np.testing.assert_array_equal(displacement, [[0, 0], [-0.5, 0], [-0.5, 0], [0, 0]])
        self.assertEqual(track, original)
        self.assertEqual(metadata["changed_samples"], 2)
        self.assertEqual(metadata["baseline_length_m"], 120)

    def test_stale_source_or_station_mapping_rejected(self):
        track, study = self.fixture()
        track["samples"][1]["p"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "baseline track changed"):
            apply_plan_study(track, study)
        track, study = self.fixture()
        study["candidate"]["stations_m"].reverse()
        with self.assertRaisesRegex(ValueError, "stations differ"):
            apply_plan_study(track, study)

    def test_nonfinite_or_changed_displacement_rejected(self):
        track, study = self.fixture()
        study["candidate"]["candidate_xz_m"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "coordinates"):
            apply_plan_study(track, study)
        track, study = self.fixture()
        study["candidate"]["candidate_xz_m"][0][0] = 29
        with self.assertRaisesRegex(ValueError, "displacement differs"):
            apply_plan_study(track, study)


if __name__ == "__main__":
    unittest.main()
