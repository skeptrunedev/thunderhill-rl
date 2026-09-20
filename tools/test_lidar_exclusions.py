"""Exclusions must reach full fitting supports, including boundary contacts."""

import unittest

from refit_lidar_exclusions import affected_patches
from shapely.geometry import box


class ExclusionSupportTests(unittest.TestCase):
    def test_full_fit_square_not_blending_radius(self):
        patches = [{"origin": [0, 0]}, {"origin": [36, 0]}, {"origin": [0, 36]}]
        # At x15 this is beyond the12m blending radius but inside the16m fit.
        self.assertEqual(affected_patches(patches, box(15, -1, 16, 1)), [0])

    def test_touching_square_boundary_and_disjoint(self):
        patches = [{"origin": [0, 0]}, {"origin": [40, 0]}]
        self.assertEqual(affected_patches(patches, box(16, -1, 17, 1)), [0])
        self.assertEqual(affected_patches(patches, box(17, -1, 18, 1)), [])


if __name__ == "__main__":
    unittest.main()
