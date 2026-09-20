# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1"]
# ///
"""Behavior contracts for aerial terrain color filtering."""
import unittest

import numpy as np
from build_terrain_color import dry_terrain_mask, terrain_gains


class TerrainColorTests(unittest.TestCase):
    def test_rejects_pavement_shadows_trees_roofs(self):
        samples = np.array([[[.55, .48, .37], [.35, .35, .35], [.10, .08, .05],
                             [.20, .35, .15], [.93, .92, .89]]])
        np.testing.assert_array_equal(dry_terrain_mask(samples), [[True, False, False, False, False]])

    def test_rejected_colors_cannot_dark_or_color_bleed(self):
        base = np.full((80, 80, 3), [.55, .48, .37])
        for rejected in ([0, 0, 0], [.4, .4, .4], [0, .8, 0], [1, 1, 1]):
            altered = base.copy()
            altered[10:70, 10:70] = rejected
            gains, mask, _ = terrain_gains(altered, (1, 1))
            self.assertFalse(mask[40, 40])
            np.testing.assert_allclose(gains, 1, atol=1e-12)

    def test_retains_broad_variation_with_bounded_gains(self):
        image = np.full((120, 160, 3), [.55, .48, .37])
        image[:, :80] *= .7
        gains, _, _ = terrain_gains(image, (1, 1))
        self.assertLess(gains[60, 20, 0], gains[60, 140, 0])
        self.assertGreaterEqual(gains.min(), .6)
        self.assertLessEqual(gains.max(), 1.4)
        self.assertLess(np.abs(np.diff(gains[60, :, 0])).max(), .08)

    def test_all_rejected_source_fails(self):
        with self.assertRaisesRegex(ValueError, "No accepted"):
            terrain_gains(np.zeros((20, 20, 3)), (1, 1))


if __name__ == "__main__":
    unittest.main()
