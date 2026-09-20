# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1", "pyproj==3.7.2"]
# ///
"""Behavior contracts for aerial terrain color filtering."""

import unittest

import numpy as np
from build_terrain_color import (
    dry_terrain_mask,
    local_raster_bounds,
    resample_local_gains,
    terrain_gains,
)
from pyproj import Transformer


class TerrainColorTests(unittest.TestCase):
    def test_rejects_pavement_shadows_trees_roofs(self):
        samples = np.array(
            [
                [
                    [0.55, 0.48, 0.37],
                    [0.35, 0.35, 0.35],
                    [0.10, 0.08, 0.05],
                    [0.20, 0.35, 0.15],
                    [0.93, 0.92, 0.89],
                ]
            ]
        )
        np.testing.assert_array_equal(
            dry_terrain_mask(samples), [[True, False, False, False, False]]
        )

    def test_rejected_colors_cannot_dark_or_color_bleed(self):
        base = np.full((80, 80, 3), [0.55, 0.48, 0.37])
        for rejected in ([0, 0, 0], [0.4, 0.4, 0.4], [0, 0.8, 0], [1, 1, 1]):
            altered = base.copy()
            altered[10:70, 10:70] = rejected
            gains, mask, _ = terrain_gains(altered, (1, 1))
            self.assertFalse(mask[40, 40])
            np.testing.assert_allclose(gains, 1, atol=1e-12)

    def test_retains_broad_variation_with_bounded_gains(self):
        image = np.full((120, 160, 3), [0.55, 0.48, 0.37])
        image[:, :80] *= 0.7
        gains, _, _ = terrain_gains(image, (1, 1))
        self.assertLess(gains[60, 20, 0], gains[60, 140, 0])
        self.assertGreaterEqual(gains.min(), 0.6)
        self.assertLessEqual(gains.max(), 1.4)
        self.assertLess(np.abs(np.diff(gains[60, :, 0])).max(), 0.08)

    def test_transformed_linear_field_pixel_centers_and_axis_direction(self):
        # Affine scaling/shear ensures a translation-only implementation fails.
        transform = Transformer.from_pipeline(
            "+proj=affine +s11=1.01 +s12=0.02 +s21=-0.01 +s22=0.99 +xoff=2 +yoff=-3"
        )
        extent = {"xmin": 0, "xmax": 100, "ymin": 0, "ymax": 100}
        origin = {"easting": 10, "northing": 90}
        e, n = np.meshgrid(np.arange(100) + 0.5, 99.5 - np.arange(100))
        scalar = 0.7 + e * 0.002 + n * 0.001
        source = np.repeat(scalar[:, :, None], 3, axis=2)
        lower, size = np.array([10.0, 10.0]), np.array([40.0, 40.0])
        result = resample_local_gains(
            source, extent, origin, transform, lower, size, (10, 10)
        )
        x, z = np.meshgrid(12 + 4 * np.arange(10), 12 + 4 * np.arange(10))
        east, north = transform.transform(x + 10, 90 - z)
        expected = np.repeat(
            (0.7 + east * 0.002 + north * 0.001)[:, :, None], 3, axis=2
        )
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_inverse_bounds_and_exterior_neutral(self):
        transform = Transformer.from_pipeline("+proj=affine +xoff=2 +yoff=-3")
        extent = {"xmin": 0, "xmax": 100, "ymin": 0, "ymax": 100}
        origin = {"easting": 10, "northing": 90}
        lower, size = local_raster_bounds(extent, origin, transform)
        np.testing.assert_allclose(lower, [-12, -13])
        np.testing.assert_allclose(size, [100, 100])
        outside = resample_local_gains(
            np.full((100, 100, 3), 0.7),
            extent,
            origin,
            transform,
            [-30, -30],
            [4, 4],
            (1, 1),
        )
        np.testing.assert_array_equal(outside, np.ones((1, 1, 3)))

    def test_all_rejected_source_fails(self):
        with self.assertRaisesRegex(ValueError, "No accepted"):
            terrain_gains(np.zeros((20, 20, 3)), (1, 1))


if __name__ == "__main__":
    unittest.main()
