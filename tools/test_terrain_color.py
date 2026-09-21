# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1", "pyproj==3.7.2", "shapely==2.1.2"]
# ///
"""Behavior contracts for aerial terrain color filtering."""

import unittest

import numpy as np
from build_terrain_color import (
    dry_terrain_mask,
    local_raster_bounds,
    resample_local_gains,
    shoulder_coverage,
    terrain_detail_gains,
    terrain_gains,
    validate_detail_settings,
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

    def test_detail_rejects_color_bleed_and_retains_coherent_variation(self):
        base = np.full((80, 80, 3), [0.55, 0.48, 0.37])
        for rejected in ([0, 0, 0], [0.4, 0.4, 0.4], [0, 0.8, 0], [1, 1, 1]):
            altered = base.copy()
            altered[10:70, 10:70] = rejected
            np.testing.assert_allclose(terrain_detail_gains(altered, (1, 1)), 1, atol=1e-12)
        base[:, 35:40] *= 0.88
        detail = terrain_detail_gains(base, (1, 1))
        self.assertLess(detail[40, 37, 0], 0.9)
        self.assertGreater(detail[40, 30, 0], 1.01)
        self.assertGreaterEqual(detail.min(), 0.75)
        self.assertLessEqual(detail.max(), 1.25)

    def test_detail_survives_bright_and_dark_macro_saturation(self):
        # The majority neutral field anchors the median. Far inside each other
        # field, its broad and fine absolute gains both exceed the same macro
        # limit. A small stripe must still retain relative local contrast.
        for field_gain, stripe_gain, direction in (
            (1.3, 0.95, -1),
            (0.65, 1.05, 1),
        ):
            with self.subTest(field_gain=field_gain):
                image = np.full((160, 400, 3), [0.55, 0.48, 0.37])
                image[:, 240:] *= field_gain
                image[:, 310:315] *= stripe_gain
                self.assertTrue(dry_terrain_mask(image).all())
                macro, _, _ = terrain_gains(image, (1, 1))
                limit = 1.4 if field_gain > 1 else 0.6
                np.testing.assert_allclose(macro[80, 312], limit)
                detail = terrain_detail_gains(image, (1, 1))
                self.assertTrue(np.all(direction * (detail[80, 312] - 1) > 0.05))
                np.testing.assert_allclose(detail[80, 370], 1, atol=1e-12)
                self.assertGreaterEqual(detail.min(), 0.75)
                self.assertLessEqual(detail.max(), 1.25)

    def test_fine_detail_preserves_narrow_stripes_without_rejected_color_bleed(self):
        # Two source pixels (1.2 m) represent a narrow historical field streak.
        # Compare contrast after the complete output pixel area integration.
        image = np.full((80, 80, 3), [0.55, 0.48, 0.37])
        image[:, 39:41] *= 0.94
        transform = Transformer.from_pipeline("+proj=affine")
        extent = {"xmin": 0, "xmax": 48, "ymin": 0, "ymax": 48}
        origin = {"easting": 0, "northing": 48}
        contrasts = []
        for sigma, pixel in ((0.6, 1.0), (0.2, 0.6)):
            detail = terrain_detail_gains(image, (0.6, 0.6), sigma)
            sampled = resample_local_gains(
                detail, extent, origin, transform, [0, 0], [48, 48],
                [round(48 / pixel)] * 2,
            )
            center_row = sampled[sampled.shape[0] // 2, :, 0]
            contrasts.append(1 - center_row.min())
            for rejected in ([0, 0, 0], [0.4, 0.4, 0.4], [0, 0.8, 0], [1, 1, 1]):
                uniform = np.full_like(image, [0.55, 0.48, 0.37])
                uniform[10:70, 10:70] = rejected
                np.testing.assert_allclose(
                    terrain_detail_gains(uniform, (0.6, 0.6), sigma), 1, atol=1e-12
                )
        self.assertGreater(contrasts[1], contrasts[0] * 1.4)

    def test_detail_settings_reject_invalid_bounds(self):
        for invalid in (float("nan"), float("inf"), -1, 0, 0.09, 2.01):
            with self.subTest(sigma=invalid), self.assertRaises(ValueError):
                validate_detail_settings(invalid, 1.0)
        for invalid in (float("nan"), float("inf"), -1, 0, 0.49, 2.01):
            with self.subTest(pixel=invalid), self.assertRaises(ValueError):
                validate_detail_settings(0.6, invalid)
        for sigma in (0.1, 2.0):
            for pixel in (0.5, 2.0):
                validate_detail_settings(sigma, pixel)

    def test_shoulder_distance_and_pixel_center_mapping(self):
        # Rectangle right edge is x=0, well away from the other three edges.
        ring = [[-20, -20], [0, -20], [0, 20], [-20, 20]]
        actual = shoulder_coverage([ring], [0, -0.5], [5, 1], [5, 1])
        np.testing.assert_allclose(actual, [[0, 7 / 27, 20 / 27, 1, 1]], atol=1e-14)
        # Nearest corner, distance sqrt(2), checks segments rather than infinite lines.
        corner = shoulder_coverage([ring], [0.5, 20.5], [1, 1], [1, 1])
        t = (np.sqrt(2) - 0.5) / 3
        np.testing.assert_allclose(corner, [[t * t * (3 - 2 * t)]])

    def test_all_rejected_source_fails(self):
        with self.assertRaisesRegex(ValueError, "No accepted"):
            terrain_gains(np.zeros((20, 20, 3)), (1, 1))


if __name__ == "__main__":
    unittest.main()
