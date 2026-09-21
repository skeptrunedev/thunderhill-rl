# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1", "pyproj==3.7.2", "shapely==2.1.2"]
# ///
import unittest
import numpy as np
from build_pavement_tone import (
    road_strip,
    tone_gains,
    pavement_samples,
    resample_pavement,
)


class PavementToneTests(unittest.TestCase):
    def test_paint_and_grass_do_not_bleed_into_pavement(self):
        for rejected in ([1, 1, 1], [0.55, 0.45, 0.30], [0.1, 0.4, 0.1], [0, 0, 0]):
            rgb = np.full((32, 16, 3), 0.35)
            rgb[8:24, 3:13] = rejected
            gains, mask = tone_gains(*pavement_samples(rgb))
            self.assertFalse(mask[16, 8])
            np.testing.assert_allclose(gains, 1, atol=1e-12)

    def test_rejection_precedes_bilinear_sampling(self):
        for rejected in ([1, 1, 1], [0.55, 0.45, 0.30]):
            rgb = np.full((8, 8, 3), 0.35)
            rgb[3, 3] = rejected
            row, col = np.meshgrid(np.linspace(2, 4, 16), np.linspace(2, 4, 16))
            numerator, weights = resample_pavement(rgb, row, col)
            gains, _ = tone_gains(numerator, weights)
            np.testing.assert_allclose(gains, 1, atol=1e-12)

    def test_filter_is_periodic_along_lap(self):
        rgb = np.full((64, 16, 3), 0.4)
        rgb[:8] = 0.25
        original, _ = tone_gains(*pavement_samples(rgb))
        shifted, _ = tone_gains(*pavement_samples(np.roll(rgb, 17, axis=0)))
        np.testing.assert_allclose(shifted, np.roll(original, 17, axis=0), atol=1e-12)
        self.assertLess(original[2, 8], original[30, 8])
        self.assertGreaterEqual(original.min(), 0.65)
        self.assertLessEqual(original.max(), 1.35)

    def test_transverse_and_station_pixel_centers(self):
        track = {
            "length_m": 40,
            "samples": [
                {"s": i * 10, "p": p, "width": 2}
                for i, p in enumerate([[0, 0, 0], [0, 0, 10], [10, 0, 10], [10, 0, 0]])
            ],
        }
        strip = road_strip(track, rows=4, columns=2)
        np.testing.assert_allclose(
            strip[0],
            [
                [-np.sqrt(0.125), 5 - np.sqrt(0.125)],
                [np.sqrt(0.125), 5 - np.sqrt(0.125)],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            road_strip(track, rows=4, columns=1)[0, 0], [0, 5 - np.sqrt(0.5)], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
