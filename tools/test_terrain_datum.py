# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "rasterio==1.4.4", "shapely==2.1.2", "pyproj==3.7.2"]
# ///
"""Check datum direction and pixel sampling against an analytic DEM ramp."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from build_horizon import sample_window
from build_surface_mesh import raw_terrain
from pyproj.enums import TransformDirection
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from terrain_datum import terrain_transform


class DatumTests(unittest.TestCase):
    def test_round_trip_and_nonidentity(self):
        transform, metadata = terrain_transform()
        e, n = 557466.3229144721, 4377233.643425343
        x, y = transform.transform(e, n, errcheck=True)
        self.assertGreater(np.hypot(x - e, y - n), 0.4)
        self.assertLess(np.hypot(x - e, y - n), 0.7)
        back = transform.transform(
            x, y, direction=TransformDirection.INVERSE, errcheck=True
        )
        np.testing.assert_allclose(back, [e, n], atol=1e-6, rtol=0)
        self.assertEqual(len(metadata["grid_sha256"]), 4)

    def test_raster_sampling_uses_transformed_pixel_centers(self):
        transform, _ = terrain_transform()
        e, n = 557466.3229144721, 4377233.643425343
        template = {
            "schema_version": 1,
            "nx": 3,
            "nz": 3,
            "step": 8,
            "x0": -8,
            "z0": -8,
        }
        origin = {"easting": e, "northing": n, "elevation_m": 90}
        affine = from_origin(e - 32, n + 32, 1, 1)
        cols, rows = np.meshgrid(np.arange(64) + 0.5, np.arange(64) + 0.5)
        ramp = 100 + 0.2 * (cols - 32) + 0.3 * (32 - rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ramp.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=64,
                height=64,
                count=1,
                dtype="float64",
                crs="EPSG:26910",
                transform=affine,
            ) as dst:
                dst.write(ramp, 1)
            result = raw_terrain({"origin": origin}, template, path)
            xx, zz = np.meshgrid([-8, 0, 8], [-8, 0, 8])
            east, north = transform.transform(e + xx, n - zz, errcheck=True)
            expected = 10 + 0.2 * (east - e) + 0.3 * (north - n)
            np.testing.assert_allclose(
                result["heights"], expected.ravel(), atol=0.000501, rtol=0
            )
            with self.assertRaisesRegex(ValueError, "leaves the source DEM"):
                raw_terrain({"origin": origin}, {**template, "x0": 1000}, path)


class HorizonSamplingTests(unittest.TestCase):
    def setUp(self):
        self.east_origin = 557466.3229144721
        self.north_origin = 4377233.643425343
        affine = from_origin(self.east_origin - 64, self.north_origin + 96, 2, 3)
        cols, rows = np.meshgrid(np.arange(64) + 0.5, np.arange(64) + 0.5)
        east = affine.c + cols * affine.a
        north = affine.f + rows * affine.e
        ramp = self.expected_height(east, north)
        memory = MemoryFile()
        self.addCleanup(memory.close)
        self.raster = memory.open(
            driver="GTiff",
            width=64,
            height=64,
            count=1,
            dtype="float64",
            crs="EPSG:26910",
            transform=affine,
        )
        self.addCleanup(self.raster.close)
        self.raster.write(ramp, 1)

    def expected_height(self, east, north):
        return (
            100
            + 0.2 * (east - self.east_origin)
            + 0.3 * (north - self.north_origin)
        )

    def test_transformed_local_grid_samples_analytic_ramp(self):
        transform, _ = terrain_transform()
        xx, zz = np.meshgrid([-16, 0, 16], [-16, 16])
        east, north = transform.transform(
            self.east_origin + xx, self.north_origin - zz, errcheck=True
        )
        heights, raw, bounds = sample_window(self.raster, east, north)
        self.assertEqual(heights.shape, xx.shape)
        self.assertIsInstance(raw, np.ma.MaskedArray)
        self.assertEqual(raw.ndim, 2)
        self.assertEqual(len(bounds), 4)
        self.assertLessEqual(bounds[0], east.min())
        self.assertLessEqual(bounds[1], north.min())
        self.assertGreaterEqual(bounds[2], east.max())
        self.assertGreaterEqual(bounds[3], north.max())
        np.testing.assert_allclose(
            heights, self.expected_height(east, north), atol=1e-8, rtol=0
        )

    def test_source_pixel_centers_define_sampling_limits(self):
        affine = self.raster.transform
        left = affine.c + 0.5 * affine.a
        right = affine.c + (self.raster.width - 0.5) * affine.a
        top = affine.f + 0.5 * affine.e
        bottom = affine.f + (self.raster.height - 0.5) * affine.e
        east, north = np.meshgrid([left, right], [bottom, top])
        heights, _, _ = sample_window(self.raster, east, north)
        np.testing.assert_allclose(
            heights, self.expected_height(east, north), atol=1e-8, rtol=0
        )
        # These points remain inside the raster's outer pixel bounds, but
        # interpolation would require extending its pixel center support.
        for edge, easting, northing in (
            ("left", left - 0.1, self.north_origin),
            ("right", right + 0.1, self.north_origin),
            ("bottom", self.east_origin, bottom - 0.1),
            ("top", self.east_origin, top + 0.1),
        ):
            with self.subTest(edge=edge):
                with self.assertRaisesRegex(ValueError, "leaves the source DEM"):
                    sample_window(
                        self.raster, np.array([[easting]]), np.array([[northing]])
                    )


if __name__ == "__main__":
    unittest.main()
