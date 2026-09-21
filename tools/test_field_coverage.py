# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1", "pyproj==3.7.2", "shapely==2.1.2"]
# ///
"""Field annotation contracts for datum mapping and regional blending."""

import unittest

import numpy as np
import shapely
from build_field_coverage import local_polygon, rasterize
from pyproj import Transformer


class FieldCoverageTests(unittest.TestCase):
    def test_pixel_centers_inverse_projection_and_south_axis(self):
        transform = Transformer.from_pipeline(
            "+proj=affine +s11=2 +s22=3 +xoff=10 +yoff=20"
        )
        polygon = local_polygon(
            [[0, 0], [2, 0], [2, 2], [0, 2]],
            {"xmin": 10, "xmax": 30, "ymin": 20, "ymax": 50},
            [10, 10],
            {"easting": 0, "northing": 10},
            transform,
        )
        np.testing.assert_allclose(polygon.bounds, [0.5, 0.5, 2.5, 2.5])
        self.assertAlmostEqual(polygon.area, 4)

    def test_interior_feather_endpoints_and_outside_neutral(self):
        values = rasterize(
            [(shapely.box(0, 0, 10, 10), 0.95, 0.8, 3)], [-1.5, 4.5], [6, 1], [6, 1]
        )[0]
        np.testing.assert_array_equal(values[:2], 0)
        np.testing.assert_allclose(values[2:, :3], np.tile([0.95, 0.8, 0], (4, 1)))
        np.testing.assert_allclose(values[:, 3], [0, 0, 7 / 27, 20 / 27, 1, 1])

    def test_overlaid_region_feather_preserves_underlying_material(self):
        values = rasterize(
            [
                (shapely.box(-10, -10, 10, 10), 0.9, 0.8, 1),
                (shapely.box(0, -10, 10, 10), 0.5, 0.2, 2),
            ],
            [-1.5, -0.5],
            [4, 1],
            [4, 1],
        )[0]
        np.testing.assert_allclose(values[:, 0], [0.9, 0.9, 0.7, 0.5])
        np.testing.assert_allclose(values[:, 1], [0.8, 0.8, 0.5, 0.2])
        np.testing.assert_allclose(values[:, 3], 1)

    def test_invalid_polygon_and_parameters_fail(self):
        extent = {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10}
        origin = {"easting": 0, "northing": 0}
        transform = Transformer.from_pipeline("+proj=affine")
        for polygon in (
            [[0, 0], [3, 3], [0, 3], [3, 0]],
            [[0, 0], [1, float("nan")], [2, 0]],
            [[0, 0], [10, 1], [2, 0]],
        ):
            with self.assertRaises(ValueError):
                local_polygon(polygon, extent, [10, 10], origin, transform)
        for parameters in ((1.1, 0.8, 3), (0.9, float("nan"), 3), (0.9, 0.8, 0)):
            with self.assertRaises(ValueError):
                rasterize(
                    [(shapely.box(0, 0, 10, 10), *parameters)], [0, 0], [2, 2], [2, 2]
                )


if __name__ == "__main__":
    unittest.main()
