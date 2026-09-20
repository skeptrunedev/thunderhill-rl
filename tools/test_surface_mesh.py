# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "rasterio==1.4.4", "shapely==2.1.2", "pyproj==3.7.2"]
# ///
"""Independent geometry contracts: uv run tools/test_surface_mesh.py."""
import unittest
import numpy as np
import shapely
from build_surface_mesh import build_mesh, clip_triangles, segment_heights, terrain_heights


class SurfaceGeometryTests(unittest.TestCase):
    def check_partition(self, triangle, road):
        output = clip_triangles(np.array([triangle], dtype=float), road)
        pieces = shapely.polygons(output)
        union = shapely.union_all(pieces)
        expected = shapely.Polygon(triangle).difference(road)
        self.assertLess(union.symmetric_difference(expected).area, 1e-10)
        self.assertLess(abs(shapely.area(pieces).sum() - union.area), 1e-10)
        self.assertLess(union.intersection(road).area, 1e-10)
        return union

    def test_road_hole_inside_triangle_is_not_filled(self):
        road = shapely.box(1, 1, 2, 2)
        surface = self.check_partition([[0, 0], [10, 0], [0, 10]], road)
        self.assertFalse(surface.covers(shapely.Point(1.5, 1.5)))

    def test_road_with_inner_island_preserves_island(self):
        road = shapely.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)],
                               holes=[[(2, 2), (3, 2), (3, 3), (2, 3)]])
        surface = self.check_partition([[0, 0], [12, 0], [0, 12]], road)
        self.assertTrue(surface.covers(shapely.Point(2.5, 2.5)))
        self.assertFalse(surface.covers(shapely.Point(1.5, 1.5)))

    def test_crossing_road_splits_triangle(self):
        self.check_partition([[0, 0], [10, 0], [0, 10]], shapely.box(2, -1, 3, 11))

    def test_touching_boundary_preserves_area(self):
        self.check_partition([[0, 0], [10, 0], [0, 10]], shapely.box(-2, -2, 0, 0))

    def test_transition_partition_preserves_exterior_dem_planes(self):
        road = shapely.box(-2, -2, 2, 2)
        coords = np.asarray(road.exterior.coords)
        starts = np.column_stack((coords[:-1, 0], np.full(4, 3.), coords[:-1, 1]))
        ends = np.roll(starts, -1, axis=0)
        xx, zz = np.meshgrid(np.arange(-16, 17, 8), np.arange(-16, 17, 8))
        terrain = {'nx': 5, 'nz': 5, 'x0': -16, 'z0': -16, 'step': 8,
                   'heights': (0.1*xx+0.002*xx*zz).ravel().tolist()}
        vertices, triangles, _ = build_mesh(terrain, road, starts, ends)
        vertices, triangles = np.asarray(vertices), np.asarray(triangles)
        envelope = road.buffer(6, quad_segs=32)
        outside_count = 0
        for triangle in vertices[triangles]:
            polygon = shapely.Polygon(triangle[:, [0, 2]])
            exterior_area = polygon.difference(envelope).area
            if exterior_area < 1e-9:
                continue
            self.assertLess(abs(exterior_area - polygon.area), 1e-8,
                            'A triangle straddles the transition boundary')
            outside_count += 1
            for weights in ([1/3]*3, [.1, .2, .7]):
                point = np.asarray(weights) @ triangle
                expected = terrain_heights(point[None, [0, 2]], terrain)[0]
                self.assertAlmostEqual(point[1], expected, places=8)
        self.assertGreater(outside_count, 0)

    def test_nearest_segment_interpolates_slope_and_clamps(self):
        heights, distance = segment_heights(np.array([[2, 3], [-2, 0], [12, 0]]),
                                            np.array([[0, 5, 0]]), np.array([[10, 25, 0]]))
        np.testing.assert_allclose(heights, [9, 5, 25])
        np.testing.assert_allclose(distance, [3, 2, 2])

    def test_dem_uses_triangle_planes_not_bilinear_saddle(self):
        terrain = {'nx': 2, 'nz': 2, 'x0': 0, 'z0': 0, 'step': 8,
                   'heights': [0, 0, 0, 8]}
        actual = terrain_heights(np.array([[4, 4], [6, 2], [2, 6], [8, 8]]), terrain)
        np.testing.assert_allclose(actual, [4, 2, 2, 8])


if __name__ == '__main__':
    unittest.main()
