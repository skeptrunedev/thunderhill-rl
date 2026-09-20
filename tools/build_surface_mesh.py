# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "rasterio==1.4.4", "shapely==2.1.2", "pyproj==3.7.2"]
# ///
"""Build road conforming terrain for rendering and offroad contact.

Run: uv run tools/build_surface_mesh.py --output-dir artifacts/surface
Heights outside the six metre shoulder blend preserve the original DEM grid's
triangle planes. The shoulder blend is provisional, not surveyed pavement or tire physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import rasterio
import shapely
from scipy.ndimage import map_coordinates
from shapely import Polygon, STRtree
from terrain_datum import terrain_transform

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_terrain(track, template, source):
    """Resample the raw DEM on exactly the existing terrain grid, without recess."""
    nx, nz, step = template['nx'], template['nz'], template['step']
    xx, zz = np.meshgrid(template['x0'] + np.arange(nx) * step,
                         template['z0'] + np.arange(nz) * step)
    origin = track['origin']
    transform, registration = terrain_transform()
    east, north = transform.transform(xx + origin['easting'], origin['northing'] - zz, errcheck=True)
    with rasterio.open(source) as src:
        if src.crs.to_epsg() != 26910:
            raise ValueError("Unexpected DEM coordinate reference system")
        cols = (east - src.transform.c) / src.transform.a - 0.5
        rows = (north - src.transform.f) / src.transform.e - 0.5
        if (np.any(cols < 0) or np.any(cols > src.width - 1)
                or np.any(rows < 0) or np.any(rows > src.height - 1)):
            raise ValueError("Transformed terrain grid leaves the source DEM")
        heights = map_coordinates(src.read(1), [rows.ravel(), cols.ravel()], order=1,
                                  mode='nearest') - origin['elevation_m']
    if not np.isfinite(heights).all():
        raise ValueError('DEM contains nonfinite samples')
    return {**{k: template[k] for k in ('schema_version', 'nx', 'nz', 'step', 'x0', 'z0')},
            'heights': np.round(heights, 3).tolist(),
            'metadata': {'source': 'USGS 2023 1 meter DEM', 'source_sha256': digest(source),
                         'source_crs': 'EPSG:26910', 'source_vertical_datum': 'NAVD88',
                         'horizontal_registration': registration,
                         'modifications': 'Bilinear DEM sampling on original 8 meter grid; no road recess.',
                         'license': 'Public domain USGS DEM'}}


def road_edges(track):
    samples = track['samples']
    p = np.array([s['p'] for s in samples], dtype=float)
    tangent = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    left = np.column_stack((tangent[:, 2], np.zeros(len(p)), -tangent[:, 0]))
    left /= np.linalg.norm(left, axis=1)[:, None]
    half = np.array([s['width'] / 2 for s in samples])
    bank = np.tan([s['bank'] for s in samples])
    edges = []
    for sign in (-1, 1):
        edge = p + left * (half * sign)[:, None]
        edge[:, 1] += bank * half * sign + 0.04
        edges.append(edge)
    quads = [Polygon([edges[0][i, [0, 2]], edges[0][(i+1) % len(p), [0, 2]],
                      edges[1][(i+1) % len(p), [0, 2]], edges[1][i, [0, 2]]])
             for i in range(len(p))]
    if not all(q.is_valid for q in quads):
        raise ValueError('Road contains invalid quads')
    footprint = shapely.union_all(quads)
    starts = np.concatenate(edges)
    ends = np.concatenate([np.roll(edge, -1, axis=0) for edge in edges])
    return footprint, starts, ends


def terrain_heights(xz, terrain):
    nx, nz = terrain['nx'], terrain['nz']
    grid = (xz - [terrain['x0'], terrain['z0']]) / terrain['step']
    grid = np.clip(grid, [0, 0], [nx-1, nz-1])
    ij = np.minimum(np.floor(grid).astype(int), [nx-2, nz-2])
    f = grid - ij
    h = np.array(terrain['heights']).reshape(nz, nx)
    x, z = ij.T
    a, b, c, d = h[z, x], h[z, x+1], h[z+1, x+1], h[z+1, x]
    return np.where(f[:, 0] >= f[:, 1], a+(b-a)*f[:, 0]+(c-b)*f[:, 1],
                    a+(c-d)*f[:, 0]+(d-a)*f[:, 1])


def segment_heights(xz, starts, ends):
    """Exact nearest segment lookup, with linear interpolation of its height."""
    lines = shapely.linestrings(np.stack((starts[:, [0, 2]], ends[:, [0, 2]]), axis=1))
    indices, distance = STRtree(lines).query_nearest(shapely.points(xz), return_distance=True,
                                                   all_matches=False)
    order = np.argsort(indices[0])
    if not np.array_equal(indices[0][order], np.arange(len(xz))):
        raise ValueError("Nearest segment lookup did not return every input exactly once")
    segment = indices[1][order]
    distance = distance[order]
    a, b = starts[segment], ends[segment]
    direction = b[:, [0, 2]] - a[:, [0, 2]]
    t = np.clip(np.sum((xz - a[:, [0, 2]]) * direction, axis=1) /
                np.sum(direction * direction, axis=1), 0, 1)
    return a[:, 1] + t * (b[:, 1] - a[:, 1]), distance


def clip_triangles(triangles, footprint, transition=None):
    """Subtract the road from each triangle; retain holes during constrained CDT."""
    polygons = shapely.polygons(triangles)
    touching = shapely.intersects(polygons, footprint)
    if transition is not None:
        touching |= shapely.intersects(polygons, shapely.boundary(transition))
    output = list(triangles[~touching])
    for polygon in polygons[touching]:
        clipped = polygon.difference(footprint)
        if clipped.is_empty:
            continue
        regions = [clipped] if transition is None else [
            clipped.intersection(transition), clipped.difference(transition)]
        for region in regions:
            for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(region)):
                if tri.area > 1e-12:
                    output.append(np.asarray(tri.exterior.coords)[:3])
    return np.array(output)


def build_mesh(terrain, footprint, starts, ends):
    step = terrain['step'] / 2
    xs = terrain['x0'] + np.arange((terrain['nx']-1)*2+1)*step
    zs = terrain['z0'] + np.arange((terrain['nz']-1)*2+1)*step
    xx, zz = np.meshgrid(xs[:-1], zs[:-1])
    a = np.column_stack((xx.ravel(), zz.ravel()))
    b, c, d = a+[step, 0], a+[step, step], a+[0, step]
    base = np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))
    transition = footprint.buffer(6, quad_segs=32)
    clipped = clip_triangles(base, footprint, transition)
    # A single canonical coordinate and height per vertex prevents shoulder cracks.
    xz, inverse = np.unique(np.round(clipped.reshape(-1, 2), 9), axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    raw = terrain_heights(xz, terrain)
    # Union boundary may remove interior edge segments at overlapping quads.
    rings = shapely.get_parts(shapely.boundary(footprint))
    segments = []
    for ring in rings:
        coords = np.asarray(ring.coords)
        segments.extend(pairwise(coords))
    boundary = np.asarray(segments)
    endpoint_h, _ = segment_heights(boundary.reshape(-1, 2), starts, ends)
    boundary3 = np.zeros((len(boundary), 2, 3))
    boundary3[:, :, [0, 2]] = boundary
    boundary3[:, :, 1] = endpoint_h.reshape(-1, 2)
    edge_h, distance = segment_heights(xz, boundary3[:, 0], boundary3[:, 1])
    blend = np.clip(distance / 6, 0, 1)
    blend = blend * blend * (3 - 2 * blend)
    inside = shapely.contains(transition, shapely.points(xz))
    transition_distance = shapely.distance(shapely.points(xz), shapely.boundary(transition))
    blend[(~inside) | (transition_distance < 1e-8)] = 1.0
    height = edge_h * (1-blend) + raw * blend
    vertices = np.column_stack((xz[:, 0], height, xz[:, 1]))
    # All triangles face up in Godot's clockwise front face convention.
    tri_xz = xz[triangles]
    signed = ((tri_xz[:, 1, 0]-tri_xz[:, 0, 0])*(tri_xz[:, 2, 1]-tri_xz[:, 0, 1]) -
              (tri_xz[:, 1, 1]-tri_xz[:, 0, 1])*(tri_xz[:, 2, 0]-tri_xz[:, 0, 0]))
    triangles[signed < 0] = triangles[signed < 0, ::-1]
    polygons = shapely.polygons(xz[triangles])
    union = shapely.union_all(polygons)
    bounds = shapely.box(xs[0], zs[0], xs[-1], zs[-1])
    target = bounds.difference(footprint)
    area_sum = float(shapely.area(polygons).sum())
    coverage_error = union.symmetric_difference(target).area
    overlap = area_sum - union.area
    road_overlap = union.intersection(footprint).area
    on_edge = distance < 1e-7
    analytic_h, _ = segment_heights(xz[on_edge], starts, ends)
    height_error = float(np.max(np.abs(height[on_edge]-analytic_h)))
    report = {'vertices': len(vertices), 'triangles': len(triangles),
              'coverage_error_m2': coverage_error, 'triangle_overlap_m2': overlap,
              'road_overlap_m2': road_overlap, 'boundary_vertices': int(on_edge.sum()),
              'boundary_height_error_m': height_error,
              'far_dem_height_error_m': float(np.max(np.abs(height[distance >= 6]-raw[distance >= 6])))}
    if max(coverage_error, abs(overlap), road_overlap) > 1e-4 or height_error > 1e-7:
        raise ValueError(f'Mesh geometric validation failed: {report}')
    return vertices.tolist(), triangles.tolist(), report


def build(output_dir, track_path=None, template_path=None):
    track_path = track_path or ROOT/'godot/data/track.json'
    template_path = template_path or ROOT/'godot/data/terrain.json'
    track = json.loads(track_path.read_text())
    template = json.loads(template_path.read_text())
    terrain = raw_terrain(track, template, ROOT/'artifacts/reference/geometry/east-terrain-1m.tif')
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_path = output_dir/'terrain.json'
    terrain_path.write_text(json.dumps(terrain, separators=(',', ':'))+'\n')
    footprint, starts, ends = road_edges(track)
    vertices, triangles, report = build_mesh(terrain, footprint, starts, ends)
    rings = [list(map(list, ring.coords)) for ring in shapely.get_parts(shapely.boundary(footprint))]
    bounds = {'x0': terrain['x0'], 'z0': terrain['z0'],
              'x1': terrain['x0'] + (terrain['nx']-1)*terrain['step'],
              'z1': terrain['z0'] + (terrain['nz']-1)*terrain['step']}
    mesh = {'schema_version': 1, 'vertices': vertices, 'triangles': triangles,
            'road_rings': rings, 'bounds': bounds,
            'metadata': {'track_sha256': digest(track_path), 'terrain_sha256': digest(terrain_path),
                         'source': 'Historical runtime road edges and USGS 2023 DEM',
                         'license': 'ODbL 1.0 road derived geometry; public domain USGS terrain',
                         'attribution': 'OpenStreetMap contributors; USGS 3DEP',
                         'grid_spacing_m': 4, 'shoulder_blend_m': 6,
                         'transition_boundary': '6 meter polygon buffer, 32 segments per quadrant; chord inset at round joins at most 0.001808 meter. Boundary and exterior heights exactly raw DEM triangle planes.',
                         'limitations': 'Visual shoulder blend; provisional historical widths and curbs; not surveyed road margins. Road boundary subdivision introduces T junctions with the existing road mesh, although segment heights coincide. Not a manifold topology guarantee.',
                         'validation': report}}
    (output_dir/'surface.json').write_text(json.dumps(mesh, separators=(',', ':'))+'\n')
    (output_dir/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/surface')
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == '__main__':
    main()
