# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "rasterio==1.4.4", "shapely==2.1.2", "pyproj==3.7.2"]
# ///
"""Join pavement to the exact float32 boundary of the existing terrain mesh.

Subdivision preserves each original road quad diagonal. Inserted terrain vertices
are retained verbatim, including height; no contact tolerance or halo is added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree
from build_surface_mesh import road_edges

ROOT = Path(__file__).resolve().parents[1]
MAX_BOUNDARY_OFFSET_M = 0.0001


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_key(point):
    return tuple(map(float, point))


def edge_key(a, b):
    return tuple(sorted((point_key(a), point_key(b))))


def build(track_path, surface_path, output_path):
    track = json.loads(track_path.read_text())
    surface = json.loads(surface_path.read_text())
    if surface['metadata']['track_sha256'] != digest(track_path):
        raise ValueError('Terrain was built against a different track')
    original_footprint, starts, ends = road_edges(track)
    starts = starts.astype(np.float32).astype(float)
    ends = ends.astype(np.float32).astype(float)
    ground = np.asarray(surface['vertices'], dtype=np.float32).astype(float)
    triangles = np.asarray(surface['triangles'], dtype=int)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Terrain has nonmanifold edges')
    boundary = unique[counts == 1]
    bounds = surface['bounds']
    outer = np.zeros(len(boundary), dtype=bool)
    for axis, names in [(0, ('x0', 'x1')), (2, ('z0', 'z1'))]:
        for name in names:
            outer |= np.all(ground[boundary, axis] == bounds[name], axis=1)
    boundary = boundary[~outer]
    lines = shapely.linestrings(np.stack((starts[:, [0, 2]], ends[:, [0, 2]]), axis=1))
    midpoint = ground[boundary][:, :, [0, 2]].mean(axis=1)
    indices, distances = STRtree(lines).query_nearest(shapely.points(midpoint),
                                                     return_distance=True, all_matches=False)
    order = np.argsort(indices[0])
    if not np.array_equal(indices[0][order], np.arange(len(boundary))):
        raise ValueError('Missing nearest road segment')
    assigned = indices[1][order]
    groups = [[] for _ in starts]
    max_offset = 0.0
    max_height_offset = 0.0
    for edge, segment in zip(boundary, assigned):
        a, b = starts[segment], ends[segment]
        delta = b[[0, 2]] - a[[0, 2]]
        for vertex in edge:
            point = ground[vertex]
            t = np.dot(point[[0, 2]] - a[[0, 2]], delta) / np.dot(delta, delta)
            projected = a + np.clip(t, 0, 1) * (b - a)
            offset = float(np.linalg.norm(point[[0, 2]] - projected[[0, 2]]))
            max_offset = max(max_offset, offset)
            max_height_offset = max(max_height_offset, abs(float(point[1] - projected[1])))
            if offset > MAX_BOUNDARY_OFFSET_M:
                raise ValueError(f'Terrain boundary {vertex} is {offset}m from road segment {segment}')
            groups[segment].append((float(t), int(vertex)))
    if max_height_offset > MAX_BOUNDARY_OFFSET_M:
        raise ValueError(f'Terrain boundary height differs by {max_height_offset}m from road')
    chains = []
    for segment, group in enumerate(groups):
        chain = sorted(set(group))
        if not chain:
            raise ValueError(f'Road segment {segment} has no terrain boundary')
        points = ground[[vertex for _, vertex in chain]]
        if max(np.linalg.norm(points[0] - starts[segment]), np.linalg.norm(points[-1] - ends[segment])) > MAX_BOUNDARY_OFFSET_M:
            raise ValueError(f'Road segment {segment} endpoints do not exactly match ground: '
                             f'{points[0] - starts[segment]}, {points[-1] - ends[segment]}')
        chains.append(chain)
    # Terrain is already quantized, including its earlier nine decimal rounding.
    # Its exact endpoint is canonical when the difference is within the validated
    # float32 coordinate envelope. Adjacent road segments must name the same point.
    n = len(track['samples'])
    for segment, chain in enumerate(chains):
        next_segment = (segment // n) * n + (segment + 1) % n
        if chain[-1][1] != chains[next_segment][0][1]:
            raise ValueError(f'Boundary chain disconnect at segment {segment}')
        starts[segment] = ground[chain[0][1]]
        ends[segment] = ground[chain[-1][1]]

    vertices, uvs, output_triangles = [], [], []
    vertex_lookup = {}
    split_triangles = 0
    n = len(track['samples'])

    def vertex_id(position, uv):
        key = point_key(position), tuple(map(float, uv))
        if key not in vertex_lookup:
            vertex_lookup[key] = len(vertices)
            vertices.append(list(key[0]))
            uvs.append(list(key[1]))
        return vertex_lookup[key]

    def emit_polygon(positions, texcoords):
        nonlocal split_triangles
        positions = np.asarray(positions)
        texcoords = np.asarray(texcoords)
        if len(positions) == 3:
            faces = [(positions, texcoords)]
        else:
            split_triangles += 1
            center = positions.mean(axis=0).astype(np.float32).astype(float)
            center_uv = texcoords.mean(axis=0)
            faces = [(np.array([center, positions[k], positions[(k+1) % len(positions)]]),
                      np.array([center_uv, texcoords[k], texcoords[(k+1) % len(positions)]]))
                     for k in range(len(positions))]
        for p, uv in faces:
            xz = p[:, [0, 2]]
            ab, ac = xz[1] - xz[0], xz[2] - xz[0]
            area2 = float(ab[0]*ac[1] - ab[1]*ac[0])
            if area2 == 0 or not np.isfinite(p).all():
                raise ValueError('Degenerate or nonfinite pavement triangle')
            if area2 < 0:
                p, uv = p[::-1], uv[::-1]
            output_triangles.append([vertex_id(v, t) for v, t in zip(p, uv)])

    for i, sample in enumerate(track['samples']):
        j = (i + 1) % n
        a, d = starts[i], ends[i]
        b, c = starts[n+i], ends[n+i]
        s, sj = sample['s'], track['samples'][j]['s'] if j else track['length_m']
        ua, ub = np.array([0, s]), np.array([sample['width'], s])
        uc, ud = np.array([track['samples'][j]['width'], sj]), np.array([0, sj])
        right = chains[n+i][1:-1]
        left = list(reversed(chains[i][1:-1]))
        emit_polygon([a, b] + [ground[v] for _, v in right] + [c],
                     [ua, ub] + [ub + t*(uc-ub) for t, _ in right] + [uc])
        emit_polygon([a, c, d] + [ground[v] for _, v in left],
                     [ua, uc, ud] + [ua + t*(ud-ua) for t, _ in left])

    pv, pt = np.asarray(vertices), np.asarray(output_triangles)
    boundary_counts = Counter(edge_key(pv[a], pv[b]) for tri in pt
                              for a, b in zip(tri, np.roll(tri, -1)))
    actual_boundary = {edge for edge, count in boundary_counts.items() if count == 1}
    expected_boundary = {edge_key(ground[a], ground[b]) for a, b in boundary}
    if any(count > 2 for count in boundary_counts.values()) or actual_boundary != expected_boundary:
        raise ValueError(f'Pavement boundary mismatch: extra={len(actual_boundary-expected_boundary)}, '
                         f'missing={len(expected_boundary-actual_boundary)}')
    polygons = shapely.polygons(pv[pt][:, :, [0, 2]])
    union = shapely.union_all(polygons)
    target_lines = shapely.linestrings(ground[boundary][:, :, [0, 2]])
    rings = shapely.polygonize(target_lines)
    # Polygonization produces the ribbon plus its infield. Select regions whose
    # interior representative lies inside the original road, not the infield.
    target = shapely.union_all([region for region in shapely.get_parts(rings)
                               if original_footprint.covers(region.representative_point())])
    coverage_error = float(union.symmetric_difference(target).area)
    overlap = float(shapely.area(polygons).sum() - union.area)
    if max(coverage_error, abs(overlap)) > 1e-6:
        raise ValueError(f'Pavement coverage failed: error={coverage_error}, overlap={overlap}')
    report = {'vertices': len(vertices), 'triangles': len(output_triangles),
              'original_triangles': 2*n, 'split_original_triangles': split_triangles,
              'shared_boundary_edges': len(expected_boundary),
              'max_boundary_offset_m': max_offset, 'max_boundary_height_offset_m': max_height_offset,
              'coverage_error_m2': coverage_error, 'triangle_overlap_m2': overlap,
              'boundary_edges_exact': True}
    output = {'schema_version': 1, 'vertices': vertices, 'uvs': uvs,
              'triangles': output_triangles,
              'metadata': {'track_sha256': digest(track_path), 'surface_sha256': digest(surface_path),
                           'generator_sha256': digest(Path(__file__)),
                           'source': 'Historical road ribbon subdivided at exact terrain boundary vertices',
                           'license': 'ODbL 1.0 road derived geometry; public domain USGS terrain',
                           'attribution': 'OpenStreetMap contributors; USGS 3DEP',
                           'limitations': 'Historical provisional road geometry, not surveyed road margins.',
                           'validation': report}}
    output_path.write_text(json.dumps(output, separators=(',', ':'))+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=Path, default=ROOT/'godot/data/track.json')
    parser.add_argument('--surface', type=Path, default=ROOT/'godot/data/surface.json')
    parser.add_argument('--output', type=Path, default=ROOT/'godot/data/pavement.json')
    args = parser.parse_args()
    build(args.track, args.surface, args.output)
