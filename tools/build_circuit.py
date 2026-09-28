# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "rasterio==1.4.4", "shapely==2.1.2", "Pillow==12.1.1"]
# ///
"""Build a selectable MotoGP circuit from pinned, openly licensed sources.

  uv run tools/build_circuit.py <circuit-id>            fetch, verify, build
  uv run tools/build_circuit.py <circuit-id> --update-lock   accept changed sources
  uv run tools/build_circuit.py discover <circuit-id>   propose an OSM route
  uv run tools/build_circuit.py preview <circuit-id> --station S [--span M]
  uv run tools/build_circuit.py views <circuit-id> --godot G --stations S..   game screenshots
  uv run tools/build_circuit.py list                    circuits and build state
  uv run tools/build_circuit.py docs                    regenerate docs/motogp-circuits.md

The manifest data/circuits/<id>.json names every source: the OSM ways of the
layout MotoGP uses, the orthoimagery and elevation services, their licences, and
the SHA256 locks of every response. Raw responses stay in ignored
artifacts/circuits/<id>/. Outputs:

  godot/tracks/<id>/track.json          committed: centerline, width, bank, provenance
  godot/tracks/<id>/generated/          ignored: terrain, surface and pavement meshes,
                                        horizon, orthoimagery tiles for the game

The centerline follows build_track.py: OSM nodes projected into the circuit's
metric CRS, resampled at 0.5 m, Gaussian smoothed with a bounded displacement,
and sampled every 3 m. Pavement edges are then measured across the road in the
orthoimagery itself, which also recentres the OSM line on the imaged pavement.
Heights and bank come from the pinned DEM. The computed lap length must be
within 1% of the official MotoGP length or the build fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Proj, Transformer
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_sources as sources  # noqa: E402
from build_pavement_mesh import build as build_pavement  # noqa: E402
from build_surface_mesh import build_mesh, road_edges  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/circuits"
TRACKS = ROOT / "godot/tracks"
ROAD_SPACING = 3.0
TERRAIN_SPACING = 8.0
TILE_M = 256.0
CORRIDOR_M = 70.0
DOMAIN_MARGIN_M = 260.0
HORIZON_MARGIN_M = 4000.0
HORIZON_SPACING = 32.0
LENGTH_TOLERANCE = 0.01
# Aged asphalt albedo is about 0.10 to 0.18 (new asphalt about 0.05; Lawrence Berkeley
# National Laboratory Heat Island Group, "Cool Pavements"). Appearance calibration only.
ASPHALT_ALBEDO = 0.12
# Measured widths are accepted from this far below to this far above the published width.
WIDTH_BELOW_M = 1.5
WIDTH_ABOVE_M = 3.5
PROFILE_HALF_M = 22.0
PROFILE_STEP_M = 0.25
# Thunderhill's reviewed curb profile (godot/data/curb-placement.json), reused as a
# provisional shape; the imagery locates painted kerbs but cannot measure their height.
CURB_PROFILE = {"width_m": 0.9, "inner_height_m": 0.05, "outer_height_m": 0.11}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(circuit: str) -> tuple[dict, Path]:
    path = MANIFESTS / f"{circuit}.json"
    if not path.exists():
        raise SystemExit(f"No manifest {path}")
    manifest = json.loads(path.read_text())
    if manifest["id"] != circuit:
        raise SystemExit("Manifest id differs from its file name")
    return manifest, path


# Route -------------------------------------------------------------------------

def way_chain(part: dict, ids: list[int]) -> list[int]:
    """Node ids of one route part, from part["from"] to part["to"] inclusive.

    A closed way (a ring) is walked forward in its own node order, wrapping past
    its closure node when needed, or backward with "reverse": true; from == to
    takes the whole ring. An open way is walked from one index to the other; its
    direction follows the indices, and a contradicting "reverse" flag is an error.
    Nodes that occur more than once (other than a ring's closure) are ambiguous.
    """
    start, end = part["from"], part["to"]
    closed = ids[0] == ids[-1]
    body = ids[:-1] if closed else ids
    for node in {start, end}:
        if body.count(node) != 1:
            raise SystemExit(f"Way {part['way']}: node {node} occurs {body.count(node)} times; "
                             "the route part is ambiguous")
    a, b = body.index(start), body.index(end)
    reverse = bool(part.get("reverse"))
    if closed:
        n = len(body)
        steps = ((b - a) % n or n) if not reverse else ((a - b) % n or n)
        sign = -1 if reverse else 1
        return [body[(a + sign * k) % n] for k in range(steps + 1)]
    if a == b:
        raise SystemExit(f"Way {part['way']} is open; from and to must differ")
    if "reverse" in part and reverse != (a > b):
        raise SystemExit(f"Way {part['way']}: reverse flag contradicts the from/to node order")
    return ids[a:b + 1] if a < b else ids[b:a + 1][::-1]


def route_nodes(manifest: dict, elements: dict) -> tuple[list[dict], list]:
    """Ordered node documents of the route, and the pinned snapshot rows."""
    ordered, snapshot = [], []
    for part in manifest["osm"]["route"]:
        way = elements[("way", part["way"])]
        ids = way["nodes"]
        chain = way_chain(part, ids)
        nodes = [elements[("node", n)] for n in chain]
        if ordered and ordered[-1]["id"] != chain[0]:
            # OSM sometimes leaves two unconnected nodes at one position where a way
            # was drawn to meet another. The manifest must say so explicitly, and the
            # coordinates must be exactly equal; any gap at all is a route break.
            same_place = (ordered[-1]["lon"], ordered[-1]["lat"]) == (nodes[0]["lon"], nodes[0]["lat"])
            if not (part.get("join_coincident_node") and same_place):
                raise SystemExit(f"Route break before way {part['way']}: {ordered[-1]['id']} to {chain[0]}")
        ordered.extend(nodes[1:] if ordered else nodes)
        snapshot.append([part["way"], way["version"],
                         [[n["id"], n["version"], n["lon"], n["lat"]] for n in nodes]])
    closes = ordered[0]["id"] == ordered[-1]["id"] or (
        manifest["osm"]["route"][0].get("join_coincident_node")
        and (ordered[0]["lon"], ordered[0]["lat"]) == (ordered[-1]["lon"], ordered[-1]["lat"]))
    if not closes:
        raise SystemExit("Route is not closed")
    return ordered[:-1], snapshot


def project(crs: str, lon, lat):
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return transform.transform(lon, lat)


def closed_length(points: np.ndarray) -> float:
    closed = np.vstack([points, points[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def resample(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    return np.column_stack([np.interp(distances % station[-1], station, closed[:, k])
                            for k in range(points.shape[1])])


def normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    forward /= np.linalg.norm(forward, axis=1)[:, None]
    return forward, np.column_stack([-forward[:, 1], forward[:, 0]])


def rotate_to_start(points: np.ndarray, start_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Insert the projection of the start/finish point and make it index zero."""
    best, where = math.inf, None
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        ab = b - a
        t = float(np.clip(np.dot(start_xy - a, ab) / np.dot(ab, ab), 0, 1))
        d = float(np.linalg.norm(a + t * ab - start_xy))
        if d < best:
            best, where = d, (i, t)
    i, t = where
    a, b = points[i], points[(i + 1) % len(points)]
    q = a + t * (b - a)
    rest = np.vstack([points[i + 1:], points[:i + 1]])
    keep = [p for p in rest if np.linalg.norm(p - q) > 0.05]
    return np.vstack([q, np.array(keep)]), best


def smooth_centerline(points: np.ndarray, max_displacement: float) -> tuple[np.ndarray, float, float]:
    length = closed_length(points)
    dense = resample(points, np.arange(0, length, 0.5))
    sigma = 4.0
    while True:
        smooth = gaussian_filter1d(dense, sigma=sigma, axis=0, mode="wrap")
        moved = float(np.linalg.norm(smooth - dense, axis=1).max())
        if moved <= max_displacement:
            return smooth, moved, sigma * 0.5
        sigma *= 0.8


# Orthoimagery ------------------------------------------------------------------

class TileSet:
    """Detail orthoimagery on an aligned 256 m grid in the circuit CRS."""

    def __init__(self, resolution: float):
        self.resolution = resolution
        self.pixels = round(TILE_M / resolution)
        self.tiles: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def keys_for(points: np.ndarray, radius: float) -> list[tuple[int, int]]:
        keys = set()
        step = TILE_M / 4
        for p in points:
            for dx in np.arange(-radius, radius + step, step):
                for dy in np.arange(-radius, radius + step, step):
                    keys.add((int(math.floor((p[0] + dx) / TILE_M)), int(math.floor((p[1] + dy) / TILE_M))))
        return sorted(keys)

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear RGB in [0, 1]; NaN where no tile was acquired."""
        out = np.full((len(xy), 3), np.nan)
        tx = np.floor(xy[:, 0] / TILE_M).astype(int)
        ty = np.floor(xy[:, 1] / TILE_M).astype(int)
        for key in set(zip(tx.tolist(), ty.tolist())):
            image = self.tiles.get(key)
            if image is None:
                continue
            mask = (tx == key[0]) & (ty == key[1])
            col = (xy[mask, 0] - key[0] * TILE_M) / self.resolution - 0.5
            row = ((key[1] + 1) * TILE_M - xy[mask, 1]) / self.resolution - 0.5
            for band in range(3):
                out[mask, band] = map_coordinates(image[:, :, band].astype(float), [row, col],
                                                  order=1, mode="nearest") / 255.0
        return out


def acquire_tiles(manifest, product, keys, crs) -> TileSet:
    spec = manifest["imagery"]["detail"]
    tiles = TileSet(spec["resolution_m"])
    from concurrent.futures import ThreadPoolExecutor

    def one(key):
        bounds = (key[0] * TILE_M, key[1] * TILE_M, (key[0] + 1) * TILE_M, (key[1] + 1) * TILE_M)
        return key, sources.imagery_tile(spec["source"], product, crs, bounds, tiles.pixels,
                                         f"{key[0]}_{key[1]}")

    with ThreadPoolExecutor(spec["source"].get("parallel", 4)) as pool:
        for key, image in pool.map(one, keys):
            tiles.tiles[key] = image
    return tiles


def measure_edges(tiles: TileSet, points: np.ndarray, left: np.ndarray, width_range) -> dict:
    """Pavement edges across the road from orthoimagery colour, per sample.

    Asphalt is characterised from pixels within two metres of the OSM line around
    the whole lap. Moving outward from the centre, an edge is the start of at least
    one metre of non-asphalt, non-marking colour, or the outside of a white line
    that lies in the plausible edge band. Samples whose width leaves the published
    range, widened by two metres, are rejected and filled from their neighbours.
    """
    offsets = np.arange(-PROFILE_HALF_M, PROFILE_HALF_M + 1e-9, PROFILE_STEP_M)
    grid = points[:, None, :] + left[:, None, :] * offsets[None, :, None]
    rgb = tiles.sample(grid.reshape(-1, 2)).reshape(len(points), len(offsets), 3)
    if np.isnan(rgb).any():
        raise SystemExit("Detail imagery does not cover the road corridor")
    lum = rgb.mean(axis=2)
    sat = (rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 1e-3)
    core = np.abs(offsets) <= 2.0
    ref_lum = float(np.median(lum[:, core]))
    ref_sat = float(np.median(sat[:, core]))
    mad = float(np.median(np.abs(lum[:, core] - ref_lum)))
    tol = max(3.5 * mad, 0.07)
    asphalt = (np.abs(lum - ref_lum) <= tol) & (sat <= ref_sat + 0.10)
    white = (lum >= max(ref_lum + 0.25, 0.62)) & (sat <= 0.22)
    lo, hi = width_range
    min_half, max_half = lo / 2 - 2.0, hi / 2 + 2.0
    step = PROFILE_STEP_M
    run = int(round(1.0 / step))
    centre = len(offsets) // 2
    edges = np.full((len(points), 2), np.nan)
    for i in range(len(points)):
        for side, direction in ((0, 1), (1, -1)):
            k = centre
            found = None
            gap = 0
            while 0 <= k + direction < len(offsets):
                k += direction
                distance = abs(offsets[k])
                if white[i, k]:
                    # Outer boundary of a white line inside the edge band is the edge.
                    j = k
                    while 0 <= j + direction < len(offsets) and white[i, j + direction]:
                        j += direction
                    if min_half <= abs(offsets[j]) + step / 2 <= max_half and not asphalt[i, j + direction if 0 <= j + direction < len(offsets) else j]:
                        found = abs(offsets[j]) + step / 2
                        break
                    k = j
                    gap = 0
                    continue
                if asphalt[i, k]:
                    gap = 0
                    continue
                gap += 1
                if gap >= run:
                    candidate = distance - (gap - 0.5) * step
                    if candidate >= min_half:
                        found = candidate
                        break
                    gap = 0
            if found is not None and found <= max_half:
                edges[i, side] = found
    width = edges.sum(axis=1)
    valid = np.isfinite(width) & (width >= lo - 1.0) & (width <= hi + 1.0)
    return {"edges": edges, "valid": valid, "asphalt_luminance": ref_lum,
            "asphalt_saturation": ref_sat, "luminance_tolerance": tol}


def fill_periodic(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    index = np.arange(len(values))
    if valid.sum() < 2:
        raise SystemExit("Too few valid edge measurements")
    return np.interp(index, index[valid], values[valid], period=len(values))


def fit_width(measured: dict, width_range, published: float, count: int) -> tuple[np.ndarray, np.ndarray, dict]:
    lo, hi = width_range
    valid = measured["valid"]
    fraction = float(valid.mean())
    if fraction < 0.4:
        width = np.full(count, float(published))
        offset = np.zeros(count)
        status = "Published width; fewer than 40% of imagery edge measurements were plausible"
    else:
        left_edge = fill_periodic(measured["edges"][:, 0], valid)
        right_edge = fill_periodic(measured["edges"][:, 1], valid)
        left_edge = gaussian_filter1d(median_filter(left_edge, 9, mode="wrap"), 2.0, mode="wrap")
        right_edge = gaussian_filter1d(median_filter(right_edge, 9, mode="wrap"), 2.0, mode="wrap")
        width = np.clip(left_edge + right_edge, lo - 0.5, hi + 0.5)
        offset = np.clip((left_edge - right_edge) / 2, -3.0, 3.0)
        status = "Measured from orthoimagery pavement colour; median 27 m and Gaussian 6 m smoothed; gaps interpolated"
    return width, offset, {"valid_fraction": fraction, "status": status,
                           "asphalt_luminance": measured["asphalt_luminance"],
                           "asphalt_saturation": measured["asphalt_saturation"],
                           "luminance_tolerance": measured["luminance_tolerance"]}


def detect_curbs(tiles: TileSet, points, left, width) -> list[dict]:
    """Painted kerbs: a band outside the edge alternating saturated colour and white."""
    runs = []
    n = len(points)
    flags = np.zeros((n, 2), dtype=bool)
    along = np.arange(0, ROAD_SPACING, 0.25)
    for side, sign in ((1, 1.0), (-1, -1.0)):
        column = 0 if side == 1 else 1
        tangent = np.roll(points, -1, axis=0) - points
        samples = []
        for k in range(n):
            base = points[k] + tangent[k] * (along / ROAD_SPACING)[:, None]
            across = np.array([0.25, 0.55, 0.85])
            xy = base[:, None, :] + (sign * left[k] * (width[k] / 2 + across[:, None]))[None, :, :]
            samples.append(xy.reshape(-1, 2))
        rgb = tiles.sample(np.concatenate(samples)).reshape(n, len(along), 3, 3).mean(axis=2)
        lum = rgb.mean(axis=2)
        sat = (rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 1e-3)
        coloured = (sat >= 0.35) & (lum >= 0.2)
        bright = (lum >= 0.65) & (sat <= 0.25)
        window = 2  # samples each side: a 15 m window
        for k in range(n):
            idx = [(k + j) % n for j in range(-window, window + 1)]
            c, b = coloured[idx].mean(), bright[idx].mean()
            flags[k, column] = c >= 0.2 and b >= 0.2 and c + b >= 0.6
    for side, column in ((1, 0), (-1, 1)):
        k = 0
        f = flags[:, column]
        while k < n:
            if f[k]:
                start = k
                while k < n and f[k]:
                    k += 1
                if k - start >= 2:
                    runs.append({"id": f"K{len(runs) + 1:03d}", "side": side,
                                 "segments": list(range(start, k)),
                                 "status": "Detected in orthoimagery: alternating saturated and white band outside the measured edge"})
            else:
                k += 1
    return runs


# Elevation ---------------------------------------------------------------------

class Heights:
    def __init__(self, array: np.ndarray, bounds, resolution: float):
        self.array, self.bounds, self.resolution = array, bounds, resolution
        if np.isnan(array).all():
            raise SystemExit("DEM has no valid samples")
        if np.isnan(array).any():
            from scipy.ndimage import distance_transform_edt
            index = distance_transform_edt(np.isnan(array), return_distances=False, return_indices=True)
            self.filled_fraction = float(np.isnan(array).mean())
            self.array = array[tuple(index)]
        else:
            self.filled_fraction = 0.0

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        col = (xy[:, 0] - x0) / self.resolution - 0.5
        row = (y1 - xy[:, 1]) / self.resolution - 0.5
        if (col.min() < -0.5 or row.min() < -0.5 or col.max() > self.array.shape[1] - 0.5
                or row.max() > self.array.shape[0] - 0.5):
            raise SystemExit("Sample leaves the DEM window")
        return map_coordinates(self.array, [row, col], order=1, mode="nearest")


def road_profile(heights: Heights, points, left, width, dem_resolution, bank_from_dem=True):
    centre = heights(points)
    sigma_m = max(6.0, dem_resolution)
    smooth = gaussian_filter1d(median_filter(centre, 5, mode="wrap"), sigma_m / ROAD_SPACING, mode="wrap")
    if dem_resolution <= 5.0 and bank_from_dem:
        half = (width / 2)[:, None]
        hl = heights(points + left * half)
        hr = heights(points - left * half)
        bank = np.arctan((hl - hr) / width)
        bank = gaussian_filter1d(median_filter(bank, 5, mode="wrap"), sigma_m / ROAD_SPACING, mode="wrap")
        bank = np.clip(bank, -math.radians(8), math.radians(8))
        bank_status = "DEM cross slope between the measured edges, median and Gaussian smoothed, clamped to 8 degrees"
    else:
        bank = np.zeros(len(points))
        bank_status = (f"Zero: the {dem_resolution:g} m DEM cannot resolve cross slope" if dem_resolution > 5.0
                       else "Zero: the DEM service rounds heights to whole metres, too coarse for cross slope")
    return smooth, bank, {"height_sigma_m": sigma_m, "bank": bank_status,
                          "max_smoothing_change_m": float(np.abs(smooth - centre).max())}


# Build -------------------------------------------------------------------------

def build(circuit: str, update_lock: bool) -> None:
    manifest, manifest_path = load_manifest(circuit)
    crs = manifest["crs"]
    cache = ROOT / "artifacts/circuits" / circuit / "raw"
    locks = manifest.setdefault("locks", {})
    new_locks = {}

    osm = sources.Product("osm", cache)
    elements = sources.fetch_osm_ways(osm, [{"id": r["way"], "version": r["version"]} for r in manifest["osm"]["route"]])
    nodes, snapshot = route_nodes(manifest, elements)
    snapshot_hash = sources.sha256_bytes(json.dumps(snapshot, separators=(",", ":")).encode())
    if manifest["osm"].get("snapshot_sha256") not in (None, snapshot_hash) and not update_lock:
        raise SystemExit("OSM route nodes changed. Review and rerun with --update-lock.")
    new_locks["osm_snapshot"] = snapshot_hash
    x, y = project(crs, [n["lon"] for n in nodes], [n["lat"] for n in nodes])
    raw = np.column_stack([x, y])
    area = 0.5 * float(np.sum(raw[:, 0] * np.roll(raw[:, 1], -1) - np.roll(raw[:, 0], -1) * raw[:, 1]))
    direction = "counterclockwise" if area > 0 else "clockwise"
    if direction != manifest["direction"]:
        raise SystemExit(f"Route runs {direction}, manifest says {manifest['direction']}")
    sx, sy = project(crs, *manifest["start_finish"]["lonlat"])
    raw, start_offset = rotate_to_start(raw, np.array([sx, sy]))
    if start_offset > 15:
        raise SystemExit(f"Start/finish point is {start_offset:.1f} m from the route")
    source_length = closed_length(raw)
    official = manifest["official_length_m"]
    smooth, smoothing_moved, smoothing_sigma = smooth_centerline(raw, manifest.get("max_smoothing_m", 0.75))

    # Imagery corridor, measured pavement edges and recentred line.
    count = int(np.ceil(source_length / ROAD_SPACING))
    stations = np.linspace(0, closed_length(smooth), count, endpoint=False)
    points = resample(smooth, stations)
    _, left = normals(points)
    # Published MotoGP widths are nominal; real pavement narrows and widens around it.
    published_width = manifest["width_m"]["published"]
    width_range = (published_width - WIDTH_BELOW_M, published_width + WIDTH_ABOVE_M)
    tiles = None
    if manifest["imagery"].get("detail"):
        detail = sources.Product("imagery-detail", cache)
        keys = TileSet.keys_for(points[::10], CORRIDOR_M)
        tiles = acquire_tiles(manifest, detail, keys, crs)
        new_locks["imagery_detail"] = sources.check_lock(detail, locks.get("imagery_detail"), update_lock)
        measured = measure_edges(tiles, points, left, width_range)
        width, offset, width_meta = fit_width(measured, width_range, published_width, count)
    else:
        # Satellite imagery (10 m) cannot resolve pavement edges or kerbs.
        width, offset = np.full(count, float(published_width)), np.zeros(count)
        width_meta = {"valid_fraction": 0.0, "status": "Published width; no open imagery fine enough to measure edges"}
    points = points + left * offset[:, None]
    # Recentre and resample once more so samples stay evenly spaced.
    length_after = closed_length(points)
    stations2 = np.linspace(0, length_after, count, endpoint=False)
    s_old = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1))][:-1]
    width = np.interp(stations2, s_old, width, period=length_after)
    points = resample(points, stations2)
    heading, left = normals(points)
    curbs = detect_curbs(tiles, points, left, width) if tiles is not None else []

    # Domain, elevation and local origin.
    lo_xy, hi_xy = points.min(axis=0), points.max(axis=0)
    domain = (math.floor((lo_xy[0] - DOMAIN_MARGIN_M) / TERRAIN_SPACING) * TERRAIN_SPACING,
              math.floor((lo_xy[1] - DOMAIN_MARGIN_M) / TERRAIN_SPACING) * TERRAIN_SPACING,
              math.ceil((hi_xy[0] + DOMAIN_MARGIN_M) / TERRAIN_SPACING) * TERRAIN_SPACING,
              math.ceil((hi_xy[1] + DOMAIN_MARGIN_M) / TERRAIN_SPACING) * TERRAIN_SPACING)
    dem_spec = manifest["elevation"]["detail"]
    dem_res = dem_spec["resolution_m"]
    pad = 4 * dem_res
    dem_bounds = (domain[0] - pad, domain[1] - pad, domain[2] + pad, domain[3] + pad)
    dem_product = sources.Product("dem", cache)
    dem = Heights(sources.elevation(dem_spec["source"], dem_product, crs, dem_bounds, dem_res), dem_bounds, dem_res)
    new_locks["dem"] = sources.check_lock(dem_product, locks.get("dem"), update_lock)
    heights, banks, height_meta = road_profile(dem, points, left, width, dem_res,
                                               dem_spec.get("bank_from_dem", True))
    # Whole metre origin: grid bounds stay exactly representable in float32, which
    # the pavement join relies on to recognise the outer terrain boundary.
    origin = np.array([round(points[0, 0]), round(points[0, 1]), round(float(heights[0]))], dtype=float)

    xyz = np.column_stack([points[:, 0] - origin[0], heights - origin[2], origin[1] - points[:, 1]])
    left_xyz = np.column_stack([left[:, 0], np.zeros(count), -left[:, 1]])
    tangent = np.roll(xyz, -1, axis=0) - np.roll(xyz, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    segments = np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1)
    station = np.r_[0, np.cumsum(segments[:-1])]
    turn = np.arctan2(heading[:, 0] * np.roll(heading[:, 1], -1) - heading[:, 1] * np.roll(heading[:, 0], -1),
                      np.sum(heading * np.roll(heading, -1, axis=0), axis=1))
    curvature = turn / segments
    length = float(segments.sum())
    # The projection scales distances slightly (UTM up to about 0.1%); the ground length
    # divides each horizontal step by the point scale factor before adding the climb.
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(points[:, 0], points[:, 1])
    scale = np.asarray(Proj(crs).get_factors(lon, lat).meridional_scale)
    horizontal = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    climb = np.roll(heights, -1) - heights
    ground_length = float(np.sum(np.hypot(horizontal / scale, climb)))
    # A manifest may declare that the built layout is not the official one (a held-out
    # test track built from the layout OSM holds). The 1% gate then checks the length
    # of that declared layout, which must itself cite a published source.
    reference = manifest.get("length_reference")
    gate = float(reference["length_m"]) if reference else float(official)
    error = abs(length - gate) / gate
    ground_error = abs(ground_length - gate) / gate
    report = {"circuit": circuit, "official_length_m": official, "gate_length_m": gate,
              "computed_length_m": round(length, 3),
              "ground_length_m": round(ground_length, 3), "length_error": round(error, 5),
              "ground_length_error": round(ground_error, 5), "projection_scale_factor": round(float(scale.mean()), 7),
              "osm_polyline_length_m": round(source_length, 3)}
    print(json.dumps(report), flush=True)
    if max(error, ground_error) > LENGTH_TOLERANCE:
        raise SystemExit(f"Computed length {length:.1f} m (ground {ground_length:.1f} m) is more than "
                         f"{100 * LENGTH_TOLERANCE:g}% from {'the declared reference' if reference else 'official'} {gate:g} m")
    assert segments.min() > 1.0 and segments.max() < 5.0

    samples = [{"s": round(float(station[i]), 3), "p": np.round(xyz[i], 4).tolist(),
                "tangent": np.round(tangent[i], 7).tolist(), "left": np.round(left_xyz[i], 7).tolist(),
                "width": round(float(width[i]), 4), "bank": round(float(banks[i]), 7),
                "curvature": round(float(curvature[i]), 7)} for i in range(count)]
    out_dir = TRACKS / circuit
    generated = out_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / ".gdignore").write_text("")
    primary = manifest["imagery"].get("detail") or manifest["imagery"]["overview"]
    imagery_meta = {k: v for k, v in primary.items() if k != "source"}
    track = {
        "schema_version": 1, "id": circuit, "version": manifest["version"], "name": manifest["name"],
        "closed": True, "length_m": round(length, 3),
        "origin": {"easting": float(origin[0]), "northing": float(origin[1]), "elevation_m": float(origin[2]),
                   "crs": crs},
        "coordinates": "x east, z south, y up; meters", "bank_convention":
        "Positive bank raises the geometric left road edge relative to center",
        "samples": samples, "terrain_file": f"res://tracks/{circuit}/generated/terrain.json",
        "metadata": {
            "country": manifest["country"], "layout": manifest["layout"], "direction": manifest["direction"],
            "official_length_m": official, "official_length_source": manifest["official_length_source"],
            **({"length_reference": reference} if reference else {}),
            **({"purpose": manifest["purpose"]} if manifest.get("purpose") else {}),
            "length_error": round(error, 5), "ground_length_m": round(ground_length, 3),
            "ground_length_error": round(ground_error, 5),
            "projection_scale_factor": round(float(scale.mean()), 7), "start_finish": manifest["start_finish"],
            "start_finish_offset_from_osm_m": round(start_offset, 3),
            "osm": {"route": manifest["osm"]["route"], "snapshot_sha256": snapshot_hash,
                    "polyline_length_m": round(source_length, 3),
                    "plan_smoothing_max_displacement_m": round(smoothing_moved, 4),
                    "plan_smoothing_sigma_m": round(smoothing_sigma, 4)},
            "width": {**width_meta, "published_m": published_width, "plausible_range_m": width_range,
                      "source": manifest["width_m"]["source"],
                      "range_m": [round(float(width.min()), 3), round(float(width.max()), 3)]},
            "elevation": {**{k: v for k, v in dem_spec.items() if k != "source"}, **height_meta,
                          "dem_filled_fraction": dem.filled_fraction},
            "imagery": imagery_meta, "curb_runs": len(curbs),
            "license": "Open Database License 1.0 (ODbL) for OSM derived geometry, separate from repository MIT source code",
            "attribution": "; ".join(["OpenStreetMap contributors", primary["attribution"],
                                      dem_spec["attribution"]]),
            "limitations": manifest.get("limitations", []) + [
                "Centerline is OSM geometry recentred on imaged pavement, not a survey",
                "Kerb positions come from imagery colour; kerb height reuses Thunderhill's provisional profile",
                "Grip, run-off surfaces, walls and barriers are not modelled"],
        },
    }
    track_path = out_dir / "track.json"
    track_path.write_text(json.dumps(track, separators=(",", ":")) + "\n")
    curb_doc = {"schema_version": 1, "track_sha256": digest(track_path), "sample_count": count, **CURB_PROFILE,
                "source_reference": f"data/circuits/{circuit}.json", "runs": curbs}
    (out_dir / "curb-placement.json").write_text(json.dumps(curb_doc, indent=1) + "\n")

    # Terrain grid, road conforming surface, and pavement ribbon.
    x0, x1 = domain[0] - origin[0], domain[2] - origin[0]
    z0, z1 = origin[1] - domain[3], origin[1] - domain[1]
    nx, nz = round((x1 - x0) / TERRAIN_SPACING) + 1, round((z1 - z0) / TERRAIN_SPACING) + 1
    gx, gz = np.meshgrid(x0 + np.arange(nx) * TERRAIN_SPACING, z0 + np.arange(nz) * TERRAIN_SPACING)
    grid_heights = dem(np.column_stack([gx.ravel() + origin[0], origin[1] - gz.ravel()])) - origin[2]
    terrain = {"schema_version": 1, "nx": nx, "nz": nz, "step": TERRAIN_SPACING, "x0": float(x0), "z0": float(z0),
               "heights": np.round(grid_heights, 3).tolist(),
               "metadata": {"source": dem_spec["name"], "source_lock": new_locks["dem"], "source_crs": crs,
                            "modifications": "Bilinear DEM sampling on an 8 meter grid; no road recess.",
                            "license": dem_spec["license"], "attribution": dem_spec["attribution"]}}
    terrain_path = generated / "terrain.json"
    terrain_path.write_text(json.dumps(terrain, separators=(",", ":")) + "\n")
    footprint, starts, ends = road_edges(track)
    vertices, triangles, validation = build_mesh(terrain, footprint, starts, ends)
    rings = [list(map(list, ring.coords)) for ring in __import__("shapely").get_parts(__import__("shapely").boundary(footprint))]
    surface = {"schema_version": 1, "vertices": vertices, "triangles": triangles, "road_rings": rings,
               "bounds": {"x0": terrain["x0"], "z0": terrain["z0"], "x1": terrain["x0"] + (nx - 1) * TERRAIN_SPACING,
                          "z1": terrain["z0"] + (nz - 1) * TERRAIN_SPACING},
               "metadata": {"track_sha256": digest(track_path), "terrain_sha256": digest(terrain_path),
                            "source": "Circuit road edges and pinned DEM", "grid_spacing_m": 4,
                            "shoulder_blend_m": 6, "license": "ODbL 1.0 road derived geometry; " + dem_spec["license"],
                            "attribution": "OpenStreetMap contributors; " + dem_spec["attribution"],
                            "validation": validation}}
    surface_path = generated / "surface.json"
    surface_path.write_text(json.dumps(surface, separators=(",", ":")) + "\n")
    build_pavement(track_path, surface_path, generated / "pavement.json",
                   metadata={"source": "Circuit road ribbon subdivided at exact terrain boundary vertices",
                             "license": "ODbL 1.0 road derived geometry; " + dem_spec["license"],
                             "attribution": "OpenStreetMap contributors; " + dem_spec["attribution"],
                             "limitations": "OSM centerline recentred on imaged pavement; not surveyed road margins."})

    # Horizon terrain from the wide area DEM.
    hspec = manifest["elevation"]["horizon"]
    hres = hspec["resolution_m"]
    hdomain = (domain[0] - HORIZON_MARGIN_M, domain[1] - HORIZON_MARGIN_M,
               domain[2] + HORIZON_MARGIN_M, domain[3] + HORIZON_MARGIN_M)
    hpad = 4 * hres
    hbounds = (hdomain[0] - hpad, hdomain[1] - hpad, hdomain[2] + hpad, hdomain[3] + hpad)
    hproduct = sources.Product("dem-horizon", cache)
    hdem = Heights(sources.elevation(hspec["source"], hproduct, crs, hbounds, hres), hbounds, hres)
    new_locks["dem_horizon"] = sources.check_lock(hproduct, locks.get("dem_horizon"), update_lock)
    hx0 = x0 - HORIZON_MARGIN_M
    hz0 = z0 - HORIZON_MARGIN_M
    hnx = round((x1 - x0 + 2 * HORIZON_MARGIN_M) / HORIZON_SPACING) + 1
    hnz = round((z1 - z0 + 2 * HORIZON_MARGIN_M) / HORIZON_SPACING) + 1
    hx, hz = np.meshgrid(hx0 + np.arange(hnx) * HORIZON_SPACING, hz0 + np.arange(hnz) * HORIZON_SPACING)
    hh = hdem(np.column_stack([hx.ravel() + origin[0], origin[1] - hz.ravel()])) - origin[2]
    horizon = {"schema_version": 1, "nx": hnx, "nz": hnz, "step": HORIZON_SPACING, "x0": hx0, "z0": hz0,
               "heights": np.round(hh, 2).tolist(),
               "metadata": {"source": hspec["name"], "source_lock": new_locks["dem_horizon"],
                            "license": hspec["license"], "attribution": hspec["attribution"]}}
    (generated / "horizon.json").write_text(json.dumps(horizon, separators=(",", ":")) + "\n")

    # Game imagery: detail tiles, a domain overview and a horizon image.
    imagery_dir = generated / "imagery"
    if imagery_dir.exists():
        shutil.rmtree(imagery_dir)
    imagery_dir.mkdir()
    layers, tile_grid = [], None
    if tiles is not None:
        game_keys = TileSet.keys_for(points[::4], CORRIDOR_M - 10)
        for key in game_keys:
            name = f"detail_{key[0]}_{key[1]}.jpg"
            Image.fromarray(tiles.tiles[key]).save(imagery_dir / name, quality=92)
            layers.append({"file": name, "key": list(key)})
        cols = [k[0] for k in game_keys]
        rows = [k[1] for k in game_keys]
        tile_grid = {"tile_m": TILE_M, "pixels": tiles.pixels, "col0": min(cols), "row_top": max(rows),
                     "ncols": max(cols) - min(cols) + 1, "nrows": max(rows) - min(rows) + 1,
                     # Local x of the west edge and local z of the north edge of the grid.
                     "x0": min(cols) * TILE_M - origin[0], "z0": origin[1] - (max(rows) + 1) * TILE_M}
    overview_spec = manifest["imagery"]["overview"]
    overview_product = sources.Product("imagery-overview", cache)
    overview = mosaic(overview_spec, overview_product, crs, domain, max_pixels=4096)
    new_locks["imagery_overview"] = sources.check_lock(overview_product, locks.get("imagery_overview"), update_lock)
    Image.fromarray(overview["image"]).save(imagery_dir / "overview.jpg", quality=90)
    horizon_spec = manifest["imagery"]["horizon"]
    horizon_product = sources.Product("imagery-horizon", cache)
    himage = mosaic(horizon_spec, horizon_product, crs, hdomain, max_pixels=2048)
    new_locks["imagery_horizon"] = sources.check_lock(horizon_product, locks.get("imagery_horizon"), update_lock)
    Image.fromarray(himage["image"]).save(imagery_dir / "horizon.jpg", quality=88)

    def local_bounds(b):
        return {"x0": b[0] - origin[0], "z0": origin[1] - b[3], "x1": b[2] - origin[0], "z1": origin[1] - b[1]}

    # Orthophotos are exposure-adjusted radiance, not albedo. One gain for the whole
    # image maps the imaged pavement to a typical aged asphalt albedo, keeping every
    # colour relation in the photograph.
    if tiles is not None:
        pavement_rgb = tiles.sample(points)
    else:
        col = (points[:, 0] - domain[0]) / overview["resolution"] - 0.5
        row = (domain[3] - points[:, 1]) / overview["resolution"] - 0.5
        pavement_rgb = np.stack([map_coordinates(overview["image"][:, :, b].astype(float), [row, col], order=1)
                                 for b in range(3)], axis=1) / 255.0
    linear = np.where(pavement_rgb <= 0.04045, pavement_rgb / 12.92, ((pavement_rgb + 0.055) / 1.055) ** 2.4)
    pavement_linear = float(np.median(linear @ np.array([0.2126, 0.7152, 0.0722])))
    albedo_gain = ASPHALT_ALBEDO / pavement_linear
    imagery = {"schema_version": 1, "track_sha256": digest(track_path),
               "albedo_gain": round(albedo_gain, 4), "imaged_pavement_linear_luminance": round(pavement_linear, 4),
               "target_asphalt_albedo": ASPHALT_ALBEDO,
               "detail": {"grid": tile_grid, "layers": layers,
               "resolution_m": tiles.resolution if tiles is not None else overview["resolution"]},
               "overview": {"file": "overview.jpg", "bounds": local_bounds(domain), "resolution_m": overview["resolution"]},
               "horizon": {"file": "horizon.jpg", "bounds": local_bounds(hdomain), "resolution_m": himage["resolution"]},
               "attribution": {k: manifest["imagery"][k]["attribution"] for k in ("detail", "overview", "horizon")
                               if manifest["imagery"].get(k)}}
    (generated / "imagery.json").write_text(json.dumps(imagery, indent=1) + "\n")

    if update_lock or locks != {**locks, **new_locks}:
        manifest["locks"] = new_locks
        manifest["osm"]["snapshot_sha256"] = snapshot_hash
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    summary = {**report, "samples": count, "width_m": track["metadata"]["width"]["range_m"],
               "width_valid_fraction": width_meta["valid_fraction"], "curb_runs": len(curbs),
               "height_range_m": [float(heights.min()), float(heights.max())],
               "max_bank_deg": float(np.degrees(np.abs(banks)).max()), "detail_tiles": len(layers),
               "terrain_grid": [nx, nz], "surface_triangles": len(triangles)}
    (generated / "build-report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def mosaic(spec, product, crs, bounds, max_pixels):
    """A single north-up image over bounds, built from 1024 pixel requests."""
    resolution = spec["resolution_m"]
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    resolution = max(resolution, span / max_pixels)
    chunk = 1024 * resolution
    width = round((bounds[2] - bounds[0]) / resolution)
    height = round((bounds[3] - bounds[1]) / resolution)
    image = np.zeros((height, width, 3), np.uint8)
    y_top = bounds[3]
    for r in range(math.ceil(height / 1024)):
        for c in range(math.ceil(width / 1024)):
            b = (bounds[0] + c * chunk, y_top - (r + 1) * chunk, bounds[0] + (c + 1) * chunk, y_top - r * chunk)
            tile = sources.imagery_tile(spec["source"], product, crs, b, 1024, f"{resolution:g}-{r}-{c}")
            h = min(1024, height - r * 1024)
            w = min(1024, width - c * 1024)
            image[r * 1024:r * 1024 + h, c * 1024:c * 1024 + w] = tile[:h, :w]
    return {"image": image, "resolution": resolution}


# Discovery and inspection -------------------------------------------------------

def discover(circuit: str) -> None:
    """Propose closed raceway routes near the manifest centre, nearest the official length."""
    manifest, _ = load_manifest(circuit)
    lon, lat = manifest["center_lonlat"]
    d = 0.012
    url = f"https://api.openstreetmap.org/api/0.6/map.json?bbox={lon - d / math.cos(math.radians(lat))},{lat - d},{lon + d / math.cos(math.radians(lat))},{lat + d}"
    document = json.loads(sources.http(url))
    nodes = {e["id"]: e for e in document["elements"] if e["type"] == "node"}
    ways = [e for e in document["elements"] if e["type"] == "way" and e.get("tags", {}).get("highway") == "raceway"
            and e.get("tags", {}).get("sport", "motor") != "karting"
            and "pit" not in e.get("tags", {}).get("name", "").lower()
            and e.get("tags", {}).get("raceway") not in ("pit_lane", "pitlane")
            and all(n in nodes for n in e["nodes"])]
    crs = manifest["crs"]
    xy = {}
    for n in nodes.values():
        xy[n["id"]] = project(crs, n["lon"], n["lat"])
    usage = {}
    for w in ways:
        for n in w["nodes"]:
            usage[n] = usage.get(n, 0) + 1
    edges = []  # (from, to, way, version, length)
    for w in ways:
        ids = w["nodes"]
        cuts = [0] + [k for k in range(1, len(ids) - 1) if usage[ids[k]] > 1] + [len(ids) - 1]
        oneway = w.get("tags", {}).get("oneway") in ("yes", "1", "true")
        for a, b in zip(cuts, cuts[1:]):
            part = ids[a:b + 1]
            length = sum(math.dist(xy[p], xy[q]) for p, q in zip(part, part[1:]))
            edges.append((part[0], part[-1], w["id"], w["version"], length))
            if not oneway:
                edges.append((part[-1], part[0], w["id"], w["version"], length))
    graph = {}
    for e in edges:
        graph.setdefault(e[0], []).append(e)
    rings = []
    for w in ways:
        ids = w["nodes"]
        if ids[0] == ids[-1] and all(usage[n] == 1 or n == ids[0] for n in ids[1:-1]):
            length = sum(math.dist(xy[p], xy[q]) for p, q in zip(ids, ids[1:]))
            rings.append((length, [(ids[0], ids[0], w["id"], w["version"], length)]))
    official = manifest["official_length_m"]
    cycles = []

    def walk(start, node, path, length, seen):
        if length > official * 1.3 or len(cycles) > 20000:
            return
        for e in graph.get(node, []):
            if e[1] == start and path:
                cycles.append((length + e[4], path + [e]))
            elif e[1] not in seen and e[1] != start:
                walk(start, e[1], path + [e], length + e[4], seen | {e[1]})

    for start in sorted(graph):
        walk(start, start, [], 0.0, {start})
    unique = {}
    for length, path in cycles:
        key = frozenset((e[0], e[1], e[2]) for e in path)
        unique[key] = (length, path)
    ranked = sorted([*unique.values(), *rings], key=lambda row: abs(row[0] - official))[:6]
    for length, path in ranked:
        route = [{"way": e[2], "version": e[3], "from": e[0], "to": e[1]} for e in path]
        by_id = {w["id"]: w for w in ways}
        # Direction from every node of the route, the same walk the build performs.
        pts = np.array([xy[n] for part in route for n in way_chain(part, by_id[part["way"]]["nodes"])[:-1]])
        area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
        names = sorted({w.get("tags", {}).get("name", "") for w in ways if w["id"] in {e[2] for e in path}})
        print(json.dumps({"length_m": round(length, 1), "error": round((length - official) / official, 4),
                          "direction": "counterclockwise" if area > 0 else "clockwise",
                          "names": names, "route": route}))


def preview(circuit: str, station: float, span: float, output: Path | None) -> None:
    """Orthoimagery around a station with the built centerline and edges drawn on it."""
    manifest, _ = load_manifest(circuit)
    track = json.loads((TRACKS / circuit / "track.json").read_text())
    imagery = json.loads((TRACKS / circuit / "generated/imagery.json").read_text())
    origin = track["origin"]
    grid = imagery["detail"]["grid"]
    samples = track["samples"]
    s = np.array([r["s"] for r in samples])
    k = int(np.argmin(np.abs(s - station)))
    cx, cz = samples[k]["p"][0], samples[k]["p"][2]
    gen = TRACKS / circuit / "generated/imagery"
    if not imagery["detail"]["layers"]:
        # Satellite only circuits: crop the overview, drawn at a minimum of 1 m per
        # pixel so the overlay stays legible over 10 m source pixels.
        res = min(1.0, imagery["overview"]["resolution_m"])
        pixels = round(span / res)
        overview = Image.open(gen / imagery["overview"]["file"])
        bounds = imagery["overview"]["bounds"]
        scale = imagery["overview"]["resolution_m"]
        box = ((cx - span / 2 - bounds["x0"]) / scale, (cz - span / 2 - bounds["z0"]) / scale,
               (cx + span / 2 - bounds["x0"]) / scale, (cz + span / 2 - bounds["z0"]) / scale)
        canvas = overview.resize((pixels, pixels), Image.Resampling.BILINEAR, box=box)
        grid = None
    else:
        res = imagery["detail"]["resolution_m"]
        pixels = round(span / res)
        canvas = Image.new("RGB", (pixels, pixels))
    for layer in imagery["detail"]["layers"]:
        col, row = layer["key"]
        tx = col * TILE_M - origin["easting"]
        tz = origin["northing"] - (row + 1) * TILE_M
        px, pz = round((tx - (cx - span / 2)) / res), round((tz - (cz - span / 2)) / res)
        if -grid["pixels"] < px < pixels and -grid["pixels"] < pz < pixels:
            canvas.paste(Image.open(gen / layer["file"]), (px, pz))
    draw = ImageDraw.Draw(canvas)

    def to_px(x, z):
        return ((x - (cx - span / 2)) / res, (z - (cz - span / 2)) / res)

    for sign, colour in ((0, (255, 255, 0)), (1, (0, 255, 255)), (-1, (255, 0, 255))):
        line = []
        for r in samples:
            if abs(r["p"][0] - cx) > span or abs(r["p"][2] - cz) > span:
                continue
            off = sign * r["width"] / 2
            line.append(to_px(r["p"][0] + r["left"][0] * off, r["p"][2] + r["left"][2] * off))
        for a, b in zip(line, line[1:]):
            if math.dist(a, b) < 40 / res:
                draw.line([a, b], fill=colour, width=2)
    for r in samples[::10]:
        if abs(r["p"][0] - cx) < span / 2 and abs(r["p"][2] - cz) < span / 2:
            x, y = to_px(r["p"][0], r["p"][2])
            draw.text((x + 4, y + 4), f"{r['s']:.0f}", fill=(255, 255, 255))
    lon, lat = Transformer.from_crs(manifest["crs"], "EPSG:4326", always_xy=True).transform(
        origin["easting"] + cx, origin["northing"] - cz)
    print(json.dumps({"station_m": samples[k]["s"], "lonlat": [round(lon, 7), round(lat, 7)]}))
    output = output or ROOT / f"artifacts/circuits/{circuit}/preview-{station:.0f}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)
    print(output)


def views(circuit: str, stations: list[float], godot: str, output: Path | None, workers: int) -> None:
    """Rendered rider (camera 1) and chase (camera 0) screenshots from the real game."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    output = output or ROOT / f"artifacts/circuits/{circuit}/views"
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(s, c) for s in stations for c in (1, 0)]

    def shoot(job):
        station, camera = job
        name = output / f"{circuit}-{station:.0f}-{'rider' if camera == 1 else 'chase'}.png"
        command = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", godot, "--path", str(ROOT / "godot"),
                   "--resolution", "1600x1000", "--", f"--track={circuit}", "--preview",
                   f"--preview-station={station}", f"--preview-camera={camera}", f"--screenshot={name}"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=600)
        ok = f"SCREENSHOT {name} result=0" in run.stdout
        errors = [line for line in (run.stdout + run.stderr).splitlines() if "ERROR" in line]
        return {"file": str(name), "ok": ok, "errors": errors[:3]}

    with ThreadPoolExecutor(max(1, min(workers, 4))) as pool:
        for result in pool.map(shoot, jobs):
            print(json.dumps(result))


def source_row(spec: dict | None) -> str:
    if not spec:
        return "none"
    return f"{spec['name']}, {spec['resolution_m']:g} m, {spec['license']}"


def write_docs(output: Path) -> None:
    """Regenerate the per-circuit source and build tables from manifests and track.json."""
    rows, results, details = [], [], []
    verification = record_verification()
    manifests = [json.loads(p.read_text()) for p in sorted(MANIFESTS.glob("*.json"))]
    unbuilt = [m["id"] for m in manifests if not (TRACKS / m["id"] / "track.json").exists()]
    incomplete = [m["id"] for m in manifests if not all(m.get(k) for k in ("imagery", "elevation", "osm", "layout"))]
    manifests = [m for m in manifests if m["id"] not in incomplete]
    for m in sorted(manifests, key=lambda m: m.get("motogp_round", 99)):
        imagery = m["imagery"]
        rows.append(f"| {m.get('motogp_round', 'test')} | `{m['id']}`{' (held out)' if m.get('purpose') else ''} | {m['name']}, {m['country']} | {m['layout']} | "
                    f"[{m['official_length_m']} m]({m['official_length_source']}) | "
                    f"{source_row(imagery.get('detail'))} | {source_row(imagery.get('overview'))} | "
                    f"{source_row(m['elevation']['detail'])} |")
        track_path = TRACKS / m["id"] / "track.json"
        if track_path.exists():
            meta = json.loads(track_path.read_text())["metadata"]
            track = json.loads(track_path.read_text())
            width = meta["width"]
            results.append(f"| `{m['id']}` | {m['official_length_m']} | {track['length_m']:.1f} | "
                           f"{meta.get('ground_length_m', 0):.1f} | {100 * meta.get('ground_length_error', meta['length_error']):.2f}% | "
                           f"{m['direction']} | {width['range_m'][0]:.1f} to {width['range_m'][1]:.1f} | "
                           f"{100 * width['valid_fraction']:.0f}% | {meta['curb_runs']} | "
                           f"{meta['elevation'].get('bank', '')[:40]} | {lap_cell(verification.get(m['id']))} |")
        lines = [f"### {m['name']} (`{m['id']}`)", "",
                 *([f"- **Purpose: {m['purpose']}**"] if m.get("purpose") else []),
                 f"- Layout: {m['layout']}" + (f" ([source]({m['layout_source']}))" if m.get("layout_source") else ""),
                 *([f"- Length gate: {m['length_reference']['length_m']} m, {m['length_reference']['layout']} "
                    f"([source]({m['length_reference']['source']})), not the official length. "
                    f"{m['length_reference']['reason']}"] if m.get("length_reference") else []),
                 f"- Direction: {m['direction']}" + (
                     (f" ([source]({m['direction_source']}))" if m["direction_source"].startswith("http")
                      else f" ({m['direction_source']})") if m.get("direction_source") else ""),
                 f"- Official lap: {m['official_length_m']} m, published width {m['width_m']['published']} m ({m['width_m']['source']})",
                 f"- Start line: {m['start_finish']['lonlat']}, {m['start_finish']['source']}",
                 "- OSM route (ODbL 1.0, OpenStreetMap contributors): " + ", ".join(
                     f"[{r['way']}](https://www.openstreetmap.org/way/{r['way']}) v{r['version']}" for r in m["osm"]["route"]),
                 f"- OSM node snapshot SHA256: `{m['osm'].get('snapshot_sha256', 'unpinned')}`"]
        for group in ("imagery", "elevation"):
            for level, spec in m[group].items():
                if not spec:
                    continue
                src = spec["source"]
                where = src.get("url") or ", ".join(src.get("urls", []))
                layer = src.get("layer") or src.get("coverage") or ""
                lines.append(f"- {group.capitalize()} {level}: {spec['name']}, {spec['resolution_m']:g} m, "
                             f"`{src['kind']}` {where} {('layer `' + str(layer) + '`') if layer else ''}. "
                             f"Licence: [{spec['license']}]({spec.get('license_url', '')}). Attribution: \"{spec['attribution']}\"")
        lines.append("- Response locks: " + ", ".join(f"{k} `{v[:16]}…`" for k, v in m.get("locks", {}).items()))
        for note in m.get("notes", []):
            lines.append(f"- Note: {note}")
        details.append("\n".join(lines))
    text = "\n".join([
        "# MotoGP circuits",
        "",
        "Generated by `uv run tools/build_circuit.py docs` from `data/circuits/*.json` and the built",
        "`godot/tracks/<id>/track.json`. Do not edit by hand; edit the manifests and regenerate.",
        "",
        "Build a circuit with `uv run tools/build_circuit.py <id>`: it fetches the pinned OSM ways,",
        "orthoimagery and elevation, verifies every response against the manifest's SHA256 locks, and",
        "writes the committed `godot/tracks/<id>/track.json` plus the ignored `generated/` meshes and",
        "imagery. Select it with `--track=<id>` in the game, `ThunderhillSACEnv(track=<id>)`,",
        "`RoadTelemetry(track=<id>)`, `tools/drive_lap.py --track <id>` and `sac_async.py --track <id>`.",
        "Thunderhill East (`thunderhill-east`) stays the default and keeps its own pipeline",
        "(docs/geometry-sources.md).",
        "",
        "## Method",
        "",
        "- Layout: the OSM `highway=raceway` ways of the layout MotoGP uses, chained in race order",
        "  (`build_circuit.py discover` proposes closed routes nearest the official length). Nodes are",
        "  projected into the circuit's metric CRS, resampled at 0.5 m, Gaussian smoothed with a bounded",
        "  displacement and sampled every 3 m, as `build_track.py` does for Thunderhill.",
        "- Width and centre: where detail orthoimagery exists, each 3 m sample is measured across the",
        "  road from pavement colour (asphalt characterised from pixels near the OSM line), accepting",
        "  widths from 1.5 m below to 3.5 m above the published width; the line is recentred on the",
        "  imaged pavement. Without detail imagery the published width is used on the OSM line.",
        "- Kerbs: painted kerbs are detected as a band just outside the measured edge that alternates",
        "  saturated colour and white. Their 3D profile reuses Thunderhill's provisional curb shape.",
        "- Elevation: road height from the DEM along the line, median and Gaussian smoothed; bank from",
        "  the DEM cross slope only where the DEM is 5 m or finer and not rounded to whole metres.",
        "- Terrain: the DEM on an 8 m grid over the track bounds plus 260 m, joined to the road by the",
        "  same road conforming mesh builders Thunderhill uses; a horizon DEM extends 4 km further.",
        "- Imagery: 256 m detail tiles along a 60 m corridor, a domain overview and a coarse horizon",
        "  image, all georeferenced in the circuit CRS. One exposure gain per circuit maps the imaged",
        "  pavement to an aged asphalt albedo of 0.12, keeping the photograph's colour relations.",
        "- Start line: identified in the imagery (front of the painted grid) where it is visible.",
        "",
        "Limits: no buildings, grandstands, walls, barriers or trees are modelled (they appear only as",
        "flat imagery); run-off grip is not modelled; Sentinel-2 circuits (10 m pixels) have no",
        "imaged pavement and use a generic CC0 asphalt texture. Imagery tiles are regenerated by the",
        "build command and are not committed (about 10 to 30 MB per circuit).",
        "",
        "## Calendar and sources",
        "",
        "Calendar: the 2026 MotoGP season as published by motogp.com (Qatar moved to 6 to 8 November).",
        "Held-out test tracks (marked \"held out\", never used to train the general SAC policy) measure",
        "generalization: `portimao` is built from the layout OSM holds, which matches the 4.653 km car",
        "Grand Prix circuit rather than the verified 4.592 km MotoGP layout; `laguna-seca` is not a",
        "MotoGP circuit.",
        "",
        "| Round | Id | Circuit | Layout | Official lap | Detail imagery | Overview imagery | Elevation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        *([f"Manifests still incomplete: {', '.join(incomplete)}.", ""] if incomplete else []),
        *([f"Not built (see the notes in their provenance sections): {', '.join(unbuilt)}.", ""] if unbuilt else []),
        "## Build results",
        "",
        "Length is the game's projected centerline; ground length removes the projection scale factor",
        "and adds climb. Width is measured across the road in the detail imagery where it exists.",
        "",
        "Scripted lap: `tools/drive_lap.py --track <id>` (privileged path follower capped at 18 m/s, a",
        "QA driver, not a racing lap time), recorded in `data/circuit-verification.json`.",
        "",
        "| Id | Official m | Game m | Ground m | Ground error (vs length gate) | Direction | Width m | Width measured | Kerb runs | Bank | Scripted lap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *results,
        "",
        "## Per circuit provenance",
        "",
        "\n\n".join(details),
        "",
    ])
    output.write_text(text)
    print(output)
    write_attribution(manifests)


VERIFICATION = ROOT / "data/circuit-verification.json"


def record_verification() -> dict:
    """Merge the latest drive_lap summaries (artifacts/qa/<id>-lap) into verification.json."""
    data = json.loads(VERIFICATION.read_text()) if VERIFICATION.exists() else {}
    for path in sorted((ROOT / "artifacts/qa").glob("*-lap/summary.json")):
        summary = json.loads(path.read_text())
        circuit = summary.get("track")
        track_path = TRACKS / str(circuit) / "track.json"
        if not circuit or not track_path.exists():
            continue
        if summary.get("track_sha256", digest(track_path)) != digest(track_path):
            continue  # a lap driven on another track.json proves nothing about this one
        final = summary["final_observation"]
        data[circuit] = {"track_sha256": digest(track_path), "driver": summary["driver"],
                         "success": summary["success"], "reason": summary["reason"],
                         "sim_time_s": round(summary["sim_time_s"], 2),
                         "legal_distance_m": round(final["track"]["legal_distance"], 1),
                         "max_lateral_m": round(summary["max_lateral_m"], 2),
                         "max_target_speed_m_s": summary["max_target_speed_m_s"]}
    VERIFICATION.write_text(json.dumps(dict(sorted(data.items())), indent=1) + "\n")
    return data


def lap_cell(row: dict | None) -> str:
    if not row:
        return "not run"
    status = "complete, valid" if row["success"] else row["reason"]
    return f"{status}, {row['sim_time_s']:.1f} s"


ATTRIBUTION = ROOT / "godot/assets/ATTRIBUTION.md"
BEGIN, END = "<!-- motogp-circuits:begin (tools/build_circuit.py docs) -->", "<!-- motogp-circuits:end -->"


def write_attribution(manifests: list[dict]) -> None:
    """Regenerate the MotoGP circuit block of godot/assets/ATTRIBUTION.md."""
    lines = [BEGIN, "", "## MotoGP circuits", "",
             "`godot/tracks/<id>/track.json` and the generated meshes under `godot/tracks/<id>/generated/` are",
             "derived geographic databases under ODbL 1.0 (OpenStreetMap contributors,",
             "https://www.openstreetmap.org/copyright), separate from the MIT game code. The orthoimagery",
             "tiles and elevation in `generated/` are not committed; `tools/build_circuit.py` fetches them",
             "from the sources below, whose licences and attributions apply. Full provenance, pinned URLs",
             "and response hashes: docs/motogp-circuits.md. The project is not affiliated with MotoGP,",
             "Dorna or any circuit.", ""]
    for m in sorted(manifests, key=lambda m: m.get("motogp_round", 99)):
        credits = []
        for group in ("imagery", "elevation"):
            for spec in m[group].values():
                if spec and (spec["attribution"], spec["license"]) not in [(c[0], c[1]) for c in credits]:
                    credits.append((spec["attribution"], spec["license"], spec.get("license_url", "")))
        lines.append(f"* `{m['id']}` ({m['name']}): OpenStreetMap contributors (ODbL 1.0); " + "; ".join(
            f"{a} ({lic}{', ' + url if url else ''})" for a, lic, url in credits) + ".")
    lines += ["", END]
    text = ATTRIBUTION.read_text()
    block = "\n".join(lines)
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    ATTRIBUTION.write_text(text)
    print(ATTRIBUTION)


def list_circuits() -> None:
    for path in sorted(MANIFESTS.glob("*.json")):
        manifest = json.loads(path.read_text())
        track = TRACKS / manifest["id"] / "track.json"
        generated = TRACKS / manifest["id"] / "generated/imagery.json"
        print(f"{manifest['id']:<16} {manifest['name']:<48} track={'yes' if track.exists() else 'no'} "
              f"generated={'yes' if generated.exists() else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", help="circuit id, or discover / preview / list")
    parser.add_argument("circuit", nargs="?")
    parser.add_argument("--update-lock", action="store_true")
    parser.add_argument("--station", type=float, default=0.0)
    parser.add_argument("--span", type=float, default=200.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stations", type=float, nargs="+", default=[0.0])
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("godot4"))
    parser.add_argument("--workers", type=int, default=2, help="parallel Godot renders (at most 4)")
    args = parser.parse_args()
    if args.command == "docs":
        write_docs(args.output or ROOT / "docs/motogp-circuits.md")
    elif args.command == "list":
        list_circuits()
    elif args.command == "discover":
        discover(args.circuit)
    elif args.command == "views":
        if not args.godot:
            parser.error("Provide --godot")
        views(args.circuit, args.stations, args.godot, args.output, args.workers)
    elif args.command == "preview":
        preview(args.circuit, args.station, args.span, args.output)
    else:
        build(args.command, args.update_lock)


if __name__ == "__main__":
    main()
