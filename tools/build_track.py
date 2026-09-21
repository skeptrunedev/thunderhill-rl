# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "rasterio==1.4.4", "shapely==2.1.2"]
# ///
"""Build the playable historical East circuit from acquired reference geometry.

Run uv run tools/build_track.py after fetch_geometry.py and measure_lidar.py.
Generated geographic databases retain ODbL, separately from MIT source code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from apply_pavement_envelope import apply_envelope
from build_surface_mesh import build as build_surface_mesh
from build_surface_mesh import raw_terrain
from pyproj import Transformer
from pyproj.enums import TransformDirection
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from terrain_datum import terrain_transform

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "godot/data"
ROAD_SPACING = 3.0
TERRAIN_SPACING = 8.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample(points: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, float]:
    closed = np.vstack([points, points[0]])
    station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    output = np.column_stack([np.interp(distances % station[-1], station, closed[:, axis]) for axis in range(points.shape[1])])
    return output, float(station[-1])


def normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    forward /= np.linalg.norm(forward, axis=1)[:, None]
    left = np.column_stack([-forward[:, 1], forward[:, 0]])
    return forward, left


def read_route() -> tuple[np.ndarray, dict]:
    raw_path = ROOT / "artifacts/reference/geometry/east-osm.json"
    source = load_json(raw_path)
    manifest = load_json(ROOT / "data/reference/geometry-sources.json")
    way = next(row for row in source["elements"] if row["type"] == "way")
    nodes = {row["id"]: row for row in source["elements"] if row["type"] == "node"}
    snapshot = [[key, nodes[key]["version"], nodes[key]["lon"], nodes[key]["lat"]] for key in way["nodes"]]
    actual_hash = hashlib.sha256(json.dumps(snapshot, separators=(",", ":")).encode()).hexdigest()
    assert actual_hash == manifest["osm"]["ordered_node_snapshot_sha256"], "OSM source changed"
    assert way["nodes"][0] == way["nodes"][-1], "Expected closed circuit"
    coordinates = np.array([[nodes[key]["lon"], nodes[key]["lat"]] for key in way["nodes"][:-1]])
    transform = Transformer.from_crs("EPSG:4326", "EPSG:6339", always_xy=True)
    x, y = transform.transform(coordinates[:, 0], coordinates[:, 1])
    return np.column_stack([x, y]), {"osm_way": way["id"], "osm_way_version": way["version"], "ordered_node_snapshot_sha256": actual_hash}


def reviewed_widths(stations: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, list]:
    source = load_json(ROOT / "data/reference/ortho-measurements.json")
    anchors = []
    for section in source["sections"]:
        if "selected_width_m" not in section:
            continue
        matching = [c for c in section["candidates"] if abs(c["width_m"] - section["selected_width_m"]) < 0.01]
        candidate = min(matching, key=lambda c: abs(c["threshold"] - 0.07))
        anchors.append({"id": section["id"], "s": section["station_m_from_osm_first_node"], "width": section["selected_width_m"], "center_offset_left_m": (candidate["left_offset_m"] + candidate["right_offset_m"]) * 0.5, "minimum_interpretive_uncertainty_m": section["minimum_interpretive_uncertainty_m"]})
    assert len(anchors) == 8, "Expected the eight reviewed historical width anchors"
    ss = np.array([row["s"] for row in anchors])
    widths = np.interp(stations, ss, [row["width"] for row in anchors], period=length)
    offsets = np.interp(stations, ss, [row["center_offset_left_m"] for row in anchors], period=length)
    return widths, offsets, anchors


def road_elevation(points: np.ndarray, heading: np.ndarray, left: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    path = ROOT / "artifacts/reference/lidar/road-ground-points.npz"
    data = np.load(path)["points"]
    tree = cKDTree(data[:, :2])
    height, bank, errors, counts = [], [], [], []
    for center, along, across in zip(points, heading, left):
        nearby = data[tree.query_ball_point(center, 5.5)]
        offsets = nearby[:, :2] - center
        t, n = offsets @ along, offsets @ across
        valid = (np.abs(t) <= 2.0) & (np.abs(n) <= 3.0)
        assert valid.sum() >= 30, f"Insufficient raw ground observations near {center}"
        design = np.column_stack([np.ones(valid.sum()), t[valid], n[valid]])
        observed = nearby[valid, 2]
        fit = np.linalg.lstsq(design, observed, rcond=None)[0]
        height.append(fit[0])
        bank.append(np.arctan(fit[2]))
        errors.append(float(np.sqrt(np.mean((observed - design @ fit) ** 2))))
        counts.append(int(valid.sum()))
    heights, banks = np.array(height), np.array(bank)
    # Remove centimeter sample noise without replacing the measured profile.
    filtered_heights = gaussian_filter1d(heights, sigma=0.65, mode="wrap")
    filtered_banks = gaussian_filter1d(banks, sigma=0.8, mode="wrap")
    return filtered_heights, filtered_banks, {"source": "2023 class 2 nonwithheld USGS lidar", "corridor_sha256": digest(path), "fit_strip_m": [4.0, 6.0], "minimum_fit_point_count": min(counts), "maximum_fit_rmse_m": max(errors), "median_fit_rmse_m": float(np.median(errors)), "height_smoothing_max_change_m": float(np.max(np.abs(filtered_heights - heights))), "local_fit_error_is_not_absolute_accuracy": True}


def build_terrain(origin: np.ndarray, road_xyz: np.ndarray, road_left: np.ndarray, widths: np.ndarray, banks: np.ndarray) -> dict:
    source_path = ROOT / "artifacts/reference/geometry/east-terrain-1m.tif"
    transform, _ = terrain_transform()
    with rasterio.open(source_path) as src:
        if src.crs.to_epsg() != 26910:
            raise ValueError("Unexpected DEM coordinate reference system")
        # Leave a full source pixel margin, then transform the domain to the
        # local road datum before selecting the terrain grid.
        east, north = transform.transform(
            [src.bounds.left + 1, src.bounds.right - 1] * 2,
            [src.bounds.bottom + 1] * 2 + [src.bounds.top - 1] * 2,
            direction=TransformDirection.INVERSE, errcheck=True,
        )
    x0 = np.ceil((min(east) - origin[0]) / TERRAIN_SPACING) * TERRAIN_SPACING
    x1 = np.floor((max(east) - origin[0]) / TERRAIN_SPACING) * TERRAIN_SPACING
    z0 = np.ceil((origin[1] - max(north)) / TERRAIN_SPACING) * TERRAIN_SPACING
    z1 = np.floor((origin[1] - min(north)) / TERRAIN_SPACING) * TERRAIN_SPACING
    template = {
        "schema_version": 1, "nx": round((x1-x0)/TERRAIN_SPACING)+1,
        "nz": round((z1-z0)/TERRAIN_SPACING)+1,
        "x0": float(x0), "z0": float(z0), "step": TERRAIN_SPACING,
    }
    return raw_terrain({"origin": {"easting": float(origin[0]),
                       "northing": float(origin[1]), "elevation_m": float(origin[2])}},
                       template, source_path)


def geometry_samples(points, widths, origin, source_stations):
    sample_count = len(points)
    heading, left = normals(points)
    heights, banks, fit_metadata = road_elevation(points, heading, left)
    xyz = np.column_stack([points[:, 0] - origin[0], heights - origin[2], origin[1] - points[:, 1]])
    left_xyz = np.column_stack([left[:, 0], np.zeros(sample_count), -left[:, 1]])
    tangent = np.roll(xyz, -1, axis=0) - np.roll(xyz, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    segments = np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1)
    stations = np.r_[0, np.cumsum(segments[:-1])]
    directions = heading
    turn = np.arctan2(directions[:, 0] * np.roll(directions[:, 1], -1) - directions[:, 1] * np.roll(directions[:, 0], -1), np.sum(directions * np.roll(directions, -1, axis=0), axis=1))
    curvature = turn / segments
    assert np.all(np.isfinite(xyz)) and np.all(np.isfinite(banks))
    assert segments.min() > 1.0 and segments.max() < 5.0
    assert widths.min() >= 10.0 and widths.max() <= 13.0
    assert np.max(np.abs(banks)) < np.radians(15)
    samples = [{"s": round(float(stations[i]), 3), "p": np.round(xyz[i], 4).tolist(), "tangent": np.round(tangent[i], 7).tolist(), "left": np.round(left_xyz[i], 7).tolist(), "width": round(float(widths[i]), 4), "bank": round(float(banks[i]), 7), "curvature": round(float(curvature[i]), 7), "source_s": round(float(source_stations[i]), 3)} for i in range(sample_count)]
    return samples, segments, heights, banks, xyz, left_xyz, fit_metadata


def apply_plan_study(track, study):
    """Apply pinned horizontal candidates without reinterpreting sample stations."""
    encoded = json.dumps(track, separators=(",", ":")) + "\n"
    if hashlib.sha256(encoded.encode()).hexdigest() != study["track_sha256"]:
        raise ValueError("Plan study baseline track changed")
    samples = track["samples"]
    stations = np.array([row["s"] for row in samples])
    points = np.array([row["p"] for row in samples], dtype=float)[:, [0, 2]]
    start, end = study["station_interval_m"]
    if not 0 <= start < end <= track["length_m"]:
        raise ValueError("Invalid plan study interval")
    selected = (stations >= start) & (stations <= end)
    candidate = study["candidate"]
    targets = np.asarray(candidate["candidate_xz_m"], dtype=float)
    if not np.array_equal(np.asarray(candidate["stations_m"]), stations[selected]):
        raise ValueError("Plan candidate stations differ from baseline samples")
    if targets.shape != points[selected].shape or not np.isfinite(targets).all():
        raise ValueError("Invalid plan candidate coordinates")
    displacement = np.zeros_like(points)
    displacement[selected] = targets - points[selected]
    maximum = float(np.linalg.norm(displacement, axis=1).max())
    recorded = float(candidate["maximum_center_displacement_m"])
    if not np.isfinite(recorded) or abs(maximum - recorded) > 1e-9:
        raise ValueError("Plan candidate displacement differs from reviewed report")
    return displacement, {
        "baseline_track_sha256": study["track_sha256"],
        "baseline_length_m": track["length_m"],
        "baseline_station_interval_m": [start, end],
        "maximum_center_displacement_m": maximum,
        "changed_samples": int(np.count_nonzero(np.linalg.norm(displacement, axis=1))),
        "status": "Provisional horizontal reconstruction; lidar height and bank refitted, station labels recomputed from resulting 3D geometry",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--pavement-envelope", type=Path, default=ROOT / "data/reference/pit-envelope.json")
    parser.add_argument("--plan-study", type=Path,
                        help="Apply a hash pinned horizontal study and refit lidar, then regenerate terrain interfaces")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source_points, osm_metadata = read_route()
    closed = np.vstack([source_points, source_points[0]])
    source_length = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
    dense_stations = np.arange(0, source_length, 0.5)
    dense, _ = resample(source_points, dense_stations)
    sigma = 2.0
    while True:
        smooth = gaussian_filter1d(dense, sigma=sigma, axis=0, mode="wrap")
        max_displacement = float(np.linalg.norm(smooth - dense, axis=1).max())
        if max_displacement <= 0.35:
            break
        sigma *= 0.8
    # Sample the bounded source curve at approximately three meter intervals.
    sample_count = int(np.ceil(source_length / ROAD_SPACING))
    source_stations = np.linspace(0, source_length, sample_count, endpoint=False)
    points, _smoothed_length = resample(smooth, source_stations * (float(np.linalg.norm(np.diff(np.vstack([smooth, smooth[0]]), axis=0), axis=1).sum()) / source_length))
    _heading, left = normals(points)
    widths, center_offsets, anchors = reviewed_widths(source_stations, source_length)
    points += left * center_offsets[:, None]
    origin = np.array([source_points[0, 0], source_points[0, 1], 90.0])
    samples, segments, heights, banks, xyz, left_xyz, fit_metadata = geometry_samples(points, widths, origin, source_stations)
    track = {"schema_version": 1, "name": "Thunderhill East", "closed": True, "length_m": round(float(segments.sum()), 3), "origin": {"easting": float(origin[0]), "northing": float(origin[1]), "elevation_m": float(origin[2])}, "coordinates": "x east, z south, y up; meters", "bank_convention": "Positive bank raises the geometric left road edge relative to center", "samples": samples, "terrain_file": "res://data/terrain.json", "metadata": {"osm": osm_metadata, "historical_geometry_year": 2023, "route_configuration": "Angular hill branch, consistent with Cyclone, not Hill Bypass", "width_anchors": anchors, "width_and_center_status": "Provisional periodic linear interpolation of eight reviewed 2022 aerial pavement envelopes, not 2026 surveyed edges", "source_polyline_length_m": source_length, "plan_smoothing_max_displacement_m": max_displacement, "plan_smoothing_sigma_m": sigma * 0.5, "lidar": fit_metadata, "license": "Open Database License 1.0 (ODbL), separate from repository MIT source code", "attribution": "OpenStreetMap contributors; USGS 3DEP; USDA NAIP", "osm_license_url": "https://www.openstreetmap.org/copyright", "limitations": ["2023 lidar and 2022 width evidence predate repave", "Width and center interpolation between reviewed sections is provisional", "Curbs, current asphalt boundaries and tire grip are not surveyed by this data", "Local lidar plane fits and spatial smoothing do not model pavement microtexture"]}}
    envelope = load_json(args.pavement_envelope)
    corrected_xz, widths, correction = apply_envelope(track, envelope)
    baseline_xz = np.array([s["p"] for s in track["samples"]])[:, [0, 2]]
    corrected_points = points + (corrected_xz-baseline_xz)*[1, -1]
    samples, segments, heights, banks, xyz, left_xyz, fit_metadata = geometry_samples(corrected_points, widths, origin, source_stations)
    track["samples"] = samples
    track["length_m"] = round(float(segments.sum()), 3)
    track["metadata"]["lidar"] = fit_metadata
    track["metadata"]["pavement_envelope"] = {"source_sha256": digest(args.pavement_envelope), "baseline_track_sha256": envelope["baseline_track_sha256"], **correction}
    track["metadata"]["width_and_center_status"] = "Historical aerial and lidar pit envelope applied to the provisional eight section baseline; unresolved apron and paint interpretation retained in source manifest"
    if args.plan_study is not None:
        displacement, plan_metadata = apply_plan_study(track, load_json(args.plan_study))
        corrected_points += displacement * [1, -1]
        samples, segments, heights, banks, xyz, left_xyz, fit_metadata = geometry_samples(corrected_points, widths, origin, source_stations)
        track["samples"] = samples
        track["length_m"] = round(float(segments.sum()), 3)
        track["metadata"]["lidar"] = fit_metadata
        track["metadata"]["plan_study"] = {"source_sha256": digest(args.plan_study), **plan_metadata}
    terrain = build_terrain(origin, xyz, left_xyz, widths, banks)
    (output / "track.json").write_text(json.dumps(track, separators=(",", ":")) + "\n")
    (output / "terrain.json").write_text(json.dumps(terrain, separators=(",", ":")) + "\n")
    build_surface_mesh(output, output / "track.json", output / "terrain.json")
    print(json.dumps({"samples": sample_count, "length_m": track["length_m"], "segment_m": [float(segments.min()), float(segments.max())], "height_m": [float(heights.min()), float(heights.max())], "width_m": [float(widths.min()), float(widths.max())], "maximum_bank_degrees": float(np.degrees(np.abs(banks)).max()), "plan_smoothing_max_displacement_m": max_displacement, "lidar_fit": fit_metadata, "terrain_grid": [terrain["nx"], terrain["nz"]]}, indent=2))


if __name__ == "__main__":
    main()
