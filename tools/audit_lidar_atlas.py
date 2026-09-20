# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "matplotlib==3.10.8", "shapely==2.1.2", "rasterio==1.4.4"]
# ///
"""Audit full circuit atlas coverage, curvature and Godot parity inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_road_alignment import fit_height_graph
from build_lidar_atlas import HeightAtlas
from build_road_surface import evaluate
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas", type=Path, default=ROOT / "artifacts/road-surface/lidar-atlas.json"
    )
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/road-surface/lidar-atlas-circuit.json",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=ROOT / "data/reference/pavement-exclusions.json",
    )
    args = parser.parse_args()
    if not np.isfinite(args.step) or args.step <= 0:
        raise ValueError("Step must be finite and positive")
    data = json.loads(args.atlas.read_text())
    track_path = ROOT / "godot/data/track.json"
    road_path = ROOT / "artifacts/road-surface/road-surface.json"
    lidar_path = ROOT / "artifacts/reference/lidar/road-ground-points.npz"
    for name, path in [
        ("track", track_path),
        ("road", road_path),
        ("lidar", lidar_path),
    ]:
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != data["metadata"]["provenance_sha256"][name]
        ):
            raise ValueError(f"Stale atlas source: {name}")
    if data["metadata"]["diagnostic_limit"]:
        raise ValueError("Diagnostic partial atlas is not a circuit")
    track = json.loads(track_path.read_text())
    road = json.loads(road_path.read_text())
    origin = track["origin"]
    stations = np.array([s["s"] for s in track["samples"]] + [track["length_m"]])
    widths = np.array(
        [s["width"] for s in track["samples"]] + [track["samples"][0]["width"]]
    )
    entries = []
    points = []
    directions = []
    old_heights = []
    for s in np.arange(0, track["length_m"], args.step):
        width = np.interp(s, stations, widths)
        for u in np.linspace(-width / 2, width / 2, 7):
            r = evaluate(road, float(s), float(u))
            points.append(r["R"][[0, 2]])
            d = r["Rs"][[0, 2]]
            directions.append(d / np.linalg.norm(d))
            old_heights.append(r["R"][1] - road["road_lift_m"])
            entries.append({"station_m": float(s), "lateral_m": float(u)})
    points = np.array(points)
    d = np.array(directions)
    result = HeightAtlas(data).evaluate(points)
    gradient = result["gradient"]
    hessian = result["hessian"]
    curvature = (
        hessian[:, 0] * d[:, 0] ** 2
        + 2 * hessian[:, 1] * d[:, 0] * d[:, 1]
        + hessian[:, 2] * d[:, 1] ** 2
    ) / (
        np.sqrt(1 + np.sum(gradient**2, axis=1))
        * (1 + np.sum(gradient * d, axis=1) ** 2)
    )
    raw = np.load(lidar_path)["points"]
    raw_xz = np.column_stack(
        (raw[:, 0] - origin["easting"], origin["northing"] - raw[:, 1])
    )
    raw_tree = cKDTree(raw_xz)
    nearest, _ = raw_tree.query(points)
    usable = np.ones(len(raw), dtype=bool)
    mask_info = data["metadata"].get("pavement_mask")
    if mask_info is not None:
        import shapely
        from build_surface_mesh import road_edges

        if (
            hashlib.sha256(
                (ROOT / "tools/build_surface_mesh.py").read_bytes()
            ).hexdigest()
            != mask_info["builder_sha256"]
        ):
            raise ValueError("Pavement mask builder changed")
        footprint, _, _ = road_edges(track)
        footprint = footprint.buffer(mask_info["buffer_m"])
        if hashlib.sha256(footprint.wkb).hexdigest() != mask_info["polygon_wkb_sha256"]:
            raise ValueError("Pavement mask differs from atlas")
        usable &= shapely.contains_xy(footprint, raw_xz[:, 0], raw_xz[:, 1])
        if int(sum(usable)) != mask_info["selected_points"]:
            raise ValueError("Pavement observation count differs")
    exclusion_queries = np.zeros(len(points), dtype=bool)
    refit = data["metadata"].get("structural_exclusion_refit")
    if refit is not None:
        import shapely
        from refit_lidar_exclusions import load_exclusions

        if (
            hashlib.sha256(args.exclusions.read_bytes()).hexdigest()
            != refit["exclusions_sha256"]
        ):
            raise ValueError("Structural exclusion differs from atlas")
        excluded = load_exclusions(
            args.exclusions, track, data["metadata"]["provenance_sha256"]["track"]
        )
        usable &= ~shapely.intersects_xy(excluded, raw_xz[:, 0], raw_xz[:, 1])
        if int(sum(usable)) != refit["remaining_observations"]:
            raise ValueError("Usable observation count differs from refit")
        exclusion_queries = shapely.intersects_xy(excluded, points[:, 0], points[:, 1])
    nearest_usable, _ = cKDTree(raw_xz[usable]).query(points)
    discrepancy = result["height"] - np.array(old_heights)
    extrema = {}
    for label, index in [
        ("min_curvature", int(np.argmin(curvature))),
        ("max_curvature", int(np.argmax(curvature))),
        ("max_height_change", int(np.argmax(abs(discrepancy)))),
    ]:
        extrema[label] = {
            **entries[index],
            "xz_m": points[index].tolist(),
            "normal_curvature_per_m": float(curvature[index]),
            "height_change_m": float(discrepancy[index]),
            "nearest_raw_point_m": float(nearest[index]),
            "nearest_usable_point_m": float(nearest_usable[index]),
            "inside_structural_exclusion": bool(exclusion_queries[index]),
        }
    raw_checks = {}
    for label, location in extrema.items():
        q = np.array(location["xz_m"])
        direction = evaluate(road, location["station_m"], location["lateral_m"])["Rs"][
            [0, 2]
        ]
        fits = []
        for radius in [1.5, 2, 3, 4, 6]:
            ids = raw_tree.query_ball_point(q, radius)
            if len(ids) < 20:
                fits.append(
                    {
                        "radius_m": radius,
                        "point_count": len(ids),
                        "status": "Insufficient points for local quadratic audit",
                    }
                )
                continue
            fits.append(
                {
                    "radius_m": radius,
                    **fit_height_graph(raw_xz[ids] - q, raw[ids, 2], direction),
                }
            )
        raw_checks[label] = fits
    summary = {
        "raw_extrema_checks": raw_checks,
        "queries": len(points),
        "station_step_m": args.step,
        "min_weight": float(result["weight"].min()),
        "patch_count_range": [
            int(result["patch_count"].min()),
            int(result["patch_count"].max()),
        ],
        "max_nearest_raw_point_m": float(nearest.max()),
        "max_nearest_usable_point_m": float(nearest_usable.max()),
        "queries_inside_structural_exclusion": int(sum(exclusion_queries)),
        "height_change_p95_abs_m": float(np.quantile(abs(discrepancy), 0.95)),
        "curvature_quantiles_per_m": np.quantile(
            curvature, [0, 0.01, 0.5, 0.99, 1]
        ).tolist(),
        "extrema": extrema,
        "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
        "limitations": "Sampled coverage and historical fit agreement, not survey accuracy or continuous bound proof",
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    fixture = {
        "atlas": data,
        "samples": [
            {
                "x": float(p[0]),
                "z": float(p[1]),
                "height": float(result["height"][i]),
                "gradient": gradient[i].tolist(),
                "hessian": hessian[i].tolist(),
            }
            for i, p in enumerate(points)
        ],
    }
    args.output.with_suffix(".parity.json").write_text(
        json.dumps(fixture, separators=(",", ":")) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
