# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "shapely==2.1.2", "rasterio==1.4.4"]
# ///
"""Compare local spline fits with provisional pavement versus all ground support.

A mask based on existing road edges is a sensitivity experiment, not a surveyed
pavement classification. Only review artifacts are written.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from build_lidar_atlas import weight_axis
from build_road_surface import evaluate
from build_surface_mesh import road_edges
from fit_cyclone_surface import bending_matrix, fit, knots, metrics, normal_curvature
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    atlas_path = ROOT / "artifacts/road-surface/lidar-atlas.json"
    data = json.loads(atlas_path.read_text())
    paths = {
        "track": ROOT / "godot/data/track.json",
        "road": ROOT / "artifacts/road-surface/road-surface.json",
        "lidar": ROOT / "artifacts/reference/lidar/road-ground-points.npz",
    }
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    if hashes != data["metadata"]["provenance_sha256"]:
        raise ValueError("Atlas source provenance mismatch")
    track = json.loads(paths["track"].read_text())
    road = json.loads(paths["road"].read_text())
    raw = np.load(paths["lidar"])["points"]
    o = track["origin"]
    xz = np.column_stack((raw[:, 0] - o["easting"], o["northing"] - raw[:, 1]))
    heights = raw[:, 2] - o["elevation_m"]
    tree = cKDTree(xz)
    footprint, _, _ = road_edges(track)
    audit = json.loads(
        (ROOT / "artifacts/road-surface/lidar-atlas-circuit.json").read_text()
    )
    locations = [audit["extrema"][k] for k in ["min_curvature", "max_curvature"]]
    locations.append({"station_m": 1820.392, "lateral_m": 5.76445})
    reports = []
    for location in locations:
        r = evaluate(road, location["station_m"], location["lateral_m"])
        p = r["R"][[0, 2]]
        d = r["Rs"][[0, 2]]
        best = max(
            data["patches"],
            key=lambda patch: np.prod(
                weight_axis(p - np.array(patch["origin"]), data["support_radius_m"])[0]
            ),
        )
        center = np.array(best["origin"])
        ids = np.array(tree.query_ball_point(center, 16, p=np.inf), dtype=int)
        q = xz[ids] - center
        z = heights[ids]
        # One global spatial fold shared across candidates. Score common pavement
        # observations as well as each candidate's own training domain.
        blocks = np.floor(xz[ids] / 2).astype(int)
        held = (blocks[:, 0] + 2 * blocks[:, 1]) % 3 == 0
        common = shapely.contains_xy(footprint, xz[ids, 0], xz[ids, 1])
        candidates = []
        for buffer in [None, 0.0, -0.5, 0.5]:
            mask = (
                np.ones(len(ids), dtype=bool)
                if buffer is None
                else shapely.contains_xy(
                    footprint.buffer(buffer), xz[ids, 0], xz[ids, 1]
                )
            )
            for spacing in [1.0, 2.0, 3.0]:
                t = (knots([-16, 16], spacing), knots([-16, 16], spacing))
                penalty = bending_matrix(t)
                cv, cv_solve = fit(q[mask & ~held], z[mask & ~held], t, 0.01, penalty)
                spline, solve = fit(q[mask], z[mask], t, 0.01, penalty)
                at = p - center
                entry = {
                    "mask_buffer_m": buffer,
                    "knot_spacing_m": spacing,
                    "lambda_m4": 0.01,
                    "points": int(sum(mask)),
                    "common_pavement_heldout": metrics(
                        cv(q[held & common]) - z[held & common]
                    ),
                    "training": metrics(spline(q[mask]) - z[mask]),
                    "height_m": float(spline(at)),
                    "curvature_per_m": normal_curvature(spline, at, d),
                    "solves": [cv_solve, solve],
                }
                candidates.append(entry)
                print(
                    json.dumps({"station_m": location["station_m"], **entry}),
                    flush=True,
                )
        reports.append(
            {
                "location": location,
                "query_xz": p.tolist(),
                "patch_center": center.tolist(),
                "query_inside_provisional_mask": bool(
                    shapely.contains_xy(footprint, p[0], p[1])
                ),
                "common_pavement_heldout_count": int(sum(held & common)),
                "candidates": candidates,
            }
        )
    output = ROOT / "artifacts/road-surface/pavement-fit-comparison.json"
    output.write_text(
        json.dumps(
            {
                "provenance_sha256": hashes,
                "atlas_sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "status": "Sensitivity experiment; provisional masks, no automatic selection or game integration",
                "locations": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
