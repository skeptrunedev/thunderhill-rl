# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1"]
# ///
"""Experimental local C2 lidar height graphs; never changes shipped geometry.

API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdBSpline.html
Minimize mean squared height residual + lambda_m4 * area mean of
(f_xx**2 + 2*f_xy**2 + f_yy**2). Four point Gauss integration per knot
cell integrates the cubic spline bending energy exactly. Coordinates and
heights are meters in lidar EPSG:6339 and NAVD88, respectively. No datum
conversion, pavement classification, or automatic realism acceptance occurs.
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.interpolate import BSpline, NdBSpline
from scipy.sparse import csr_array, kron, vstack
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree
from build_road_surface import evaluate

ROOT = Path(__file__).resolve().parents[1]


def knots(bounds, spacing):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Knot spacing must be finite and positive")
    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0]:
        raise ValueError("Knot bounds must be finite and increasing")
    edges = np.linspace(
        bounds[0], bounds[1], int(np.ceil(np.diff(bounds)[0] / spacing)) + 1
    )
    return np.r_[np.repeat(edges[0], 3), edges, np.repeat(edges[-1], 3)]


def bending_matrix(t):
    axes = []
    area = np.prod([k[-1] - k[0] for k in t])
    for k in t:
        edges = np.unique(k)
        nodes, weights = np.polynomial.legendre.leggauss(4)
        half = np.diff(edges) / 2
        x = ((edges[:-1] + half)[:, None] + half[:, None] * nodes).ravel()
        w = (half[:, None] * weights).ravel()
        basis = BSpline(k, np.eye(len(k) - 4), 3, extrapolate=False)
        axes.append([csr_array(basis(x, nu=n) * np.sqrt(w)[:, None]) for n in range(3)])
    return vstack(
        [
            kron(axes[0][2], axes[1][0]),
            np.sqrt(2) * kron(axes[0][1], axes[1][1]),
            kron(axes[0][0], axes[1][2]),
        ],
        format="csr",
    ) / np.sqrt(area)


def fit(xy, z, t, strength, penalty=None):
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Bending strength must be finite and nonnegative")
    xy = np.ascontiguousarray(xy, dtype=float)
    design = NdBSpline.design_matrix(xy, t, 3, extrapolate=False)
    # Preserve unsupported trailing basis columns for the bending penalty.
    design.resize((len(xy), int(np.prod([len(k) - 4 for k in t]))))
    offset = float(np.mean(z))
    penalty = bending_matrix(t) if penalty is None else penalty
    matrix = vstack(
        [design / np.sqrt(len(z)), np.sqrt(strength) * penalty], format="csr"
    )
    rhs = np.r_[(z - offset) / np.sqrt(len(z)), np.zeros(penalty.shape[0])]
    result = lsmr(matrix, rhs, atol=1e-10, btol=1e-10, maxiter=6000)
    if result[1] not in (1, 2):
        raise RuntimeError(
            f"LSMR did not converge: stop={result[1]}, iterations={result[2]}"
        )
    coefficients = result[0].reshape(tuple(len(k) - 4 for k in t)) + offset
    return NdBSpline(t, coefficients, 3, extrapolate=False), {
        "stop": int(result[1]),
        "iterations": int(result[2]),
    }


def normal_curvature(spline, xy, direction):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    gradient = np.array([float(spline(xy, nu=n)) for n in [(1, 0), (0, 1)]])
    xx, cross, yy = [float(spline(xy, nu=n)) for n in [(2, 0), (1, 1), (0, 2)]]
    numerator = d @ np.array([[xx, cross], [cross, yy]]) @ d
    return float(
        numerator / (np.sqrt(1 + gradient @ gradient) * (1 + (gradient @ d) ** 2))
    )


def metrics(errors):
    return {
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "p95_abs_m": float(np.quantile(abs(errors), 0.95)),
        "max_abs_m": float(max(abs(errors))),
    }


def validate_provenance(track, road, hashes):
    if road.get("metadata", {}).get("track_sha256") != hashes["track"]:
        raise ValueError("Experimental road provenance does not match track bytes")
    if (
        track.get("metadata", {}).get("lidar", {}).get("corridor_sha256")
        != hashes["lidar"]
    ):
        raise ValueError("Track lidar provenance does not match corridor bytes")


def validate_extremum(road, entry):
    r = evaluate(road, entry["station"], entry["lateral"])
    curvature = float(r["normal"] @ r["Rss"] / (r["Rs"] @ r["Rs"]))
    if not np.isfinite(curvature) or not np.isclose(
        curvature, entry["curvature"], rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Stale curvature audit does not match experimental road")
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/road-surface/cyclone-height-fit.json",
    )
    args = parser.parse_args()
    paths = {
        "track": ROOT / "godot/data/track.json",
        "road": ROOT / "artifacts/road-surface/road-surface.json",
        "audit": ROOT / "artifacts/road-surface/road-surface.audit.json",
        "lidar": ROOT / "artifacts/reference/lidar/road-ground-points.npz",
    }
    track, road, audit = [
        json.loads(paths[k].read_text()) for k in ["track", "road", "audit"]
    ]
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    validate_provenance(track, road, hashes)
    origin = track["origin"]
    locations = []
    for label, entry in audit["curvature_extrema"].items():
        r = validate_extremum(road, entry)
        locations.append(
            {
                "label": label,
                "station_m": entry["station"],
                "lateral_m": entry["lateral"],
                "xy": [r["R"][0] + origin["easting"], origin["northing"] - r["R"][2]],
                "direction": [r["Rs"][0], -r["Rs"][2]],
                "ruled_curvature_per_m": entry["curvature"],
            }
        )
    center = np.mean([x["xy"] for x in locations], axis=0)
    points = np.load(paths["lidar"])["points"]
    relative = points[:, :2] - center
    selected = np.all(abs(relative) <= 12, axis=1)
    xy, z = relative[selected], points[selected, 2]
    # Entire spatial blocks are withheld. Boundary cells remain in the audit;
    # there is no adjacent point random split or boundary trimming of errors.
    blocks = np.clip(np.floor((xy + 12) / 2).astype(int), 0, 11)
    folds = (blocks[:, 0] + 2 * blocks[:, 1]) % 3
    coverage_tree = cKDTree(xy)
    occupied, counts = np.unique(blocks, axis=0, return_counts=True)
    block_coverage = {
        "cell_size_m": 2,
        "total_cells": 144,
        "occupied_cells": len(occupied),
        "empty_cells": 144 - len(occupied),
        "occupied_count_range": [int(counts.min()), int(counts.max())],
        "cells": [
            {"index": b.tolist(), "count": int(n)} for b, n in zip(occupied, counts)
        ],
    }
    report = {
        "coverage": block_coverage,
        "status": "experimental evidence only; no runtime integration or realism acceptance",
        "crs": "EPSG:6339 horizontal; NAVD88 height",
        "local_origin_xy_m": center.tolist(),
        "bounds_local_m": [[-12, 12], [-12, 12]],
        "point_count": len(z),
        "provenance_sha256": hashes,
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": __doc__,
        "holdout": {
            "block_size_m": 2,
            "fold_count": 3,
            "assignment": "(clip(floor((x+12)/2),0,11) + 2*clip(floor((y+12)/2),0,11)) % 3",
            "fold_point_counts": [int(sum(folds == f)) for f in range(3)],
        },
        "limitations": [
            "Class 2 ground includes shoulders and potentially curbs; not a pavement mask",
            "Spatial cross validation assesses interpolation against same acquisition, not independent survey accuracy",
            "Nearby training blocks remain correlated; no spatial exclusion buffer",
            "Height error does not certify derivative or tire contact accuracy",
            "C2 continuity holds inside this patch; no global seam or runtime integration",
            "Historical 2023 data precedes repave",
        ],
        "candidates": [],
    }
    for spacing in [2.0, 3.0, 4.0]:
        t = tuple(knots([-12, 12], spacing) for _ in range(2))
        penalty = bending_matrix(t)
        for strength in [0.0, 0.01]:
            predictions = np.empty(len(z))
            solves = []
            for f in range(3):
                held = folds == f
                spline, solve = fit(xy[~held], z[~held], t, strength, penalty)
                predictions[held] = spline(xy[held])
                solves.append(solve)
            spline, solve = fit(xy, z, t, strength, penalty)
            entries = []
            for location in locations:
                q = np.asarray(location["xy"]) - center
                distances, _ = coverage_tree.query(q, k=1)
                entries.append(
                    {
                        **location,
                        "height_navd88_m": float(spline(q)),
                        "curvature_per_m": normal_curvature(
                            spline, q, location["direction"]
                        ),
                        "nearest_point_m": float(distances),
                        "points_within_2m": len(coverage_tree.query_ball_point(q, 2)),
                        "distance_to_patch_boundary_m": float(min(12 - abs(q))),
                    }
                )
            candidate = {
                "knot_spacing_m": spacing,
                "lambda_m4": strength,
                "heldout": metrics(predictions - z),
                "heldout_folds": [
                    metrics((predictions - z)[folds == f]) for f in range(3)
                ],
                "training": metrics(spline(xy) - z),
                "locations": entries,
                "solves": solves + [solve],
                "coefficient_count": int(spline.c.size),
                "knots": [k.tolist() for k in t],
                "degree": [3, 3],
                "coefficients_navd88_m": spline.c.tolist(),
            }
            report["candidates"].append(candidate)
            print(
                json.dumps(
                    {
                        k: candidate[k]
                        for k in [
                            "knot_spacing_m",
                            "lambda_m4",
                            "heldout",
                            "training",
                            "locations",
                        ]
                    }
                ),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
