# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "shapely==2.1.2", "rasterio==1.4.4"]
# ///
"""Fit overlapping lidar height patches and blend their full C2 differentials.

Experimental output only. Geometry, masks and historical accuracy still require
review before the atlas can replace live road and terrain contact.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from build_road_surface import evaluate as ribbon
from fit_cyclone_surface import bending_matrix, fit, knots, metrics
from scipy.interpolate import NdBSpline
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def weight_axis(q, radius):
    t = q / radius
    a = np.maximum(1 - t * t, 0)
    inside = np.abs(t) < 1
    return (
        a**3,
        np.where(inside, -6 * t * a * a / radius, 0),
        np.where(inside, (-6 * a * a + 24 * t * t * a) / (radius * radius), 0),
    )


class HeightAtlas:
    def __init__(self, data):
        self.radius = float(data["support_radius_m"])
        self.patches = [
            (
                np.array(p["origin"]),
                NdBSpline(
                    tuple(np.array(k) for k in p["knots"]),
                    np.array(p["coefficients"]),
                    3,
                    extrapolate=False,
                ),
            )
            for p in data["patches"]
        ]

    def evaluate(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite query")
        tree = cKDTree(points)
        w = np.zeros(len(points))
        dw = np.zeros((len(points), 2))
        ddw = np.zeros((len(points), 3))
        f = w.copy()
        df = dw.copy()
        ddf = ddw.copy()
        counts = np.zeros(len(points), dtype=int)
        for center, spline in self.patches:
            ids = np.array(
                tree.query_ball_point(center, self.radius, p=np.inf), dtype=int
            )
            if not len(ids):
                continue
            q = points[ids] - center
            keep = np.all(np.abs(q) < self.radius, axis=1)
            ids, q = ids[keep], q[keep]
            if not len(ids):
                continue
            a, da, dda = weight_axis(q[:, 0], self.radius)
            b, db, ddb = weight_axis(q[:, 1], self.radius)
            wi = a * b
            dwi = np.column_stack((da * b, a * db))
            ddwi = np.column_stack((dda * b, da * db, a * ddb))
            h = spline(q)
            dh = np.column_stack([spline(q, nu=n) for n in [(1, 0), (0, 1)]])
            ddh = np.column_stack([spline(q, nu=n) for n in [(2, 0), (1, 1), (0, 2)]])
            w[ids] += wi
            dw[ids] += dwi
            ddw[ids] += ddwi
            f[ids] += wi * h
            df[ids] += dwi * h[:, None] + wi[:, None] * dh
            ddf[ids] += (
                ddwi * h[:, None]
                + wi[:, None] * ddh
                + np.column_stack(
                    (
                        2 * dwi[:, 0] * dh[:, 0],
                        dwi[:, 0] * dh[:, 1] + dwi[:, 1] * dh[:, 0],
                        2 * dwi[:, 1] * dh[:, 1],
                    )
                )
            )
            counts[ids] += 1
        if np.any(w <= 0):
            raise ValueError(f"{int(sum(w <= 0))} queries outside atlas coverage")
        h = f / w
        dh = (df - h[:, None] * dw) / w[:, None]
        ddh = (
            ddf
            - h[:, None] * ddw
            - np.column_stack(
                (
                    2 * dh[:, 0] * dw[:, 0],
                    dh[:, 0] * dw[:, 1] + dh[:, 1] * dw[:, 0],
                    2 * dh[:, 1] * dw[:, 1],
                )
            )
        ) / w[:, None]
        if not all(np.isfinite(v).all() for v in [h, dh, ddh]):
            raise ValueError("Nonfinite atlas differential")
        return {
            "height": h,
            "gradient": dh,
            "hessian": ddh,
            "weight": w,
            "patch_count": counts,
        }


def serialized_patch(spline, center):
    return {
        "schema_version": 1,
        "degree": [3, 3],
        "knots": [k.tolist() for k in spline.t],
        "coefficients": spline.c.tolist(),
        "origin": center.tolist(),
    }


def sample_road(track, road):
    points = []
    for row in track["samples"]:
        for u in np.linspace(-row["width"] / 2, row["width"] / 2, 5):
            p = ribbon(road, row["s"], u)["R"]
            points.append(p[[0, 2]])
    return np.array(points)


def initialize_fit_worker():
    global worker_knots, worker_penalty
    worker_knots = (knots([-16, 16], 2), knots([-16, 16], 2))
    worker_penalty = bending_matrix(worker_knots)


def fit_patch(task):
    center, q, h, held = task
    if len(h) < 100 or sum(held) < 20 or sum(~held) < 20:
        raise ValueError(f"Insufficient patch support at {center.tolist()}")
    cv, cv_solve = fit(q[~held], h[~held], worker_knots, 0.01, worker_penalty)
    validation = metrics(cv(q[held]) - h[held])
    spline, solve = fit(q, h, worker_knots, 0.01, worker_penalty)
    return serialized_patch(spline, center), {
        "center": center.tolist(),
        "points": len(h),
        "heldout": validation,
        "training": metrics(spline(q) - h),
        "solves": [cv_solve, solve],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/road-surface/lidar-atlas.json"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Diagnostic patch limit; never accepted as full coverage",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--pavement-mask-buffer",
        type=float,
        default=None,
        help="Restrict observations to provisional road polygon buffered by these metres",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        raise ValueError("Workers must be positive and diagnostic limit nonnegative")
    if args.pavement_mask_buffer is not None and not np.isfinite(
        args.pavement_mask_buffer
    ):
        raise ValueError("Pavement buffer must be finite")
    paths = {
        "track": ROOT / "godot/data/track.json",
        "road": ROOT / "artifacts/road-surface/road-surface.json",
        "lidar": ROOT / "artifacts/reference/lidar/road-ground-points.npz",
    }
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    track = json.loads(paths["track"].read_text())
    road = json.loads(paths["road"].read_text())
    if (
        road["metadata"]["track_sha256"] != hashes["track"]
        or track["metadata"]["lidar"]["corridor_sha256"] != hashes["lidar"]
    ):
        raise ValueError("Source provenance mismatch")
    raw = np.load(paths["lidar"])["points"]
    origin = track["origin"]
    xz = np.column_stack(
        (raw[:, 0] - origin["easting"], origin["northing"] - raw[:, 1])
    )
    heights = raw[:, 2] - origin["elevation_m"]
    mask_metadata = None
    if args.pavement_mask_buffer is not None:
        import shapely
        from build_surface_mesh import road_edges

        footprint, _, _ = road_edges(track)
        footprint = footprint.buffer(args.pavement_mask_buffer)
        mask = shapely.contains_xy(footprint, xz[:, 0], xz[:, 1])
        mask_metadata = {
            "source": "Provisional rendered road polygon, not surveyed pavement",
            "buffer_m": args.pavement_mask_buffer,
            "polygon_wkb_sha256": hashlib.sha256(footprint.wkb).hexdigest(),
            "selected_points": int(sum(mask)),
            "input_points": len(mask),
            "builder_sha256": hashlib.sha256(
                (ROOT / "tools/build_surface_mesh.py").read_bytes()
            ).hexdigest(),
        }
        xz, heights = xz[mask], heights[mask]
    tree = cKDTree(xz)
    road_points = sample_road(track, road)
    radius = 12.0
    centers = np.unique(np.round(road_points / radius).astype(int), axis=0) * radius
    if args.limit:
        centers = centers[: args.limit]
    tasks = []
    for center in centers:
        ids = np.array(tree.query_ball_point(center, 16, p=np.inf), dtype=int)
        blocks = np.floor(xz[ids] / 2).astype(int)
        held = (blocks[:, 0] + 2 * blocks[:, 1]) % 3 == 0
        tasks.append((center, xz[ids] - center, heights[ids], held))
    patches, reports = [], []
    # Spawn isolates BLAS state, and map preserves deterministic patch order.
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=get_context("spawn"),
        initializer=initialize_fit_worker,
    ) as pool:
        for index, (patch, report) in enumerate(
            pool.map(fit_patch, tasks, chunksize=1)
        ):
            patches.append(patch)
            reports.append(report)
            if index % 10 == 0 or index + 1 == len(centers):
                print(
                    json.dumps(
                        {
                            "patch": index + 1,
                            "total": len(centers),
                            "heldout_rmse_m": report["heldout"]["rmse_m"],
                        }
                    ),
                    flush=True,
                )
    data = {
        "schema_version": 1,
        "support_radius_m": radius,
        "patches": patches,
        "metadata": {
            "status": "Experimental; no runtime integration",
            "provenance_sha256": hashes,
            "source_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "knot_spacing_m": 2,
            "lambda_m4": 0.01,
            "fit_halfwidth_m": 16,
            "diagnostic_limit": args.limit,
            "pavement_mask": mask_metadata,
            "workers": args.workers,
            "validation": "One of three global 2m spatial folds withheld per patch; same acquisition, no independent survey",
            "limitations": [
                "Pavement boundary is provisional"
                if mask_metadata
                else "No pavement mask",
                "Historical ground observations include curb and soil",
                "Height agreement does not certify derivatives",
                "Patch validation includes unsupported square corners; road coverage checked separately",
            ],
        },
    }
    report = {"patches": reports}
    if not args.limit:
        result = HeightAtlas(data).evaluate(road_points)
        nearest, _ = tree.query(road_points)
        report["road_coverage"] = {
            "queries": len(road_points),
            "min_weight": float(result["weight"].min()),
            "min_patches": int(result["patch_count"].min()),
            "max_nearest_raw_point_m": float(nearest.max()),
            "height_range_m": [
                float(result["height"].min()),
                float(result["height"].max()),
            ],
            "max_slope": float(np.linalg.norm(result["gradient"], axis=1).max()),
            "max_abs_hessian_per_m": float(np.abs(result["hessian"]).max()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(report.get("road_coverage", {"diagnostic_only": True})), flush=True
    )


if __name__ == "__main__":
    main()
