# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1"]
# ///
"""Export an explicitly selected experimental lidar patch to Godot coordinates.

This does not select a calibrated surface or modify shipped game data.
The north axis is reflected into Godot south, including knot order and control
points, so analytical derivatives retain the correct signs.
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.interpolate import NdBSpline

ROOT = Path(__file__).resolve().parents[1]


def convert(candidate, patch_origin, track_origin):
    tx, tn = [np.asarray(k, dtype=float) for k in candidate["knots"]]
    coefficients = np.asarray(candidate["coefficients_navd88_m"], dtype=float)
    if candidate["degree"] != [3, 3]:
        raise ValueError("Only cubic patches are supported")
    if coefficients.shape != (len(tx) - 4, len(tn) - 4):
        raise ValueError("Coefficient shape does not match knots")
    values = np.r_[
        tx, tn, coefficients.ravel(), patch_origin, list(track_origin.values())
    ]
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite patch data")
    return {
        "schema_version": 1,
        "degree": [3, 3],
        "knots": [tx.tolist(), (-tn[::-1]).tolist()],
        "coefficients": (coefficients[:, ::-1] - track_origin["elevation_m"]).tolist(),
        "origin": [
            patch_origin[0] - track_origin["easting"],
            track_origin["northing"] - patch_origin[1],
        ],
    }


def fixture(candidate, patch_origin, track_origin, surface):
    original = NdBSpline(
        tuple(np.array(k) for k in candidate["knots"]),
        np.array(candidate["coefficients_navd88_m"]),
        3,
        extrapolate=False,
    )
    # Every knot, interval midpoint, boundary and reproducible interior samples.
    axes = []
    for k in candidate["knots"]:
        edges = np.unique(k)
        axes.append(np.unique(np.r_[edges, (edges[:-1] + edges[1:]) / 2]))
    points = np.array(np.meshgrid(*axes, indexing="ij")).reshape(2, -1).T
    rng = np.random.default_rng(1847)
    points = np.vstack(
        [points, rng.uniform([a[0] for a in axes], [a[-1] for a in axes], (100, 2))]
    )
    samples = []
    for q in points:
        samples.append(
            {
                "x": float(q[0] + surface["origin"][0]),
                "z": float(-q[1] + surface["origin"][1]),
                "height": float(original(q)) - track_origin["elevation_m"],
                "gradient": [
                    float(original(q, nu=(1, 0))),
                    -float(original(q, nu=(0, 1))),
                ],
                "hessian": [
                    float(original(q, nu=(2, 0))),
                    -float(original(q, nu=(1, 1))),
                    float(original(q, nu=(0, 2))),
                ],
            }
        )
    return {"surface": surface, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/road-surface/cyclone-height-fit.json",
    )
    parser.add_argument("--track", type=Path, default=ROOT / "godot/data/track.json")
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/road-surface/lidar-height-fixture.json",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    track = json.loads(args.track.read_text())
    track_hash = hashlib.sha256(args.track.read_bytes()).hexdigest()
    if report["provenance_sha256"]["track"] != track_hash:
        raise ValueError("Track differs from fitted patch provenance")
    candidates = [
        c
        for c in report["candidates"]
        if c["knot_spacing_m"] == args.spacing and c["lambda_m4"] == args.strength
    ]
    if len(candidates) != 1:
        raise ValueError("Select exactly one existing candidate")
    # Track origin contains extra CRS metadata; only these three are numeric inputs.
    origin = {k: track["origin"][k] for k in ["easting", "northing", "elevation_m"]}
    surface = convert(candidates[0], report["local_origin_xy_m"], origin)
    result = fixture(candidates[0], report["local_origin_xy_m"], origin, surface)
    result["metadata"] = {
        "status": "Experimental parity fixture; not shipped geometry",
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "track_sha256": track_hash,
        "knot_spacing_m": args.spacing,
        "lambda_m4": args.strength,
        "vertical_lift_m": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(result["samples"])}))


if __name__ == "__main__":
    main()
