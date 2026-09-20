# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1"]
# ///
"""Tessellate the existing road footprint onto an independently selected height atlas.

Artifact only. The historical planar edges remain unchanged. Error measurements
sample triangle centroids and edge midpoints, not a certified continuous bound.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from build_lidar_atlas import HeightAtlas

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planar_edges(track):
    samples = track["samples"]
    p = np.asarray([row["p"] for row in samples], dtype=float)
    widths = np.asarray([row["width"] for row in samples], dtype=float)
    stations = np.asarray([row["s"] for row in samples], dtype=float)
    length = float(track["length_m"])
    if (
        p.shape != (len(samples), 3)
        or len(samples) < 3
        or not all(np.isfinite(a).all() for a in (p, widths, stations))
        or not np.isfinite(length)
        or np.any(widths <= 0)
        or stations[0] != 0
        or np.any(np.diff(stations) <= 0)
        or length <= stations[-1]
    ):
        raise ValueError("Invalid track positions, widths or station sequence")
    p = p[:, [0, 2]]
    tangent = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Degenerate centered planar tangent")
    left = np.column_stack((tangent[:, 1], -tangent[:, 0])) / lengths[:, None]
    return (
        p,
        p - left * widths[:, None] / 2,
        p + left * widths[:, None] / 2,
        widths,
        stations,
    )


def build_mesh(track, atlas, spacing=1.0, lift=0.04):
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(lift):
        raise ValueError("Spacing must be finite and positive; lift must be finite")
    p, right, left, widths, stations = planar_edges(track)
    along = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    if np.any(along <= 0):
        raise ValueError("Coincident consecutive planar samples")
    subdivisions = np.ceil(along / spacing).astype(int)
    across = int(np.ceil(widths.max() / spacing))
    fraction = np.linspace(0, 1, across + 1)
    rows, uv_rows, original_rows = [], [], []
    for i, count in enumerate(subdivisions):
        original_rows.append(len(rows))
        j = (i + 1) % len(p)
        end_station = stations[j] if j else track["length_m"]
        for t in np.arange(count) / count:
            a = right[i] + t * (right[j] - right[i])
            b = left[i] + t * (left[j] - left[i])
            rows.append(a + fraction[:, None] * (b - a))
            width = widths[i] + t * (widths[j] - widths[i])
            uv_rows.append(
                np.column_stack(
                    (
                        fraction * width,
                        np.full(
                            across + 1, stations[i] + t * (end_station - stations[i])
                        ),
                    )
                )
            )
    rows.append(rows[0].copy())
    closing_uv = uv_rows[0].copy()
    closing_uv[:, 1] = track["length_m"]
    uv_rows.append(closing_uv)
    xz = np.asarray(rows).reshape(-1, 2)
    result = atlas.evaluate(xz)
    h, gradient = np.asarray(result["height"]), np.asarray(result["gradient"])
    if (
        h.shape != (len(xz),)
        or gradient.shape != (len(xz), 2)
        or not all(np.isfinite(v).all() for v in (h, gradient))
    ):
        raise ValueError("Invalid atlas result")
    vertices = np.column_stack((xz[:, 0], h + lift, xz[:, 1]))
    normals = np.column_stack((-gradient[:, 0], np.ones(len(xz)), -gradient[:, 1]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # Guarantee exact position and normal equality at the sole duplicated UV seam.
    vertices[-across - 1 :] = vertices[: across + 1]
    normals[-across - 1 :] = normals[: across + 1]
    a = (np.arange(len(rows) - 1)[:, None] * (across + 1) + np.arange(across)).ravel()
    b, d = a + 1, a + across + 1
    c = d + 1
    triangles = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    tri = xz[triangles]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(np.abs(signed) < 1e-12):
        raise ValueError("Degenerate planar mesh triangle")
    # Clockwise viewed from above is the Godot front face, matching track._quad.
    triangles[signed < 0] = triangles[signed < 0, ::-1]
    errors = []
    worst = None
    barycentric = np.array(
        [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
    )
    for start in range(0, len(triangles), 8192):
        verts = vertices[triangles[start : start + 8192]]
        queries = np.einsum("qa,tad->tqd", barycentric, verts)
        flat = queries.reshape(-1, 3)
        expected = atlas.evaluate(flat[:, [0, 2]])["height"] + lift
        error = np.abs(flat[:, 1] - expected)
        index = int(np.argmax(error))
        if worst is None or error[index] > worst["error_m"]:
            worst = {
                "error_m": float(error[index]),
                "triangle": start + index // 4,
                "sample": ["centroid", "edge_ab", "edge_bc", "edge_ca"][index % 4],
                "position_xz": flat[index, [0, 2]].tolist(),
            }
        errors.append(error)
    errors = np.concatenate(errors)
    edge_vertices = vertices.reshape(-1, across + 1, 3)[:, [0, -1]]
    report = {
        "vertices": len(vertices),
        "triangles": len(triangles),
        "rows_including_seam": len(rows),
        "across_intervals": across,
        "sampled_vertical_error_m": {
            "max": float(errors.max()),
            "p95": float(np.percentile(errors, 95)),
            "queries": len(errors),
            "worst": worst,
        },
        "bounds_min_xyz": vertices.min(axis=0).tolist(),
        "bounds_max_xyz": vertices.max(axis=0).tolist(),
        "edge_height_range_m": [
            float(edge_vertices[:, :, 1].min()),
            float(edge_vertices[:, :, 1].max()),
        ],
        "max_original_edge_height_change_m": float(
            np.max(
                np.abs(
                    vertices.reshape(-1, across + 1, 3)[original_rows][:, [0, -1], 1]
                    - np.array(
                        [
                            [
                                r["p"][1] - np.tan(r["bank"]) * r["width"] / 2 + lift,
                                r["p"][1] + np.tan(r["bank"]) * r["width"] / 2 + lift,
                            ]
                            for r in track["samples"]
                        ]
                    )
                )
            )
        ),
    }
    return {
        "schema_version": 1,
        "vertices": vertices.tolist(),
        "normals": normals.tolist(),
        "uv": np.asarray(uv_rows).reshape(-1, 2).tolist(),
        "triangles": triangles.tolist(),
        "original_sample_rows": original_rows,
        "edge_vertices": edge_vertices.tolist(),
        "metadata": {
            "spacing_m": spacing,
            "road_lift_m": lift,
            "validation": report,
            "limitations": "Provisional historical planar margins. Sampled interpolation error is not a continuous bound. Atlas fit accuracy is separate from tessellation accuracy.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--track", type=Path, default=ROOT / "godot/data/track.json")
    parser.add_argument(
        "--road", type=Path, default=ROOT / "artifacts/road-surface/road-surface.json"
    )
    parser.add_argument(
        "--lidar",
        type=Path,
        default=ROOT / "artifacts/reference/lidar/road-ground-points.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/road-surface/atlas-road-mesh.json",
    )
    parser.add_argument("--spacing", type=float, default=1.0)
    args = parser.parse_args()
    data = json.loads(args.atlas.read_text())
    track = json.loads(args.track.read_text())
    hashes = {key: digest(getattr(args, key)) for key in ("track", "road", "lidar")}
    if data["metadata"]["provenance_sha256"] != hashes or data["metadata"].get(
        "diagnostic_limit", 0
    ):
        raise ValueError("Atlas source provenance mismatch or diagnostic atlas")
    if track["metadata"]["lidar"]["corridor_sha256"] != hashes["lidar"]:
        raise ValueError("Track lidar source mismatch")
    mesh = build_mesh(track, HeightAtlas(data), args.spacing)
    mesh["metadata"].update(
        {
            "track_sha256": hashes["track"],
            "atlas_sha256": digest(args.atlas),
            "provenance_sha256": hashes,
            "builder_sha256": digest(__file__),
            "license": "ODbL 1.0 road derived geometry; public domain USGS lidar",
            "attribution": "OpenStreetMap contributors; USGS 3DEP",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(mesh, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(json.dumps(mesh["metadata"], indent=2))


if __name__ == "__main__":
    main()
