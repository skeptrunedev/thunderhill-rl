"""Apply reviewed edge displacements without straightening the baseline route."""

import hashlib
import json

import numpy as np


def apply_envelope(track, manifest):
    encoded = json.dumps(track, separators=(",", ":")) + "\n"
    if (
        hashlib.sha256(encoded.encode()).hexdigest()
        != manifest["baseline_track_sha256"]
    ):
        raise ValueError("Pavement envelope baseline changed")
    samples = track["samples"]
    p = np.array([s["p"] for s in samples])[:, [0, 2]]
    widths = np.array([s["width"] for s in samples])
    stations = np.array([s["s"] for s in samples])
    length = float(track["length_m"])
    tangent = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    left = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    left /= np.linalg.norm(left, axis=1)[:, None]
    edges = np.stack(
        (p - left * widths[:, None] / 2, p + left * widths[:, None] / 2), axis=1
    )
    coverage = manifest["coverage"]
    start, end = [float(coverage[k]) for k in ("start_station_m", "end_station_m")]
    if (
        not all(np.isfinite(v) and 0 <= v < length for v in (start, end))
        or start == end
    ):
        raise ValueError("Invalid correction interval")
    span = (end - start) % length
    anchors = manifest["anchors"]
    source_s = np.array([a["station_m"] for a in anchors], dtype=float)
    targets = np.array(
        [[a["right_xz_m"], a["left_xz_m"]] for a in anchors], dtype=float
    )
    if (
        len(anchors) < 2
        or targets.shape != (len(anchors), 2, 2)
        or not np.isfinite(targets).all()
        or not np.isfinite(source_s).all()
        or np.any(source_s < 0)
        or np.any(source_s >= length)
    ):
        raise ValueError("Invalid correction anchors")
    offsets = (source_s - start) % length
    order = np.argsort(offsets)
    offsets, source_s, targets = offsets[order], source_s[order], targets[order]
    if (
        np.any(np.diff(offsets) <= 0)
        or abs(offsets[0]) > 1e-8
        or abs(offsets[-1] - span) > 1e-8
    ):
        raise ValueError("Anchors must uniquely span the complete correction interval")
    baseline = np.empty_like(targets)
    for side in range(2):
        for axis in range(2):
            baseline[:, side, axis] = np.interp(
                source_s, stations, edges[:, side, axis], period=length
            )
    displacement = targets - baseline
    if np.max(np.abs(displacement[[0, -1]])) > 0.001:
        raise ValueError("Correction endpoints must meet the unchanged baseline edges")
    query = (stations - start) % length
    selected = query <= span
    corrected = edges.copy()
    for side in range(2):
        for axis in range(2):
            corrected[selected, side, axis] += np.interp(
                query[selected], offsets, displacement[:, side, axis]
            )
    new_p = corrected.mean(axis=1)
    new_widths = np.linalg.norm(corrected[:, 1] - corrected[:, 0], axis=1)
    if np.any(new_widths <= 0) or not np.isfinite(new_p).all():
        raise ValueError("Correction collapsed the pavement")
    new_tangent = np.roll(new_p, -1, axis=0) - np.roll(new_p, 1, axis=0)
    normal = np.column_stack((new_tangent[:, 1], -new_tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    rendered_edges = np.stack(
        (
            new_p - normal * new_widths[:, None] / 2,
            new_p + normal * new_widths[:, None] / 2,
        ),
        axis=1,
    )
    residual = np.linalg.norm(rendered_edges - corrected, axis=2)
    return (
        new_p,
        new_widths,
        {
            "changed_samples": int(
                np.count_nonzero(np.max(abs(corrected - edges), axis=(1, 2)) > 1e-9)
            ),
            "maximum_center_displacement_m": float(
                np.max(np.linalg.norm(new_p - p, axis=1))
            ),
            "maximum_normal_frame_edge_discrepancy_m": float(residual.max()),
            "width_range_m": [float(new_widths.min()), float(new_widths.max())],
            "method": "Interpolate edge displacement in baseline station coordinates, preserve route detail, recompute center and width; report edge discrepancy after recomputing normals",
        },
    )
