# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "matplotlib==3.10.8", "pyproj==3.7.2"]
# ///
"""Derive historical visual divider height and face from preserved lidar returns."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_pit_envelope import digest, frame, profile
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {
        "baseline_track": ROOT / "artifacts/road-surface/pit-baseline-track.json",
        "road": ROOT / "artifacts/road-surface/road-surface.json",
        "divider_measurements": ROOT
        / "artifacts/road-surface/pit-divider-measurements.json",
        "raw_local_points": ROOT / "artifacts/road-surface/pit-divider-points.npz",
        "adopted_envelope": ROOT / "data/reference/pit-envelope.json",
        "corrected_track": ROOT / "godot/data/track.json",
    }
    data = {
        k: json.loads(p.read_text()) for k, p in paths.items() if p.suffix == ".json"
    }
    manifest = data["adopted_envelope"]
    for key in ("road", "divider_measurements", "raw_local_points"):
        if digest(paths[key]) != manifest["sources"][key]["sha256"]:
            raise ValueError(f"Changed evidence {key}")
    if digest(paths["baseline_track"]) != manifest["baseline_track_sha256"]:
        raise ValueError("Changed baseline")
    if data["road"]["metadata"]["track_sha256"] != digest(paths["baseline_track"]):
        raise ValueError("Wrong road station frame")
    for tile in data["divider_measurements"]["tiles"]:
        if digest(ROOT / "artifacts/reference/lidar" / tile["path"]) != tile["sha256"]:
            raise ValueError("Changed raw tile")
    archive = np.load(paths["raw_local_points"])
    if archive["columns"].tolist() != [
        "x",
        "z",
        "height_local",
        "classification",
        "source_id",
    ]:
        raise ValueError("Unexpected columns")
    raw = archive["points"]
    if (
        raw.shape != (data["divider_measurements"]["observation_count"], 5)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("Malformed points")
    tree = cKDTree(raw[:, :2])
    baseline = data["baseline_track"]["samples"]
    current = data["corrected_track"]["samples"]
    if data["corrected_track"]["origin"] != data["baseline_track"]["origin"]:
        raise ValueError("Origins differ")
    p = np.array([s["p"] for s in current])
    tangent = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    lefts = np.column_stack([tangent[:, 2], -tangent[:, 0]])
    lefts /= np.linalg.norm(lefts, axis=1)[:, None]
    right = p[:, [0, 2]] - lefts * np.array([s["width"] / 2 for s in current])[:, None]
    rows = []
    for s in np.r_[43.0, np.arange(50.0, 421.0, 10.0), 429.0]:
        measured = profile(raw, tree, data["road"], s)
        center, along, left = frame(data["road"], s)
        points = raw[tree.query_ball_point(center, 13)]
        delta = points[:, :2] - center
        t, u = delta @ along, delta @ left
        design = np.column_stack([np.ones(len(points)), t, u])
        core = (abs(t) <= 3) & (u >= 0) & (u <= 3) & (points[:, 3] == 2)
        plane = np.linalg.lstsq(design[core], points[core, 2], rcond=None)[0]
        h = points[:, 2] - design @ plane
        band = measured["sensitivity"][-1]
        if not band["count"]:
            raise ValueError(f"Missing raised band {s}")
        keep = (
            (abs(t) <= 1)
            & (u >= band["band_m"][0])
            & (u <= band["band_m"][1])
            & (h >= 0.35)
            & (h <= 1.5)
        )
        longitudinal_sensitivity = []
        for half in [1.0, 2.0, 4.0]:
            candidates = (
                (abs(t) <= half) & (u >= -7) & (u <= 0) & (h >= 0.35) & (h <= 1.5)
            )
            edges = np.arange(-7, 0.001, 0.2)
            counts, _ = np.histogram(u[candidates], edges)
            occupied = np.flatnonzero(counts >= max(2, int(half * 2)))
            groups = (
                np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
                if len(occupied)
                else []
            )
            if groups:
                group = groups[-1]
                selected = (
                    candidates & (u >= edges[group[0]]) & (u <= edges[group[-1] + 1])
                )
                longitudinal_sensitivity.append(
                    {
                        "half_length_m": half,
                        "count": int(selected.sum()),
                        "face_m": float(np.quantile(u[selected], 0.95)),
                        "top_m": float(np.quantile(h[selected], 0.95)),
                        "source_counts": {
                            str(int(key)): int(count)
                            for key, count in zip(
                                *np.unique(points[selected, 4], return_counts=True)
                            )
                        },
                        "top_quantiles_90_95_100_m": np.quantile(
                            h[selected], [0.9, 0.95, 1.0]
                        ).tolist(),
                    }
                )
        top = np.quantile(h[keep], [0.5, 0.9, 0.95, 1])
        face = measured["structure_face_candidate_m"]
        face_xz = center + face * left
        source_s = np.interp(
            s, [a["s"] for a in baseline], [a["source_s"] for a in baseline]
        )
        edge = np.array(
            [
                np.interp(source_s, [a["source_s"] for a in current], right[:, j])
                for j in range(2)
            ]
        )
        ids, counts = np.unique(points[keep, 4].astype(int), return_counts=True)
        rows.append(
            {
                "longitudinal_sensitivity": longitudinal_sensitivity,
                "baseline_station_m": float(s),
                "source_s_m": float(source_s),
                "face_xz_m": face_xz.tolist(),
                "road_inward_xz": left.tolist(),
                "height_m": longitudinal_sensitivity[-1]["top_m"],
                "height_support": longitudinal_sensitivity[-1],
                "narrow_profile_height_m": float(top[2]),
                "height_quantiles_50_90_95_100_m": top.tolist(),
                "raw_return_count": int(keep.sum()),
                "source_counts": dict(zip(map(str, ids), map(int, counts))),
                "ground_plane_rmse_m": measured["ground_plane_rmse_m"],
                "face_lateral_m": face,
                "face_sensitivity_m": measured["sensitivity_range_m"],
                "corrected_road_right_xz_m": edge.tolist(),
                "road_edge_inward_of_face_m": float((edge - face_xz) @ left),
            }
        )
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout="constrained")
    for ax in axes:
        ax.plot(
            [r["baseline_station_m"] for r in rows],
            [r["face_lateral_m"] for r in rows],
            "o-",
            label="Raw nearest raised face",
        )
        ax.plot(
            [r["baseline_station_m"] for r in rows],
            [r["face_lateral_m"] + r["road_edge_inward_of_face_m"] for r in rows],
            label="Corrected road edge",
        )
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_xlim(40, 130)
    axes[1].set_xlim(130, 435)
    fig.savefig(ROOT / "artifacts/road-surface/pit-wall-profile.png", dpi=150)
    plt.close(fig)
    report = {
        "schema_version": 1,
        "status": "Historical visual reconstruction proposal; not collision geometry or surveyed dimensions",
        "origin": data["baseline_track"]["origin"],
        "sources": {
            k: {"path": str(v.relative_to(ROOT)), "sha256": digest(v)}
            for k, v in paths.items()
        },
        "raw_tiles": data["divider_measurements"]["tiles"],
        "builder_sha256": digest(Path(__file__)),
        "method": {
            "profile_half_length_m": 1,
            "ground_plane": "Class 2 interior pavement returns, along +/-3m and lateral 0..3m",
            "face": "95th lateral percentile of closest raised occupied band; 0.2m bins with at least 2 returns",
            "height_selection_above_plane_m": [0.35, 1.5],
            "top": "95th height percentile in widest evaluated along window (+/-4m), same raised band detector requiring at least8 returns per0.2m bin. Narrow +/-1m raw quantiles retained separately",
            "smoothing": "No across-station smoothing. Height pools returns within +/-4m along track; face remains original +/-1m measurement. Linear interpolation between output rows is a visual approximation",
            "visual_width_m": 0.35,
            "visual_width_status": "Artistic value retained from existing wall, not measured",
            "centerline": "face_xz_m minus road_inward_xz times half visual width",
            "base": "Runtime surface height; interior fitted plane supplies relative height only",
        },
        "limitations": [
            "Roof returns near station 392 are excluded by the 1.5m height ceiling; structure occlusion remains.",
            "Height quantile spread and threshold sensitivity are not absolute survey accuracy.",
            "Face is measured above ground, not wall base; width, end treatment and base shape remain artistic.",
            "Endpoints follow previously detected raised return span, not exact surveyed wall endpoints.",
        ],
        "rows": rows,
    }
    out = ROOT / "data/reference/pit-wall-profile.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "height_range": [
                    min(r["height_m"] for r in rows),
                    max(r["height_m"] for r in rows),
                ],
                "edge_gap_range": [
                    min(r["road_edge_inward_of_face_m"] for r in rows),
                    max(r["road_edge_inward_of_face_m"] for r in rows),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
