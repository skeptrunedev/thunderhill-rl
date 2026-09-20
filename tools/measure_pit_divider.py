# /// script
# requires-python = ">=3.11"
# dependencies = ["laspy[lazrs]==2.6.1", "numpy==2.4.3", "scipy==1.17.1", "matplotlib==3.10.8", "pyproj==3.7.2"]
# ///
"""Measure raised pit straight returns separately from pavement and shadows.

Review evidence only. Thresholds identify candidate structures, not legal track
edges or a surveyed barrier envelope. No game data is modified.
"""

import hashlib
import json
from pathlib import Path

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_road_surface import evaluate
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/road-surface"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_frame(road, station):
    r = evaluate(road, station, 0.0)
    center = r["R"][[0, 2]]
    along = r["Rs"][[0, 2]]
    along /= np.linalg.norm(along)
    return center, along, np.array([along[1], -along[0]])


def main():
    track_path = ROOT / "godot/data/track.json"
    track = json.loads(track_path.read_text())
    road_path = OUT / "road-surface.json"
    road = json.loads(road_path.read_text())
    if road["metadata"]["track_sha256"] != digest(track_path):
        raise ValueError("Stale road")
    o = track["origin"]
    stations = np.arange(0.0, 501.0, 10.0)
    centers = np.array([local_frame(road, s)[0] for s in stations])
    lower, upper = centers.min(axis=0) - 18, centers.max(axis=0) + 18
    pins = json.loads((ROOT / "data/reference/lidar-measurements.json").read_text())
    used = []
    rows = []
    for tile in pins["tiles"]:
        path = ROOT / "artifacts/reference/lidar" / tile["path"]
        with laspy.open(path) as reader:
            lo = reader.header.mins
            hi = reader.header.maxs
            tile_lower = np.array([lo[0] - o["easting"], o["northing"] - hi[1]])
            tile_upper = np.array([hi[0] - o["easting"], o["northing"] - lo[1]])
            if np.any(tile_upper < lower) or np.any(tile_lower > upper):
                continue
            if digest(path) != tile["sha256"]:
                raise ValueError("Raw lidar tile changed")
            used.append(
                {
                    "path": tile["path"],
                    "sha256": tile["sha256"],
                    "crs": str(reader.header.parse_crs()),
                }
            )
            for chunk in reader.chunk_iterator(1_000_000):
                xz = np.column_stack(
                    (
                        np.asarray(chunk.x) - o["easting"],
                        o["northing"] - np.asarray(chunk.y),
                    )
                )
                keep = np.all((xz >= lower) & (xz <= upper), axis=1) & (
                    np.asarray(chunk.withheld) == 0
                )
                if np.any(keep):
                    rows.append(
                        np.column_stack(
                            (
                                xz[keep],
                                np.asarray(chunk.z)[keep] - o["elevation_m"],
                                np.asarray(chunk.classification)[keep],
                                np.asarray(chunk.point_source_id)[keep],
                            )
                        )
                    )
        print("Extracted", path.name, flush=True)
    raw = np.concatenate(rows)
    tree = cKDTree(raw[:, :2])
    sections = []
    all_scatter = []
    for s in stations:
        center, along, left = local_frame(road, s)
        ids = tree.query_ball_point(center, 15)
        points = raw[ids]
        offset = points[:, :2] - center
        t = offset @ along
        a = offset @ left
        core = (np.abs(t) <= 3) & (a >= 0) & (a <= 3) & (points[:, 3] == 2)
        if sum(core) < 30:
            sections.append(
                {
                    "station_m": float(s),
                    "status": "insufficient interior ground support",
                    "ground_count": int(sum(core)),
                }
            )
            continue
        design = np.column_stack((np.ones(len(points)), t, a))
        plane = np.linalg.lstsq(design[core], points[core, 2], rcond=None)[0]
        residual = points[:, 2] - design @ plane
        strip = (np.abs(t) <= 2) & (a >= -12) & (a <= 8)
        candidate = strip & (residual >= 0.35) & (residual <= 1.5)
        edges = np.arange(-12, 8.001, 0.2)
        counts, _ = np.histogram(a[candidate], edges)
        bins = np.flatnonzero(counts >= 3)
        groups = (
            np.split(bins, np.flatnonzero(np.diff(bins) > 1) + 1) if len(bins) else []
        )
        bands = [
            {
                "lateral_min_m": float(edges[g[0]]),
                "lateral_max_m": float(edges[g[-1] + 1]),
                "return_count": int(sum(counts[g])),
            }
            for g in groups
        ]
        selected = points[strip]
        # Retain actual points for individual review and unthresholded plots.
        all_scatter.append(
            np.column_stack(
                (
                    np.full(sum(strip), s),
                    a[strip],
                    residual[strip],
                    selected[:, 3],
                    selected[:, 4],
                )
            )
        )
        sections.append(
            {
                "station_m": float(s),
                "center_xz_m": center.tolist(),
                "along": along.tolist(),
                "left": left.tolist(),
                "ground_count": int(sum(core)),
                "ground_plane_coefficients": plane.tolist(),
                "ground_rmse_m": float(np.sqrt(np.mean(residual[core] ** 2))),
                "raised_return_count": int(sum(candidate)),
                "candidate_bands": bands,
            }
        )
    scatter = np.concatenate(all_scatter)
    report = {
        "status": "Candidate raised structure bands for review, not adopted track limits",
        "track_sha256": digest(track_path),
        "road_sha256": digest(road_path),
        "source_code_sha256": digest(Path(__file__)),
        "tiles": used,
        "observation_count": len(raw),
        "method": {
            "station_spacing_m": 10,
            "strip_along_halfwidth_m": 2,
            "reference_plane": "Class2 ground, along +/-3m, lateral0..3m",
            "candidate_height_above_plane_m": [0.35, 1.5],
            "lateral_bin_width_m": 0.2,
            "minimum_returns_per_bin": 3,
        },
        "sections": sections,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pit-divider-measurements.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(
        OUT / "pit-divider-points.npz",
        points=raw,
        columns=["x", "z", "height_local", "classification", "source_id"],
    )
    np.savez_compressed(
        OUT / "pit-divider-sections.npz",
        points=scatter,
        columns=[
            "station",
            "lateral",
            "height_above_interior_plane",
            "classification",
            "source_id",
        ],
    )
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), layout="constrained")
    visible = (scatter[:, 2] >= -0.15) & (scatter[:, 2] <= 1.5)
    p = axes[0].scatter(
        scatter[visible, 0],
        scatter[visible, 1],
        c=scatter[visible, 2],
        s=2,
        vmin=0,
        vmax=1.2,
        cmap="viridis",
    )
    axes[0].set(
        ylabel="Lateral offset (m)",
        title="All returns within height display range, no optical shadow classifier",
    )
    fig.colorbar(p, ax=axes[0], label="Height above interior ground plane (m)")
    for section in sections:
        for band in section.get("candidate_bands", []):
            axes[1].plot(
                [section["station_m"]] * 2,
                [band["lateral_min_m"], band["lateral_max_m"]],
                color="black",
                linewidth=3,
            )
    axes[1].set(
        xlabel="Track station (m)",
        ylabel="Lateral offset (m)",
        title="Raised return bands: 0.35 to 1.5m above plane, at least3 returns per0.2m bin",
    )
    for ax in axes:
        ax.axhline(-6, color="red", linestyle=":", label="Nominal right road edge")
        ax.set_ylim(-12, 8)
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "pit-divider-measurements.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "sections": len(sections),
                "observations": len(raw),
                "output": str(OUT / "pit-divider-measurements.png"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
