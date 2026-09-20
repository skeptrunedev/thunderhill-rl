# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "matplotlib==3.10.8", "pyproj==3.7.2"]
# ///
"""Build an experimental pit pavement envelope, never modifying runtime data.

The positive edge is a historical optical estimate. The negative edge is an
observed raised structure face candidate, not a surveyed wall base. Endpoint
joins are intentionally unresolved instead of blending into unsupported edges.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_road_surface import evaluate
from pyproj import Transformer, datadir
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/road-surface"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(road, station):
    result = evaluate(road, station, 0.0)
    along = result["Rs"][[0, 2]]
    along /= np.linalg.norm(along)
    return result["R"][[0, 2]], along, np.array([along[1], -along[0]])


def profile(raw, tree, road, station):
    center, along, left = frame(road, station)
    points = raw[tree.query_ball_point(center, 13)]
    delta = points[:, :2] - center
    t, u = delta @ along, delta @ left
    design = np.column_stack((np.ones(len(points)), t, u))
    ground = (abs(t) <= 3) & (u >= 0) & (u <= 3) & (points[:, 3] == 2)
    if ground.sum() < 30:
        raise ValueError(f"Insufficient interior plane support at {station}")
    fit = np.linalg.lstsq(design[ground], points[ground, 2], rcond=None)[0]
    residual = points[:, 2] - design @ fit
    strip = (abs(t) <= 1) & (u >= -7) & (u <= 0)
    sensitivity = []
    for lower in [0.15, 0.25, 0.35]:
        raised = strip & (residual >= lower) & (residual <= 1.5)
        edges = np.arange(-7, 0.001, 0.2)
        counts, _ = np.histogram(u[raised], edges)
        bins = np.flatnonzero(counts >= 2)
        groups = (
            np.split(bins, np.flatnonzero(np.diff(bins) > 1) + 1) if len(bins) else []
        )
        # Closest occupied structure band to the pavement interior, no guessed margin.
        if groups:
            band = groups[-1]
            selected = raised & (u >= edges[band[0]]) & (u <= edges[band[-1] + 1])
            lateral = u[selected]
            sensitivity.append(
                {
                    "minimum_height_m": lower,
                    "band_m": [float(edges[band[0]]), float(edges[band[-1] + 1])],
                    "count": int(selected.sum()),
                    "face_quantiles_90_95_100_m": np.quantile(
                        lateral, [0.9, 0.95, 1]
                    ).tolist(),
                    "source_ids": np.unique(points[selected, 4]).astype(int).tolist(),
                }
            )
        else:
            sensitivity.append({"minimum_height_m": lower, "count": 0})
    nominal = sensitivity[-1]
    face = nominal.get("face_quantiles_90_95_100_m", [None, None])[1]
    candidates = [
        q for row in sensitivity for q in row.get("face_quantiles_90_95_100_m", [])
    ]
    return {
        "station_m": float(station),
        "center_xz_m": center.tolist(),
        "left_xz": left.tolist(),
        "ground_count": int(ground.sum()),
        "ground_plane_rmse_m": float(np.sqrt(np.mean(residual[ground] ** 2))),
        "structure_face_candidate_m": face,
        "sensitivity_range_m": [min(candidates), max(candidates)]
        if candidates
        else None,
        "sensitivity": sensitivity,
    }


def plot_endpoints(data, profiles, envelope):
    """Retain source only panels and register explicit candidate geometry."""
    east = data["east_measurements"]
    grid_dir = ROOT / "artifacts/reference/datum"
    for name, sha in east["datum_grid_hashes"].items():
        if digest(grid_dir / name) != sha:
            raise ValueError("Datum registration grid changed")
    datadir.append_data_dir(str(grid_dir))
    transform = Transformer.from_pipeline(east["datum_operation"])
    image_path = ROOT / "artifacts/reference/ortho/east-2022.png"
    if digest(image_path) != east["image_sha256"]:
        raise ValueError("Aerial image changed")
    extent = json.loads((image_path.parent / "export.json").read_text())["response"][
        "extent"
    ]
    pin = json.loads((ROOT / "godot/assets/materials/terrain_macro.json").read_text())
    if extent != pin["source"]["extent"] or extent["spatialReference"]["wkid"] != 26910:
        raise ValueError("Aerial extent changed")
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    dx = (extent["xmax"] - extent["xmin"]) / width
    dy = (extent["ymax"] - extent["ymin"]) / height
    origin = data["track"]["origin"]

    def project(points):
        points = np.asarray(points)
        x, y = transform.transform(
            points[:, 0] + origin["easting"],
            origin["northing"] - points[:, 1],
            errcheck=True,
        )
        return np.column_stack([x, y])

    stripe_profiles = []
    for station in list(range(0, 41, 5)) + list(range(435, 481, 5)):
        center, _, left = frame(data["road"], station)
        offsets = np.arange(-8, 0.001, 0.15)
        rgb_profiles = []
        for along_offset in [-0.6, 0, 0.6]:
            c, _, l = frame(data["road"], station + along_offset)
            xy = project(c + offsets[:, None] * l)
            cols = np.floor((xy[:, 0] - extent["xmin"]) / dx).astype(int)
            rows = np.floor((extent["ymax"] - xy[:, 1]) / dy).astype(int)
            rgb_profiles.append(image[rows, cols, :3])
        rgb = np.median(rgb_profiles, axis=0)
        brightness = rgb.mean(axis=1)
        flank = 8  # 1.2m, greater than one source pixel
        score = np.full(len(offsets), -np.inf)
        score[flank:-flank] = (
            brightness[flank:-flank]
            - (brightness[: -2 * flank] + brightness[2 * flank :]) / 2
        )
        score[(offsets < -6) | (offsets > -1)] = -np.inf
        peak = int(np.argmax(score))
        stripe_profiles.append(
            {
                "station_m": station,
                "offsets_m": offsets.tolist(),
                "median_rgb": rgb.tolist(),
                "peak_offset_m": float(offsets[peak]),
                "peak_contrast_0_to_1": float(score[peak]),
                "peak_xz_m": (center + offsets[peak] * left).tolist(),
                "pixel_scale_interval_m": [
                    float(offsets[peak] - 0.6),
                    float(offsets[peak] + 0.6),
                ],
            }
        )
    outputs = []
    for start, end in [(0, 90), (390, 480)]:
        stations = np.arange(start, end + 1)
        centers = project([frame(data["road"], s)[0] for s in stations])
        lower, upper = centers.min(axis=0) - [22, 5], centers.max(axis=0) + [16, 5]
        midpoint = (lower + upper) / 2
        col0 = max(0, int(np.floor((lower[0] - extent["xmin"]) / dx)))
        col1 = min(width, int(np.ceil((upper[0] - extent["xmin"]) / dx)))
        row0 = max(0, int(np.floor((extent["ymax"] - upper[1]) / dy)))
        row1 = min(height, int(np.ceil((extent["ymax"] - lower[1]) / dy)))
        crop_extent = [
            extent["xmin"] + col0 * dx - midpoint[0],
            extent["xmin"] + col1 * dx - midpoint[0],
            extent["ymax"] - row1 * dy - midpoint[1],
            extent["ymax"] - row0 * dy - midpoint[1],
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 12), layout="constrained")
        for ax in axes:
            ax.imshow(
                image[row0:row1, col0:col1], extent=crop_extent, interpolation="nearest"
            )
            ax.set(
                xlabel="East relative to crop center (m)",
                ylabel="North relative to crop center (m)",
            )
        axes[0].set_title("Historical NAIP source only")
        axes[1].set_title("Observed candidate overlays, no extrapolated joins")
        for p in profiles:
            s = p["station_m"]
            if not start <= s <= end:
                continue
            center = np.array(p["center_xz_m"])
            if s % 10 == 0:
                xy = project([center])[0] - midpoint
                axes[1].annotate(str(int(s)), xy, color="yellow", fontsize=9)
            if p["structure_face_candidate_m"] is not None:
                xy = (
                    project(
                        [
                            center
                            + p["structure_face_candidate_m"] * np.array(p["left_xz"])
                        ]
                    )[0]
                    - midpoint
                )
                axes[1].plot(*xy, ".", color="cyan", markersize=3)
        for p in stripe_profiles:
            if start <= p["station_m"] <= end:
                xy = project([p["peak_xz_m"]])[0] - midpoint
                axes[1].plot(*xy, "x", color="lime", markersize=5)
        for e in envelope:
            if start <= e["station_m"] <= end:
                xy = project([e["left_xz_m"]])[0] - midpoint
                axes[1].plot(*xy, "o", color="red", markersize=4)
        output = OUT / f"pit-endpoint-{start}-{end}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(
            {
                "station_interval_m": [start, end],
                "path": str(output.relative_to(ROOT)),
                "sha256": digest(output),
                "crop_center_aerial_easting_northing": midpoint.tolist(),
            }
        )
    return {
        "stripe_profiles": stripe_profiles,
        "stripe_method": "Median RGB of three along profiles spaced0.6m; strongest brightness contrast versus flanks1.2m away within lateral -6..-1m. Green crosses are optical stripe hypotheses, not accepted track limits.",
        "image_sha256": digest(image_path),
        "extent": extent,
        "datum_operation": east["datum_operation"],
        "datum_grid_hashes": east["datum_grid_hashes"],
        "outputs": outputs,
    }


def main():
    paths = {
        "track": ROOT / "godot/data/track.json",
        "road": OUT / "road-surface.json",
        "divider_measurements": OUT / "pit-divider-measurements.json",
        "raw_local_points": OUT / "pit-divider-points.npz",
        "east_measurements": OUT / "pit-east-edge-measurements.json",
    }
    data = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    for name in ["divider_measurements", "east_measurements"]:
        for source in ["track", "road"]:
            if data[name][source + "_sha256"] != digest(paths[source]):
                raise ValueError(f"Stale {name} {source}")
    if data["road"]["metadata"]["track_sha256"] != digest(paths["track"]):
        raise ValueError("Stale road station frame")
    archive = np.load(paths["raw_local_points"])
    expected_columns = ["x", "z", "height_local", "classification", "source_id"]
    if archive["columns"].tolist() != expected_columns:
        raise ValueError("Unexpected raw point column order")
    raw = archive["points"]
    if raw.ndim != 2 or raw.shape[1] != 5 or not np.isfinite(raw).all():
        raise ValueError("Malformed raw point archive")
    if len(raw) != data["divider_measurements"]["observation_count"]:
        raise ValueError("Raw archive count differs from source measurements")
    tree = cKDTree(raw[:, :2])
    stations = np.unique(
        np.r_[np.arange(0, 501, 10), np.arange(35, 61), np.arange(400, 451)]
    )
    profiles = [profile(raw, tree, data["road"], s) for s in stations]
    by_station = {p["station_m"]: p for p in profiles}
    envelope = []
    for east in data["east_measurements"]["sections"]:
        s = east["station_m"]
        west = by_station[s]
        right = west["structure_face_candidate_m"]
        left = next(c for c in east["candidates"] if c["threshold"] == 0.07)[
            "center_first_sustained_brown_m"
        ]
        if right is None or left is None:
            continue
        interval = east["threshold_envelope_plus_one_pixel_m"]
        envelope.append(
            {
                "station_m": s,
                "left_offset_m": left,
                "right_offset_m": right,
                "width_m": left - right,
                "center_offset_left_m": (left + right) / 2,
                "left_sensitivity_plus_pixel_m": interval,
                "right_height_quantile_sensitivity_m": west["sensitivity_range_m"],
                "left_xz_m": (
                    np.array(west["center_xz_m"]) + left * np.array(west["left_xz"])
                ).tolist(),
                "right_xz_m": (
                    np.array(west["center_xz_m"]) + right * np.array(west["left_xz"])
                ).tolist(),
            }
        )
    present = [
        p["station_m"] for p in profiles if p["structure_face_candidate_m"] is not None
    ]
    endpoint_registration = plot_endpoints(data, profiles, envelope)
    report = {
        "endpoint_registration": endpoint_registration,
        "schema_version": 1,
        "status": "Experimental historical envelope; endpoint joins unresolved; not adopted",
        "sources": {
            k: {"path": str(v.relative_to(ROOT)), "sha256": digest(v)}
            for k, v in paths.items()
        },
        "builder_sha256": digest(Path(__file__)),
        "raw_tile_sources": data["divider_measurements"]["tiles"],
        "method": {
            "frame": "Existing baseline road station, positive offset is left/east",
            "profile_half_length_m": 1,
            "structure_height_thresholds_m": [0.15, 0.25, 0.35],
            "structure_height_ceiling_m": 1.5,
            "occupied_lateral_bin_m": 0.2,
            "minimum_bin_returns": 2,
            "right_candidate": "95th lateral percentile in closest occupied raised band at height >=0.35m",
            "left_candidate": "Aerial first sustained brown at threshold 0.07",
            "interpolation": "Only between supplied candidate anchors, no extrapolation or automatic endpoint blend",
        },
        "limitations": [
            "Raised face is observed above ground, not a measured wall base or legal track limit",
            "Quantile and height threshold spread is sensitivity, not absolute accuracy",
            "Aerial edge has at least 0.6m pixel scale uncertainty and unquantified registration error",
            "Endpoints require pavement continuation evidence; absence of raised returns alone does not establish track edge",
            "Current source station coordinates differ from OSM source_s; apply geometric endpoints to baseline samples before recomputing station, never feed directly into reviewed_widths",
            "Frame comes from smooth experimental road; use explicit xz endpoints to avoid assuming it is identical to baseline piecewise linear frame",
        ],
        "raised_structure_station_span_m": [min(present), max(present)],
        "candidate_anchors": envelope,
        "profiles": profiles,
    }
    (OUT / "pit-envelope-candidate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax in axes:
        for threshold in [0.15, 0.25, 0.35]:
            values = [
                next(
                    v for v in p["sensitivity"] if v["minimum_height_m"] == threshold
                ).get("face_quantiles_90_95_100_m", [None, None])[1]
                for p in profiles
            ]
            ax.plot(
                stations, values, ".-", label=f"Raised face minimum height {threshold}m"
            )
        ax.plot(
            [e["station_m"] for e in envelope],
            [e["left_offset_m"] for e in envelope],
            "o-",
            label="Optical east edge",
        )
        ax.axhline(-6, color="gray", linestyle=":", label="Old nominal edges")
        ax.axhline(6, color="gray", linestyle=":")
        ax.set(xlabel="Baseline station (m)", ylabel="Lateral offset left (m)")
        ax.grid(alpha=0.2)
    axes[0].set_xlim(0, 500)
    axes[0].legend(loc="center", ncol=2)
    axes[1].set_xlim(390, 450)
    axes[1].set_ylim(-7, 0)
    fig.suptitle(
        "Experimental pit pavement envelope; observed raised face is not a surveyed wall base"
    )
    fig.savefig(OUT / "pit-envelope-candidate.png", dpi=150)
    print(
        json.dumps(
            {
                "anchors": len(envelope),
                "raised_span_m": report["raised_structure_station_span_m"],
                "width_range_m": [
                    min(e["width_m"] for e in envelope),
                    max(e["width_m"] for e in envelope),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
