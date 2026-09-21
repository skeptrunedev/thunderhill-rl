# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "matplotlib==3.10.8"]
# ///
"""Compare the production or experimental road with pinned historical aerial evidence.

This produces review artifacts, never edits track geometry or claims surveyed
edges. Run after build_road_surface.py. The plot retains a source-only panel.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_road_surface import evaluate
from pyproj import datadir
from pyproj.transformer import TransformerGroup
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth_plan_candidate(track, start, end, sigma_m):
    """Local appearance candidate only; original stations and vertical data remain.

    Filter in uniform station space, taper displacement with a C2 weight, and
    report the actual movement. This does not produce contact or surveyed data.
    """
    if not np.isfinite(sigma_m) or sigma_m <= 0 or sigma_m > 20:
        raise ValueError("Plan smoothing sigma must be positive and at most 20 m")
    length = float(track["length_m"])
    if not 0 <= start < end <= length or end - start <= 60:
        raise ValueError("Plan study requires an interval longer than 60 m")
    rows = track["samples"]
    stations = np.array([r["s"] for r in rows])
    points = np.array([r["p"] for r in rows])[:, [0, 2]]
    count = int(np.ceil(length / 0.5))
    step = length / count
    uniform = np.arange(count) * step
    dense = np.column_stack(
        [np.interp(uniform, stations, points[:, i], period=length) for i in range(2)]
    )
    filtered = gaussian_filter1d(dense, sigma_m / step, axis=0, mode="wrap")
    target = np.column_stack(
        [np.interp(stations, uniform, filtered[:, i], period=length) for i in range(2)]
    )
    weight = np.clip(np.minimum(stations - start, end - stations) / 30.0, 0, 1)
    weight = weight**3 * (weight * (weight * 6 - 15) + 10)
    candidate = points + (target - points) * weight[:, None]
    movement = np.linalg.norm(candidate - points, axis=1)
    selected = (stations >= start) & (stations <= end)

    def heading_change(xz):
        delta = np.roll(xz, -1, axis=0) - xz
        before = np.roll(delta, 1, axis=0)
        return np.degrees(
            np.arctan2(
                before[:, 0] * delta[:, 1] - before[:, 1] * delta[:, 0],
                np.sum(before * delta, axis=1),
            )
        )

    return candidate, {
        "sigma_m": sigma_m,
        "transition_m": 30.0,
        "uniform_step_m": step,
        "maximum_center_displacement_m": float(movement.max()),
        "maximum_baseline_heading_change_deg": float(
            np.abs(heading_change(points)[selected]).max()
        ),
        "maximum_candidate_heading_change_deg": float(
            np.abs(heading_change(candidate)[selected]).max()
        ),
        "stations_m": stations[selected].tolist(),
        "candidate_xz_m": candidate[selected].tolist(),
        "status": "Horizontal study only, not accepted geometry; contact, elevation and dependent assets must be rebuilt before use",
    }


def fit_height_graph(offsets, heights, direction):
    """Quadratic height fit and directional normal curvature at its origin."""
    offsets, heights, direction = map(np.asarray, (offsets, heights, direction))
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 2
        or heights.shape != (len(offsets),)
        or direction.shape != (2,)
        or len(heights) < 20
        or not all(np.isfinite(a).all() for a in (offsets, heights, direction))
        or np.linalg.norm(direction) <= 0
    ):
        raise ValueError("Invalid local height observations or direction")
    direction = direction / np.linalg.norm(direction)
    dx, dz = offsets.T
    design = np.column_stack(
        [np.ones(len(dx)), dx, dz, dx * dx / 2, dx * dz, dz * dz / 2]
    )
    fit, _, rank, singular = np.linalg.lstsq(design, heights, rcond=None)
    if rank != 6:
        raise ValueError("Rank deficient local lidar fit")
    gradient = fit[1:3]
    hessian = np.array([[fit[3], fit[4]], [fit[4], fit[5]]])
    curvature = float(
        direction
        @ hessian
        @ direction
        / (np.sqrt(1 + gradient @ gradient) * (1 + (gradient @ direction) ** 2))
    )
    return {
        "height_m": float(fit[0]),
        "gradient": gradient.tolist(),
        "hessian": hessian.tolist(),
        "point_count": len(heights),
        "design_condition_number": float(singular[0] / singular[-1]),
        "rmse_m": float(np.sqrt(np.mean((design @ fit - heights) ** 2))),
        "lidar_directional_normal_curvature_per_m": curvature,
    }


def lidar_checks(surface, track, extrema):
    source = ROOT / "artifacts/reference/lidar/road-ground-points.npz"
    if digest(source) != track["metadata"]["lidar"]["corridor_sha256"]:
        raise ValueError("Raw lidar differs from the track source")
    points = np.load(source)["points"]
    tree = cKDTree(points[:, :2])
    origin = track["origin"]
    results = []
    for label, location in extrema.items():
        sample = evaluate(surface, location["station"], location["lateral"])
        p = sample["R"]
        world = np.array([p[0] + origin["easting"], origin["northing"] - p[2]])
        direction = sample["Rs"][[0, 2]]
        direction /= np.linalg.norm(direction)
        left = sample["Ru"][[0, 2]]
        left /= np.linalg.norm(left)
        for radius in [1.5, 2.0, 3.0, 4.0, 6.0]:
            neighbors = points[tree.query_ball_point(world, radius)]
            offsets = (neighbors[:, :2] - world) * [1, -1]
            for selection in ["full_neighborhood", "assumed_pavement_side"]:
                mask = np.ones(len(neighbors), dtype=bool)
                if selection == "assumed_pavement_side":
                    # Deliberately a separate sensitivity check: this is not a
                    # surveyed pavement mask, nor an accepted boundary fit.
                    mask = offsets @ left * np.sign(location["lateral"]) < -0.25
                heights = neighbors[mask, 2] - origin["elevation_m"]
                fitted = fit_height_graph(offsets[mask], heights, direction)
                results.append(
                    {
                        "extreme": label,
                        "station_m": location["station"],
                        "lateral_m": location["lateral"],
                        "radius_m": radius,
                        "selection": selection,
                        **fitted,
                        "ribbon_normal_curvature_per_m": location["curvature"],
                        "ribbon_minus_lidar_height_excluding_lift_m": float(
                            p[1] - surface["road_lift_m"] - fitted["height_m"]
                        ),
                    }
                )
    return {
        "source_sha256": digest(source),
        "fits": results,
        "limitations": "Local quadratic fits are scale dependent. Full neighborhoods can contain curb and soil. The interior half is defined by provisional ribbon geometry, not a surveyed pavement boundary. These are sensitivity checks, not accepted replacement curvature.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=1750.0)
    parser.add_argument("--end", type=float, default=1880.0)
    parser.add_argument(
        "--production",
        action="store_true",
        help="Audit the shipped piecewise linear center and edge frame",
    )
    parser.add_argument(
        "--plan-smoothing-m",
        type=float,
        default=0.0,
        help="Overlay a local horizontal smoothing study; never modifies game data",
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=ROOT / "artifacts/road-surface/road-surface.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/road-surface/cyclone-alignment.png",
    )
    parser.add_argument(
        "--download-datum-grids",
        action="store_true",
        help="Acquire openly licensed PROJ datum grids into this project's reference artifacts",
    )
    args = parser.parse_args()
    if not np.isfinite(args.plan_smoothing_m) or args.plan_smoothing_m < 0:
        raise ValueError("Plan smoothing must be finite and nonnegative")
    if args.plan_smoothing_m and not args.production:
        raise ValueError("Plan smoothing requires --production")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite alignment evidence")
    track_path = ROOT / "godot/data/track.json"
    track = json.loads(track_path.read_text())
    surface = None if args.production else json.loads(args.surface.read_text())
    if surface is not None and surface["metadata"]["track_sha256"] != digest(
        track_path
    ):
        raise ValueError("Surface was generated from a different track")
    period = track["length_m"] if args.production else surface["period_m"]
    if not 0 <= args.start < args.end <= period:
        raise ValueError("Choose an ordered interval within the track period")
    image_path = ROOT / "artifacts/reference/ortho/east-2022.png"
    source_pin = json.loads(
        (ROOT / "godot/assets/materials/terrain_macro.json").read_text()
    )
    if digest(image_path) != source_pin["source"]["image_sha256"]:
        raise ValueError("Historical aerial image differs from its pinned source")
    extent = json.loads((image_path.parent / "export.json").read_text())["response"][
        "extent"
    ]
    if extent["spatialReference"]["wkid"] != 26910:
        raise ValueError("Expected aerial image in EPSG:26910")
    if extent != source_pin["source"]["extent"]:
        raise ValueError("Aerial extent differs from the pinned export")
    # build_track.read_route explicitly uses EPSG:6339. Do not silently equate
    # NAD83(2011) and generic NAD83; report the operation actually available.
    grid_pins = json.loads((ROOT / "data/reference/datum-grids.json").read_text())
    grid_dir = ROOT / "artifacts/reference/datum"
    if grid_dir.is_dir():
        datadir.append_data_dir(str(grid_dir))
    transforms = TransformerGroup(6339, 26910, always_xy=True)
    if args.download_datum_grids and not transforms.best_available:
        grid_dir.mkdir(parents=True, exist_ok=True)
        transforms.download_grids(directory=grid_dir, open_license=True, verbose=True)
        datadir.append_data_dir(str(grid_dir))
        transforms = TransformerGroup(6339, 26910, always_xy=True)
        if not transforms.best_available:
            raise ValueError("Requested best datum transformation remains unavailable")
    if not transforms.best_available or not transforms.transformers:
        raise ValueError(
            "Best datum transformation unavailable; run with --download-datum-grids"
        )
    transform = transforms.transformers[0]
    used_grids = []
    pins = {grid["name"]: grid for grid in grid_pins["grids"]}
    for operation in transform.operations:
        for grid in operation.grids:
            grid_hash = digest(Path(grid.full_name))
            if (
                grid.short_name not in pins
                or grid_hash != pins[grid.short_name]["sha256"]
            ):
                raise ValueError(
                    "Datum grid changed; review source before accepting registration"
                )
            used_grids.append(
                {
                    "name": grid.short_name,
                    "url": grid.url,
                    "open_license": grid.open_license,
                    "sha256": grid_hash,
                }
            )
    if {grid["name"] for grid in used_grids} != set(pins):
        raise ValueError("Datum operation uses a different grid set")
    origin = track["origin"]

    def projected(points):
        points = np.asarray(points)
        east, north = transform.transform(
            points[:, 0] + origin["easting"],
            origin["northing"] - points[:, 2],
            errcheck=True,
        )
        return np.column_stack([east, north])

    stations = np.linspace(
        args.start, args.end, int(np.ceil((args.end - args.start) / 0.25)) + 1
    )
    observed_s = [row["s"] for row in track["samples"]] + [period]
    widths = np.interp(
        stations,
        observed_s,
        [row["width"] for row in track["samples"]] + [track["samples"][0]["width"]],
    )
    ribbons = {}
    points = np.array([row["p"] for row in track["samples"]])
    plan_study = None
    if args.plan_smoothing_m:
        candidate, plan_study = smooth_plan_candidate(
            track, args.start, args.end, args.plan_smoothing_m
        )
        points[:, [0, 2]] = candidate
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    left = np.column_stack([tangent[:, 2], np.zeros(len(points)), -tangent[:, 0]])
    left /= np.linalg.norm(left, axis=1)[:, None]
    sampled_width = np.array([row["width"] for row in track["samples"]])
    bank = np.tan([row["bank"] for row in track["samples"]])

    def production_points(query, side):
        offset = side * sampled_width * 0.5
        edge = points + left * offset[:, None]
        edge[:, 1] += bank * offset + 0.04
        edge = np.vstack([edge, edge[0]])
        return np.column_stack(
            [np.interp(query, observed_s, edge[:, axis]) for axis in range(3)]
        )

    for name, sign in [("center", 0), ("left", 1), ("right", -1)]:
        ribbons[name] = projected(
            production_points(stations, sign)
            if args.production
            else [
                evaluate(surface, s, sign * width / 2)["R"]
                for s, width in zip(stations, widths)
            ]
        )
    joined = np.concatenate(list(ribbons.values()))
    lower, upper = joined.min(axis=0) - 18, joined.max(axis=0) + 18
    midpoint = (lower + upper) / 2
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    dx, dy = (
        (extent["xmax"] - extent["xmin"]) / width,
        (extent["ymax"] - extent["ymin"]) / height,
    )
    col0 = max(0, int(np.floor((lower[0] - extent["xmin"]) / dx)))
    col1 = min(width, int(np.ceil((upper[0] - extent["xmin"]) / dx)))
    row0 = max(0, int(np.floor((extent["ymax"] - upper[1]) / dy)))
    row1 = min(height, int(np.ceil((extent["ymax"] - lower[1]) / dy)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("Requested interval is outside aerial image")
    crop_extent = [
        extent["xmin"] + col0 * dx - midpoint[0],
        extent["xmin"] + col1 * dx - midpoint[0],
        extent["ymax"] - row1 * dy - midpoint[1],
        extent["ymax"] - row0 * dy - midpoint[1],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), layout="constrained")
    for ax in axes:
        ax.imshow(image[row0:row1, col0:col1], extent=crop_extent, origin="upper")
        ax.set_xlabel("East relative to crop centre (m)")
        ax.set_ylabel("North relative to crop centre (m)")
        ax.set_aspect("equal")
    axes[0].set_title("Historical NAIP, July 2022, no overlay")
    axes[1].set_title(
        "Plan smoothing study"
        if plan_study
        else "Production road"
        if args.production
        else "Experimental ribbon, source alignment audit"
    )
    for name, color in [
        ("center", "#ffee32"),
        ("left", "#ff4848"),
        ("right", "#21e8ff"),
    ]:
        xy = ribbons[name] - midpoint
        axes[1].plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.1, label=name)
    for station in np.arange(np.ceil(args.start / 10) * 10, args.end, 10):
        center = (
            production_points([station], 0)
            if args.production
            else [evaluate(surface, station, 0)["R"]]
        )
        xy = projected(center)[0] - midpoint
        axes[1].plot(*xy, "o", color="#ffee32", markersize=2)
        axes[1].annotate(
            str(int(station)),
            xy,
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1},
        )
    axes[1].legend(loc="upper right")
    fig.suptitle(
        "Thunderhill East: pavement boundaries are provisional, imagery is not a survey"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    report = {
        "track_sha256": digest(track_path),
        "geometry": "production piecewise linear edge frame"
        if args.production
        else "experimental differentiable ribbon",
        "plan_study": plan_study,
        "surface_sha256": None if args.production else digest(args.surface),
        "track_script_sha256": digest(ROOT / "godot/scripts/track.gd")
        if args.production
        else None,
        "aerial_sha256": digest(image_path),
        "station_interval_m": [args.start, args.end],
        "track_crs": "EPSG:6339, from build_track.read_route",
        "aerial_crs": "EPSG:26910",
        "transformation": transform.description,
        "transform_accuracy_m": transform.accuracy,
        "best_available": transforms.best_available,
        "unavailable_operations": [
            operation.name for operation in transforms.unavailable_operations
        ],
        "transformation_definition": transform.definition,
        "origin_datum_shift_m": (
            projected([[0, 0, 0]])[0] - [origin["easting"], origin["northing"]]
        ).tolist(),
        "datum_grids": used_grids,
        "crop_center_aerial_easting_northing": midpoint.tolist(),
        "plot_sha256": digest(args.output),
        "limitations": "Visual registration aid only. Historic color boundaries, source georeferencing and datum operations have uncertainty. No edge measurements accepted automatically.",
    }
    if not args.production:
        surface_audit = json.loads(args.surface.with_suffix(".audit.json").read_text())
        report["lidar_checks"] = lidar_checks(
            surface, track, surface_audit["curvature_extrema"]
        )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
