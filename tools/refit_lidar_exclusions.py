# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "shapely==2.1.2", "rasterio==1.4.4"]
# ///
"""Refit only atlas patches intersecting reviewed structural exclusions.

Experimental artifact only. The resulting road beneath occluding structures is
inferred from surrounding observations, not measured pavement.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from build_lidar_atlas import fit_patch, initialize_fit_worker
from build_surface_mesh import road_edges
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def affected_patches(patches, exclusions, halfwidth=16):
    return [
        i
        for i, patch in enumerate(patches)
        if box(
            patch["origin"][0] - halfwidth,
            patch["origin"][1] - halfwidth,
            patch["origin"][0] + halfwidth,
            patch["origin"][1] + halfwidth,
        ).intersects(exclusions)
    ]


def load_exclusions(path, track, track_hash):
    data = json.loads(path.read_text())
    basis = data["coordinate_basis"]
    if (
        data["schema_version"] != 1
        or data["status"] != "reviewed-structure-occlusion-not-surveyed-ground"
        or basis["frame"] != "Godot local x,z meters"
        or basis["source_horizontal_crs"] != "EPSG:6339"
        or basis["origin"] != track["origin"]
        or basis["track_sha256"] != track_hash
        or basis["x"] != "easting minus origin easting"
        or basis["z"] != "origin northing minus northing"
    ):
        raise ValueError("Exclusion coordinate provenance mismatch")
    polygons = []
    for entry in data["exclusions"]:
        source = entry["source"]
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Exclusion raw source changed")
        coords = np.asarray(entry["proposed_exclusion_polygon_xz_m"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
            raise ValueError("Invalid exclusion coordinates")
        polygon = Polygon(coords)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            raise ValueError("Invalid exclusion polygon")
        polygons.append(polygon)
    if not polygons:
        raise ValueError("No reviewed exclusions")
    return shapely.union_all(polygons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=ROOT / "data/reference/pavement-exclusions.json",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.atlas.resolve():
        raise ValueError("Output must not replace parent atlas")
    parent = json.loads(args.atlas.read_text())
    metadata = parent["metadata"]
    if metadata["source_code_sha256"] != digest(ROOT / "tools/build_lidar_atlas.py"):
        raise ValueError("Parent fitting implementation changed")
    if (
        parent["schema_version"] != 1
        or parent["support_radius_m"] != 12
        or metadata["knot_spacing_m"] != 2
        or metadata["lambda_m4"] != 0.01
        or metadata["fit_halfwidth_m"] != 16
        or metadata["diagnostic_limit"] != 0
        or metadata.get("pavement_mask") is None
        or "structural_exclusion_refit" in metadata
    ):
        raise ValueError(
            "Parent must be a full unmodified masked atlas with expected settings"
        )
    paths = {
        "track": ROOT / "godot/data/track.json",
        "road": ROOT / "artifacts/road-surface/road-surface.json",
        "lidar": ROOT / "artifacts/reference/lidar/road-ground-points.npz",
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    if hashes != metadata["provenance_sha256"]:
        raise ValueError("Parent source provenance mismatch")
    track = json.loads(paths["track"].read_text())
    road = json.loads(paths["road"].read_text())
    if (
        road["metadata"]["track_sha256"] != hashes["track"]
        or track["metadata"]["lidar"]["corridor_sha256"] != hashes["lidar"]
    ):
        raise ValueError("Track and road source provenance mismatch")
    exclusion = load_exclusions(args.exclusions, track, hashes["track"])
    mask_info = metadata["pavement_mask"]
    if digest(ROOT / "tools/build_surface_mesh.py") != mask_info["builder_sha256"]:
        raise ValueError("Parent mask builder changed")
    footprint, _, _ = road_edges(track)
    footprint = footprint.buffer(mask_info["buffer_m"])
    if hashlib.sha256(footprint.wkb).hexdigest() != mask_info["polygon_wkb_sha256"]:
        raise ValueError("Parent mask polygon differs")
    raw = np.load(paths["lidar"])["points"]
    origin = track["origin"]
    xz = np.column_stack(
        (raw[:, 0] - origin["easting"], origin["northing"] - raw[:, 1])
    )
    heights = raw[:, 2] - origin["elevation_m"]
    mask = shapely.contains_xy(footprint, xz[:, 0], xz[:, 1])
    if (
        len(mask) != mask_info["input_points"]
        or int(sum(mask)) != mask_info["selected_points"]
    ):
        raise ValueError("Parent observation selection differs")
    xz, heights = xz[mask], heights[mask]
    removed = shapely.intersects_xy(exclusion, xz[:, 0], xz[:, 1])
    parent_tree = cKDTree(xz)
    kept_xz, kept_heights = xz[~removed], heights[~removed]
    tree = cKDTree(kept_xz)
    indices = affected_patches(parent["patches"], exclusion)
    if not indices or not np.any(removed):
        raise ValueError("Reviewed exclusions remove no supported observations")
    output = copy.deepcopy(parent)
    reports = []
    initialize_fit_worker()
    for index in indices:
        center = np.array(parent["patches"][index]["origin"])
        ids = np.array(tree.query_ball_point(center, 16, p=np.inf), dtype=int)
        blocks = np.floor(kept_xz[ids] / 2).astype(int)
        held = (blocks[:, 0] + 2 * blocks[:, 1]) % 3 == 0
        patch, report = fit_patch(
            (center, kept_xz[ids] - center, kept_heights[ids], held)
        )
        output["patches"][index] = patch
        report["index"] = index
        report["parent_points"] = len(
            parent_tree.query_ball_point(center, 16, p=np.inf)
        )
        reports.append(report)
        print(
            json.dumps(
                {"index": index, "points": len(ids), "heldout": report["heldout"]}
            ),
            flush=True,
        )
    untouched = sorted(set(range(len(parent["patches"]))) - set(indices))
    if any(parent["patches"][i] != output["patches"][i] for i in untouched):
        raise ValueError("Untouched patch changed")
    provenance = {
        "parent_atlas_sha256": digest(args.atlas),
        "exclusions_sha256": digest(args.exclusions),
        "source_code_sha256": digest(Path(__file__)),
        "fit_builder_sha256": digest(ROOT / "tools/build_lidar_atlas.py"),
        "affected_indices": indices,
        "affected_patch_count": len(indices),
        "unchanged_patch_count": len(untouched),
        "unchanged_patches_exactly_equal": True,
        "removed_observations": int(sum(removed)),
        "remaining_observations": len(kept_xz),
        "interpretation": "Road under structure is inferred from surrounding observations, not surveyed ground. Exclusion margin is interpretive, not physical barrier geometry.",
    }
    output["metadata"]["structural_exclusion_refit"] = provenance
    output["metadata"]["limitations"].append(provenance["interpretation"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    args.output.with_suffix(".audit.json").write_text(
        json.dumps({"refit": provenance, "patches": reports}, indent=2) + "\n"
    )
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
