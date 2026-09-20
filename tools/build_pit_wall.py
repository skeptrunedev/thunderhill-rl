# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1"]
# ///
"""Place a provisional visual divider against the adopted pavement interpretation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from build_atlas_road_mesh import planar_edges

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    track_path = ROOT / "godot/data/track.json"
    profile_path = ROOT / "data/reference/pit-wall-profile.json"
    track = json.loads(track_path.read_text())
    profile = json.loads(profile_path.read_text())
    if profile["sources"]["corrected_track"]["sha256"] != digest(track_path):
        raise ValueError("Wall observations refer to a different road interpretation")
    if profile["origin"] != track["origin"]:
        raise ValueError("Wall origin mismatch")
    p, right, left, _, _ = planar_edges(track)
    source_s = np.array([s["source_s"] for s in track["samples"]])
    rows = profile["rows"]
    observed_s = np.array([r["source_s_m"] for r in rows])
    heights = np.array([r["height_m"] for r in rows])
    if (
        np.any(np.diff(observed_s) <= 0)
        or not np.isfinite(heights).all()
        or np.any(heights <= 0)
    ):
        raise ValueError("Invalid measured height sequence")
    query = np.r_[
        observed_s[0],
        source_s[(source_s > observed_s[0]) & (source_s < observed_s[-1])],
        observed_s[-1],
    ]
    face = np.column_stack(
        [np.interp(query, source_s, right[:, axis]) for axis in range(2)]
    )
    direction = left - p
    inward = np.column_stack(
        [np.interp(query, source_s, direction[:, axis]) for axis in range(2)]
    )
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    output = {
        "schema_version": 1,
        "origin": track["origin"],
        "track_sha256": digest(track_path),
        "source_profile_sha256": digest(profile_path),
        "builder_sha256": digest(Path(__file__)),
        "replaces_osm_way": 896075617,
        "visual_width_m": profile["method"]["visual_width_m"],
        "rows": [
            {
                "face_xz_m": f.tolist(),
                "road_inward_xz": n.tolist(),
                "height_m": float(h),
                "source_s_m": float(s),
            }
            for s, f, n, h in zip(
                query, face, inward, np.interp(query, observed_s, heights)
            )
        ],
        "interpretation": "Face follows adopted provisional road right edge at every road vertex. This resolves visual construction consistently, not an independent wall survey. Measured relative top heights are interpolated in original source station coordinates.",
        "limitations": [
            "Width remains the existing artistic 0.35 metre estimate",
            "Height is relative to fitted interior pavement, transferred to current rendered ground",
            "Independent raw face disagreements remain in the reference profile",
            "Visual geometry only; crash collision not modeled",
        ],
        "license": "Road derived positions ODbL 1.0, OpenStreetMap contributors; public domain USGS lidar heights",
    }
    path = ROOT / "godot/data/pit-wall.json"
    path.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "rows": len(query),
                "height_range_m": [float(heights.min()), float(heights.max())],
                "sha256": digest(path),
            }
        )
    )


if __name__ == "__main__":
    main()
