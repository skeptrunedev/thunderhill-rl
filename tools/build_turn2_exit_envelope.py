# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3"]
# ///
"""Build an isolated historical exit edge hypothesis, never install game data.

Offsets interpret the registered cross sections and curb audit. Endpoint tapers
are continuity constraints, not measured edges. This manifest requires review.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = "a9ad305331e90dc90b7aed703b3ff7f4d850dabd580ed6a84d826a8ce4a93c04"
REQUIRED_STATIONS = {1260, 1280, 1300, 1320, 1340, 1380, 1420}
# Station on pinned production baseline, inward movement of the right edge.
ANCHORS = [
    (1230, 0),
    (1250, 0.5),
    (1260, 0.8),
    (1270, 1.2),
    (1280, 1.4),
    (1300, 1.4),
    (1320, 1.4),
    (1360, 1.4),
    (1380, 1.0),
    (1400, 0.5),
    (1420, 0),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(track, baseline_hash, sources):
    if baseline_hash != BASELINE_SHA256:
        raise ValueError(
            "Fixed station hypothesis requires its original production baseline"
        )
    rows = track["samples"]
    stations = np.array([r["s"] for r in rows])
    points = np.array([r["p"] for r in rows])[:, [0, 2]]
    half = np.array([r["width"] / 2 for r in rows])
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    left = points + normal * half[:, None]
    right = points - normal * half[:, None]

    def at(values, station):
        return np.array(
            [
                np.interp(station, stations, values[:, axis], period=track["length_m"])
                for axis in range(2)
            ]
        )

    anchors = []
    for station, movement in ANCHORS:
        bl, br = at(left, station), at(right, station)
        across = bl - br
        across /= np.linalg.norm(across)
        anchors.append(
            {
                "station_m": station,
                "left_xz_m": bl.tolist(),
                "right_xz_m": (br + across * movement).tolist(),
                "baseline_left_xz_m": bl.tolist(),
                "baseline_right_xz_m": br.tolist(),
                "right_inward_m": movement,
                "interpretation": "Continuity constraint, unchanged endpoint"
                if movement == 0
                else "Provisional inward edge hypothesis; not an automatically detected boundary",
            }
        )
    return {
        "schema_version": 1,
        "status": "Experimental candidate only; not adopted game geometry",
        "baseline_track_sha256": baseline_hash,
        "baseline_length_m": track["length_m"],
        "coordinates": track["coordinates"],
        "origin": track["origin"],
        "coverage": {"start_station_m": ANCHORS[0][0], "end_station_m": ANCHORS[-1][0]},
        "anchors": anchors,
        "sources": sources,
        "builder_sha256": digest(Path(__file__)),
        "limitations": [
            "2022 imagery at 0.6 metre pixels predates repaving and is not a survey",
            "1.4 metre maximum inward movement is a hypothesis from source profiles and curb position",
            "Left targets retain the baseline, but recalculated road normals can move rendered edges",
            "Tapers join the unchanged baseline; extended profiles show unresolved alignment beyond this candidate",
            "No curb width, height, current legal limit or friction is established",
            "Do not install without rendered edge residual, terrain/contact and full lap validation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=ROOT / "godot/data/track.json")
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a candidate")
    track = json.loads(args.track.read_text())
    baseline_hash = digest(args.track)
    aerial = ROOT / "artifacts/reference/ortho/east-2022.png"
    aerial_hash = digest(aerial)
    sources = [{"path": str(aerial.relative_to(ROOT)), "sha256": aerial_hash}]
    sampled_stations = set()
    for path in args.profiles:
        report = json.loads(path.read_text())
        if (
            report.get("track_sha256") != baseline_hash
            or report.get("aerial_sha256") != aerial_hash
        ):
            raise ValueError("Profile evidence does not match the baseline and aerial")
        if not report.get("cross_sections", {}).get("sections"):
            raise ValueError("Profile evidence has no sampled cross sections")
        sampled_stations.update(
            section["station_m"] for section in report["cross_sections"]["sections"]
        )
        sources.append({"path": str(path), "sha256": digest(path)})
    if not REQUIRED_STATIONS.issubset(sampled_stations):
        raise ValueError(
            "Profile evidence lacks required exit and extended section stations"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(build(track, baseline_hash, sources), stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline_sha256": baseline_hash,
                "status": "Experimental, not installed",
            }
        )
    )


if __name__ == "__main__":
    main()
