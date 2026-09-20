# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "matplotlib==3.10.8", "pyproj==3.7.2"]
# ///
"""Record the reviewed historical main ribbon, separating pit merge apron.

Manual join anchors are interpretation of the registered source panels produced
by review_pit_joins.py. This is a provisional reconstruction, not a survey.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from build_pit_envelope import OUT, ROOT, digest, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-track", type=Path, default=OUT / "pit-baseline-track.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/reference/pit-envelope.json"
    )
    args = parser.parse_args()
    track_path = args.baseline_track
    candidate_path = OUT / "pit-envelope-candidate.json"
    road_path = OUT / "road-surface.json"
    track = json.loads(track_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    road = json.loads(road_path.read_text())
    baseline_hash = digest(track_path)
    if candidate["sources"]["track"]["sha256"] != baseline_hash:
        raise ValueError("Baseline track differs from candidate evidence")
    if road["metadata"]["track_sha256"] != baseline_hash:
        raise ValueError("Road frame differs from baseline track")
    if digest(road_path) != candidate["sources"]["road"]["sha256"]:
        raise ValueError("Road frame differs from candidate evidence")
    previous_path = ROOT / "data/reference/pit-envelope.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous["baseline_track_sha256"] != baseline_hash:
            raise ValueError("Baseline track differs from adopted manifest")
    samples = track["samples"]
    stations = np.array([s["s"] for s in samples])
    positions = np.array([s["p"] for s in samples])[:, [0, 2]]
    tangent = np.roll(positions, -1, axis=0) - np.roll(positions, 1, axis=0)
    left = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    left /= np.linalg.norm(left, axis=1)[:, None]
    half = np.array([s["width"] / 2 for s in samples])

    def baseline(station):
        i = np.searchsorted(stations, station, side="right") - 1
        j = (i + 1) % len(samples)
        span = (stations[j] - stations[i]) % track["length_m"]
        t = (station - stations[i]) / span

        def at(points):
            return points[i] * (1 - t) + points[j] * t

        return (
            at(positions),
            at(positions + left * half[:, None]),
            at(positions - left * half[:, None]),
        )

    anchors = []

    def add(s, l, r, reason, raw=None, uncertainty=1.2):
        c, _, normal = frame(road, s)
        bc, bl, br = baseline(s)
        lx, rx = (bl, br) if l is None else (c + l * normal, c + r * normal)
        item = {
            "station_m": s,
            "baseline_center_xz_m": bc.tolist(),
            "baseline_left_xz_m": bl.tolist(),
            "baseline_right_xz_m": br.tolist(),
            "left_xz_m": lx.tolist(),
            "right_xz_m": rx.tolist(),
            "uncertainty_m": uncertainty,
            "interpretation": reason,
        }
        if l is not None:
            item.update({"frame_left_offset_m": l, "frame_right_offset_m": r})
        if raw is not None:
            item["raw_raised_face_candidate_m"] = raw
        anchors.append(item)

    # Painted separation follows the inside of paved pit approach, not outer apron.
    manual = [
        (4500, None, None),
        (4520, 6.5, -4.5),
        (4540, 7.5, -3),
        (4560, 7.8, -3),
        (4580, 8, -3),
        (4600, 8, -3),
        (0, 8, -3),
        (20, 8, -3),
        (35, 8.3, -3.5),
        (45, 8.5, -3.86),
    ]
    for s, l, r in manual:
        add(
            s,
            l,
            r,
            "Unchanged baseline join"
            if l is None
            else "Manual historical paint and pavement interpretation from registered final turn and wall approach panels; western apron excluded",
        )
    for a in candidate["candidate_anchors"]:
        add(
            a["station_m"],
            a["left_offset_m"],
            a["right_offset_m"],
            "East optical pavement boundary; west raised face candidate used as provisional wall base without invented clearance strip",
            raw=a["right_offset_m"],
        )
    for s, l, r in [
        (420, 8.4, -2.44),
        (425, 8.4, -2.67),
        (435, 8.2, -3),
        (450, 8.2, -3),
        (480, 8.2, -3),
        (510, 8, -3),
        (550, 7.8, -3),
        (590, 7.8, -3),
        (620, 7.8, -3),
        (640, 7.5, -3.5),
        (660, 7, -4.5),
        (680, 6.5, -5.5),
        (700, None, None),
    ]:
        add(
            s,
            l,
            r,
            "Unchanged baseline join"
            if l is None
            else "Historical main ribbon and pit merge paint interpretation; taper into unaffected corner pavement, excluding western apron",
        )
    anchors.sort(key=lambda a: a["station_m"])
    apron = []
    for start, end, outer in [(4540, 4600, -8), (0, 35, -8), (435, 620, -8)]:
        rows = []
        for a in anchors:
            if start <= a["station_m"] <= end:
                c, _, n = frame(road, a["station_m"])
                rows.append(
                    {
                        "station_m": a["station_m"],
                        "inner_xz_m": a["right_xz_m"],
                        "outer_xz_m": (c + outer * n).tolist(),
                    }
                )
        apron.append(
            {
                "start_station_m": start,
                "end_station_m": end,
                "status": "Separate provisional paved apron extent; excluded from racing ribbon; not ready for collision or legal limits",
                "uncertainty_m": 2.0,
                "sections": rows,
            }
        )
    registration = candidate["endpoint_registration"]
    report = {
        "schema_version": 1,
        "status": "Adopted provisional historical reconstruction for game development, not surveyed or current legal track limits",
        "baseline_track_sha256": digest(track_path),
        "baseline_length_m": track["length_m"],
        "coverage": {"start_station_m": 4500, "end_station_m": 700},
        "coordinates": track["coordinates"],
        "origin": track["origin"],
        "sources": candidate["sources"],
        "source_aerial": {
            k: registration[k]
            for k in ["image_sha256", "extent", "datum_operation", "datum_grid_hashes"]
        },
        "adoption_builder_sha256": digest(Path(__file__)),
        "interpolation": "Interpolate displacement from original baseline edges across the wrapped coverage; keep baseline unchanged elsewhere. Endpoints have exactly zero displacement.",
        "anchors": anchors,
        "separate_paved_aprons": apron,
        "limitations": [
            "2022 aerial and 2023 lidar predate repave.",
            "Minimum 1.2m interpretive uncertainty includes pixel ambiguity; not a statistical confidence interval.",
            "Raised returns identify a wall face above ground, not surveyed wall base. No arbitrary clearance offset is applied.",
            "Join paint is manually interpreted where pit apron and main ribbon are both paved.",
            "Curved frame conversion must be diagnosed by the application builder; explicit edge xz anchors are authoritative.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "anchors": len(anchors),
                "coverage": report["coverage"],
                "baseline_track_sha256": report["baseline_track_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
