# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Migrate authored placements across a plan correction with stable sample identities.

Default execution reports the proposed migration without writing files. Historical
curb evidence remains unchanged; appended provenance describes the current layout.
"""

import argparse
import bisect
import copy
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURBS = "godot/data/curb-placement.json"
CURB_REVIEW = "data/reference/turn2-curb-review.json"
FIELDS = "godot/data/field-coverage.json"
FIELD_REVIEW = "data/reference/turn2-field-band-study.json"
SHADER = "godot/shaders/asphalt.gdshader"
JOINT = re.compile(
    r"(uniform vec2 paving_joint_station_m = vec2\()(\d+(?:\.\d+)?),\s*"
    r"(\d+(?:\.\d+)?)(\);)"
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def validate_tracks(old, new):
    if old["origin"] != new["origin"] or old["coordinates"] != new["coordinates"]:
        raise ValueError("Track coordinate frame changed")
    a, b = old["samples"], new["samples"]
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Track sample count changed")
    if [x["source_s"] for x in a] != [x["source_s"] for x in b]:
        raise ValueError("Track sample identities changed")
    for track in (old, new):
        rows = track["samples"]
        stations = [r["s"] for r in rows] + [track["length_m"]]
        if stations[0] != 0 or any(
            not math.isfinite(v) or v <= u for u, v in itertools.pairwise(stations)
        ):
            raise ValueError("Invalid physical station sequence")
        for r in rows:
            if (
                len(r["p"]) != 3
                or len(r["left"]) != 3
                or not all(
                    math.isfinite(v)
                    for v in r["p"] + r["left"] + [r["width"], r["source_s"]]
                )
                or r["width"] <= 0
            ):
                raise ValueError("Invalid placement geometry")


def map_station(station, old, new):
    """Preserve interval identity and fractional position, including lap endpoint."""
    if not math.isfinite(station) or not 0 <= station <= old["length_m"]:
        raise ValueError("Placement station outside baseline lap")
    a = [r["s"] for r in old["samples"]] + [old["length_m"]]
    b = [r["s"] for r in new["samples"]] + [new["length_m"]]
    i = min(bisect.bisect_right(a, station) - 1, len(a) - 2)
    return b[i] + (b[i + 1] - b[i]) * (station - a[i]) / (a[i + 1] - a[i])


def field_point(sample, offset):
    return [
        round(sample["p"][k] + sample["left"][k] * (sample["width"] / 2 + offset), 6)
        for k in (0, 2)
    ]


def prepare(root, baseline_path, track_path):
    old_raw, new_raw = baseline_path.read_bytes(), track_path.read_bytes()
    old_hash, new_hash = digest(old_raw), digest(new_raw)
    if old_hash == new_hash:
        raise ValueError("Tracks are identical; no migration required")
    old, new = json.loads(old_raw), json.loads(new_raw)
    validate_tracks(old, new)
    paths = (CURBS, CURB_REVIEW, FIELDS, FIELD_REVIEW, SHADER)
    originals = {p: (root / p).read_bytes() for p in paths}
    curbs, curb_review, fields, field_review = [
        json.loads(originals[p]) for p in paths[:4]
    ]
    if any(d["track_sha256"] != old_hash for d in (curbs, fields, field_review)):
        raise ValueError("Authored placement track provenance differs from baseline")
    if curbs["sample_count"] != len(old["samples"]):
        raise ValueError("Curb sample count differs from baseline")
    if curbs["source_reference"] != CURB_REVIEW or curbs[
        "source_reference_sha256"
    ] != digest(originals[CURB_REVIEW]):
        raise ValueError("Curb review provenance mismatch")
    for run in curbs["runs"]:
        if any(
            type(i) is not int or not 0 <= i < len(old["samples"])
            for i in run["segments"]
        ):
            raise ValueError("Invalid curb sample identity")
    references = [r for r in fields["source_references"] if r["path"] == FIELD_REVIEW]
    if len(references) != 1 or references[0]["sha256"] != digest(
        originals[FIELD_REVIEW]
    ):
        raise ValueError("Field review provenance mismatch")
    matching = [
        i for i, c in enumerate(fields["corridors"]) if c == field_review["corridor"]
    ]
    if len(matching) != 1:
        raise ValueError("Cannot identify the reviewed Turn 2 corridor uniquely")
    corridor_index = matching[0]
    construction = field_review["construction"]
    stations = construction["requested_stations_m"]
    indices = construction.get("selected_sample_indices")
    if indices is None:
        indices = [
            min(
                range(len(old["samples"])),
                key=lambda i: abs(old["samples"][i]["s"] - s),
            )
            for s in stations
        ]
    if len(indices) != len(stations) or len(indices) != len(
        field_review["corridor"]["points_local_xz"]
    ):
        raise ValueError("Field selection cardinality mismatch")
    if any(type(i) is not int or not 0 <= i < len(old["samples"]) for i in indices):
        raise ValueError("Invalid field sample identity")
    offset = construction["offset_inside_left_pavement_edge_m"]
    if not math.isfinite(offset):
        raise ValueError("Invalid field offset")
    previous_points = [field_point(old["samples"][i], offset) for i in indices]
    if any(
        abs(x - y) > 0.000002
        for p, q in zip(previous_points, field_review["corridor"]["points_local_xz"])
        for x, y in zip(p, q)
    ):
        raise ValueError(
            "Selected baseline samples do not reproduce authored field points"
        )
    shader = originals[SHADER].decode()
    matches = list(JOINT.finditer(shader))
    if len(matches) != 1:
        raise ValueError("Cannot locate a unique numeric paving joint declaration")
    joint = matches[0]
    joint_old = [float(joint[2]), float(joint[3])]
    joint_new = [round(map_station(s, old, new), 6) for s in joint_old]
    new_stations = [round(map_station(s, old, new), 6) for s in stations]
    migration = {
        "baseline_track_sha256": old_hash,
        "track_sha256": new_hash,
        "baseline_length_m": old["length_m"],
        "length_m": new["length_m"],
        "sample_identity": "Unchanged count, ordering and source_s; physical stations interpolated through corresponding sample intervals",
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "source_files_sha256": {p: digest(raw) for p, raw in originals.items()},
    }
    # Append current applicability without relabeling any original curb evidence.
    curb_review.setdefault("placement_migrations", []).append(copy.deepcopy(migration))
    curbs["track_sha256"] = new_hash
    curbs["source_reference_sha256"] = digest(encoded(curb_review))
    curbs.setdefault("placement_migrations", []).append(copy.deepcopy(migration))
    field_review.setdefault(
        "original_placement",
        {
            "track_sha256": old_hash,
            "construction": copy.deepcopy(construction),
            "corridor": copy.deepcopy(field_review["corridor"]),
        },
    )
    field_review.setdefault("placement_migrations", []).append(
        {
            **copy.deepcopy(migration),
            "previous_requested_stations_m": stations,
            "mapped_requested_stations_m": new_stations,
            "paving_joint_previous_station_m": joint_old,
            "paving_joint_mapped_station_m": joint_new,
        }
    )
    construction["selected_sample_indices"] = indices
    construction["requested_stations_m"] = new_stations
    construction["method"] = (
        "Preserved selected sample indices; p.xz + left.xz * (width / 2 + offset), piecewise linear centerline"
    )
    field_review["corridor"]["points_local_xz"] = [
        field_point(new["samples"][i], offset) for i in indices
    ]
    field_review["track_sha256"] = new_hash
    fields["corridors"][corridor_index] = copy.deepcopy(field_review["corridor"])
    fields["track_sha256"] = new_hash
    references[0]["sha256"] = digest(encoded(field_review))
    replacement = joint[1] + ", ".join(f"{s:.6f}" for s in joint_new) + joint[4]
    shader = shader[: joint.start()] + replacement + shader[joint.end() :]
    outputs = {
        p: encoded(d)
        for p, d in zip(paths[:4], (curbs, curb_review, fields, field_review))
    }
    outputs[SHADER] = shader.encode()
    report = {
        **migration,
        "files": list(outputs),
        "selected_field_sample_indices": indices,
        "paving_joint_previous_station_m": joint_old,
        "paving_joint_mapped_station_m": joint_new,
        "maximum_field_point_displacement_m": max(
            math.dist(a, b)
            for a, b in zip(
                previous_points, field_review["corridor"]["points_local_xz"]
            )
        ),
    }
    return originals, outputs, report


def apply(root, originals, outputs):
    # Validate every input before the first write. Never accept concurrent edits.
    if any((root / p).read_bytes() != raw for p, raw in originals.items()):
        raise ValueError("Placement files changed after migration preparation")
    for path, raw in outputs.items():
        (root / path).write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-track", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    originals, outputs, report = prepare(ROOT, args.baseline_track, args.track)
    if args.apply:
        if (ROOT / "godot/data/track.json").read_bytes() != args.track.read_bytes():
            raise ValueError(
                "Install the candidate runtime track before applying its placements"
            )
        apply(ROOT, originals, outputs)
    print(json.dumps({**report, "applied": args.apply}, indent=2))


if __name__ == "__main__":
    main()
