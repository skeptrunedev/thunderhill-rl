# /// script
# requires-python = ">=3.12"
# ///
"""Apexes hit on a recorded lap, on any circuit.

  uv run --script tools/apex_report.py --track balaton-park RECORDING.jsonl[.gz] [--json]

Corners come from the centreline (RoadTelemetry samples): a corner is a run of
samples curving the same way at more than --min-curvature, with gaps shorter
than --merge-gap-m bridged; its heading change is the integral of curvature,
its apex the curvature peak, its inside the side the curvature points to
(positive curvature turns left, and lateral_m is positive to the left).

A corner is apexed when, within --window-m of the apex station, the bike's
furthest excursion toward the inside edge reaches --apex-fraction of the
half-width (lateral_m * sign(curvature) / half_width_m). Corners turning at
least --min-corner-deg are required; smaller kinks are reported only.

The recording is a Godot episode recording (one JSON row per transition, as the
SAC evaluator and rev_drive keep them). Also reported: the median requested
throttle while upright (|lean| < --upright-lean-rad), i.e. on the straights.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from lap_policy import DEFAULT_TRACK, RoadTelemetry  # noqa: E402

MIN_CURVATURE = 1 / 250  # 1/m: gentler bends are straights here
MERGE_GAP_M = 10.0
MIN_KINK_DEG = 10.0
MIN_CORNER_DEG = 30.0
WINDOW_M = 40.0
APEX_FRACTION = 0.75
UPRIGHT_LEAN_RAD = 0.1


def corners(road: RoadTelemetry, min_curvature=MIN_CURVATURE, merge_gap_m=MERGE_GAP_M,
            min_kink_deg=MIN_KINK_DEG) -> list[dict]:
    """Corners and kinks of the circuit in station order: apex station, direction
    (+1 left, -1 right), heading change and apex radius."""
    samples, length = road.samples, road.length
    n = len(samples)
    widths = [((samples[(i + 1) % n]["s"] if i + 1 < n else length) - samples[i]["s"])
              for i in range(n)]
    sign = [0 if abs(row["curvature"]) < min_curvature else (1 if row["curvature"] > 0 else -1)
            for row in samples]
    # Start the scan on a straight so no corner straddles the wrap.
    first = next((i for i in range(n) if sign[i] == 0), 0)
    order = [(first + k) % n for k in range(n)]
    runs, current, gap = [], None, 0.0
    for i in order:
        if sign[i] and current and sign[i] == current["sign"] and gap < merge_gap_m:
            current["samples"].append(i)
            gap = 0.0
        elif sign[i]:
            current = dict(sign=sign[i], samples=[i])
            runs.append(current)
            gap = 0.0
        else:
            gap += widths[i]
            if current and gap >= merge_gap_m:
                current = None
    result = []
    for run in runs:
        indices = run["samples"]
        span = range(indices[0], indices[0] + ((indices[-1] - indices[0]) % n) + 1)
        turn = sum(samples[i % n]["curvature"] * widths[i % n] for i in span)
        degrees = abs(math.degrees(turn))
        if degrees < min_kink_deg:
            continue
        apex = max(indices, key=lambda i: abs(samples[i]["curvature"]))
        result.append(dict(apex_station_m=round(samples[apex]["s"], 1), direction=run["sign"],
                           turn_deg=round(degrees, 1),
                           apex_radius_m=round(1 / abs(samples[apex]["curvature"]), 1)))
    return sorted(result, key=lambda c: c["apex_station_m"])


def transitions(recording: Path) -> list[dict]:
    opener = gzip.open if recording.suffix == ".gz" else open
    with opener(recording, "rt") as stream:
        return [row for row in map(json.loads, stream) if row.get("type") == "transition"]


def report(track: str, recording: Path, *, window_m=WINDOW_M, apex_fraction=APEX_FRACTION,
           min_corner_deg=MIN_CORNER_DEG, upright_lean_rad=UPRIGHT_LEAN_RAD, **corner_args) -> dict:
    road = RoadTelemetry(track=track)
    rows = transitions(recording)
    points = [(float(r["track"]["progress"]) * road.length, float(r["track"]["lateral_m"]),
               float(r["track"]["half_width_m"])) for r in rows]
    found = []
    for corner in corners(road, **corner_args):
        near = [(corner["direction"] * lateral / half_width)
                for station, lateral, half_width in points
                if abs((station - corner["apex_station_m"] + road.length / 2) % road.length
                       - road.length / 2) <= window_m]
        reach = max(near) if near else None
        found.append(dict(corner, required=corner["turn_deg"] >= min_corner_deg,
                          inside_reach=None if reach is None else round(reach, 3),
                          apexed=reach is not None and reach >= apex_fraction))
    required = [c for c in found if c["required"]]
    upright = [float(r["requested_controls"]["throttle"]) for r in rows
               if abs(float(r["state"]["lean"])) < upright_lean_rad]
    return dict(track=track, recording=str(recording), transitions=len(rows),
                apexes_hit=sum(c["apexed"] for c in required), apexes_total=len(required),
                missed_apex_stations_m=[c["apex_station_m"] for c in required if not c["apexed"]],
                upright_median_throttle=round(statistics.median(upright), 3) if upright else None,
                corners=found)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--window-m", type=float, default=WINDOW_M)
    parser.add_argument("--apex-fraction", type=float, default=APEX_FRACTION)
    parser.add_argument("--min-corner-deg", type=float, default=MIN_CORNER_DEG)
    parser.add_argument("--min-curvature", type=float, default=MIN_CURVATURE)
    parser.add_argument("--merge-gap-m", type=float, default=MERGE_GAP_M)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = report(args.track, args.recording, window_m=args.window_m,
                    apex_fraction=args.apex_fraction, min_corner_deg=args.min_corner_deg,
                    min_curvature=args.min_curvature, merge_gap_m=args.merge_gap_m)
    if args.json:
        print(json.dumps(result, indent=1))
        return
    print(f"{args.track}: {result['apexes_hit']}/{result['apexes_total']} apexes, upright median "
          f"throttle {result['upright_median_throttle']}")
    for c in result["corners"]:
        side = "L" if c["direction"] > 0 else "R"
        status = ("APEX" if c["apexed"] else "miss") if c["required"] else "kink"
        print(f"  {c['apex_station_m']:7.1f} m  {side} {c['turn_deg']:6.1f} deg  r {c['apex_radius_m']:6.1f} m"
              f"  inside reach {c['inside_reach']}  {status}")


if __name__ == "__main__":
    main()
