# /// script
# requires-python = ">=3.12"
# ///
"""Apexes hit on a recorded lap, on any circuit.

  uv run --script tools/apex_report.py --track balaton-park RECORDING.jsonl[.gz] [--json]
  uv run --script tools/apex_report.py --track T --corners 666,2109 --kinks 1820 RECORDING

A corner is apexed when, within --window-m of its apex station, the bike's
furthest excursion toward the inside reaches --apex-fraction of the half-width:
max(sign * lateral_m / half_width_m) >= 0.75, where sign is the corner's
direction (positive curvature turns left, and lateral_m is positive to the
left). This is the measure of the Thunderhill apex watcher the user signed off.

Which corners count:
  - --corners/--kinks: apex stations given by hand; each one's direction is the
    sign of the largest-|curvature| centreline sample within --window-m, sampled
    every 2 m. Kinks are reported, not required.
  - thunderhill-east without --corners: the signed-off list (SIGNED_OFF).
    Regression check: on the 121.13 s specialist lap
    (runs/sac/sac-v7-lr1e4-1, step_24000071) it reports 9/9.
  - any other circuit: found from the centreline. A corner is a run of samples
    curving the same way at more than --min-curvature, with gaps shorter than
    --merge-gap-m bridged; its heading change is the integral of curvature and
    it is required from --min-corner-deg (smaller kinks are reported only). Its
    apex is the curvature peak, except on a long constant-radius arc (over
    --long-arc-m with no peak above --clear-peak times the run's mean
    curvature), whose apex is where half its heading change is done.
The automatic corners are always reported too (auto_apexes_*), also where a
list decides.

The recording is a Godot episode recording (one JSON row per transition, as the
SAC evaluator and rev_drive keep them). Also reported, as the watcher did: the
median requested throttle while upright (|lean| < --upright-lean-rad) and off
the front brake, i.e. on the straights outside the braking zones, and the share
of all rows at full throttle (>= 0.99).
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
UPRIGHT_LEAN_RAD = 0.2
LONG_ARC_M = 150.0
CLEAR_PEAK = 1.5
SMOOTH_M = 30.0
# Bumped whenever the corners or the measure change, so cached reports are redone.
VERSION = 2
# Hand-picked corners (apex stations, m) of the signed-off Thunderhill watcher.
SIGNED_OFF = {DEFAULT_TRACK: dict(corners=[666, 2109, 2450, 2686, 3019, 3566, 3695, 4414, 4537],
                                  kinks=[1820, 3141, 3232])}


def smoothed_curvature(road: RoadTelemetry, widths, smooth_m=SMOOTH_M) -> list[float]:
    """Curvature averaged over smooth_m of centreline around each sample: OSM
    polylines alternate sharp and flat samples along one bend (Catalunya's
    turn 3 reads 0.001, 0.022, 0.004, 0.019 1/m every 6 m), which would split one
    corner into several and put its apex on a vertex."""
    samples, length, n = road.samples, road.length, len(road.samples)
    result = []
    for i in range(n):
        total = weight = 0.0
        for j in range(n):
            offset = (samples[j]["s"] - samples[i]["s"] + length / 2) % length - length / 2
            if abs(offset) <= smooth_m / 2:
                total += samples[j]["curvature"] * widths[j]
                weight += widths[j]
        result.append(total / weight)
    return result


def corners(road: RoadTelemetry, min_curvature=MIN_CURVATURE, merge_gap_m=MERGE_GAP_M,
            min_kink_deg=MIN_KINK_DEG, min_corner_deg=MIN_CORNER_DEG, long_arc_m=LONG_ARC_M,
            clear_peak=CLEAR_PEAK, smooth_m=SMOOTH_M) -> list[dict]:
    """Corners and kinks found from the centreline (curvature smoothed over
    smooth_m), in station order: apex station, direction (+1 left, -1 right),
    heading change, apex radius and whether it is required."""
    samples, length = road.samples, road.length
    n = len(samples)
    widths = [((samples[(i + 1) % n]["s"] if i + 1 < n else length) - samples[i]["s"])
              for i in range(n)]
    kappa = smoothed_curvature(road, widths, smooth_m)
    sign = [0 if abs(k) < min_curvature else (1 if k > 0 else -1) for k in kappa]
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
        turn = sum(kappa[i % n] * widths[i % n] for i in span)
        degrees = abs(math.degrees(turn))
        if degrees < min_kink_deg:
            continue
        apex = max(indices, key=lambda i: abs(kappa[i]))
        arc_m = sum(widths[i % n] for i in span)
        long_arc = arc_m > long_arc_m and abs(kappa[apex]) < clear_peak * abs(turn) / arc_m
        if long_arc:
            # No clear peak: the apex is where half the heading change is done.
            done = 0.0
            for i in span:
                done += kappa[i % n] * widths[i % n]
                if abs(done) >= abs(turn) / 2:
                    apex = i % n
                    break
        result.append(dict(apex_station_m=round(samples[apex]["s"], 1), direction=run["sign"],
                           turn_deg=round(degrees, 1),
                           apex_radius_m=round(1 / abs(kappa[apex]), 1),
                           long_arc=long_arc, required=degrees >= min_corner_deg))
    return sorted(result, key=lambda c: c["apex_station_m"])


def listed_corners(road: RoadTelemetry, corner_stations, kink_stations,
                   window_m=WINDOW_M) -> list[dict]:
    """Hand-picked apex stations; the inside is the sign of the largest-|curvature|
    sample within window_m, sampled every 2 m."""
    result = []
    for station, required in [(c, True) for c in corner_stations] + [(k, False) for k in kink_stations]:
        peak = max((road.samples[road._segment(station + d)[0]]["curvature"]
                    for d in range(-int(window_m), int(window_m) + 1, 2)), key=abs)
        result.append(dict(apex_station_m=float(station), direction=1 if peak > 0 else -1,
                           apex_radius_m=round(1 / max(abs(peak), 1e-9), 1), required=required))
    return sorted(result, key=lambda c: c["apex_station_m"])


def transitions(recording: Path) -> list[dict]:
    opener = gzip.open if recording.suffix == ".gz" else open
    with opener(recording, "rt") as stream:
        return [row for row in map(json.loads, stream) if row.get("type") == "transition"]


def judge(road: RoadTelemetry, points, corner_list, window_m, apex_fraction) -> list[dict]:
    judged = []
    for corner in corner_list:
        near = [(corner["direction"] * lateral / half_width)
                for station, lateral, half_width in points
                if abs((station - corner["apex_station_m"] + road.length / 2) % road.length
                       - road.length / 2) <= window_m]
        reach = max(near) if near else None
        judged.append(dict(corner, inside_reach=None if reach is None else round(reach, 3),
                           apexed=reach is not None and reach >= apex_fraction))
    return judged


def report(track: str, recording: Path, *, corner_stations=None, kink_stations=(),
           window_m=WINDOW_M, apex_fraction=APEX_FRACTION, upright_lean_rad=UPRIGHT_LEAN_RAD,
           **corner_args) -> dict:
    road = RoadTelemetry(track=track)
    rows = transitions(recording)
    points = [(float(r["track"]["progress"]) * road.length, float(r["track"]["lateral_m"]),
               float(r["track"]["half_width_m"])) for r in rows]
    automatic = judge(road, points, corners(road, **corner_args), window_m, apex_fraction)
    if corner_stations is None and track in SIGNED_OFF:
        corner_stations, kink_stations = SIGNED_OFF[track]["corners"], SIGNED_OFF[track]["kinks"]
    found = automatic if corner_stations is None else judge(
        road, points, listed_corners(road, corner_stations, kink_stations, window_m),
        window_m, apex_fraction)
    required, auto_required = ([c for c in cs if c["required"]] for cs in (found, automatic))
    throttle = [float(r["requested_controls"]["throttle"]) for r in rows]
    upright = [float(r["requested_controls"]["throttle"]) for r in rows
               if abs(float(r["state"]["lean"])) < upright_lean_rad
               and float(r["requested_controls"]["front_brake"]) == 0.0]
    return dict(version=VERSION, track=track, recording=str(recording), transitions=len(rows),
                corner_source="automatic" if corner_stations is None else "listed",
                apexes_hit=sum(c["apexed"] for c in required), apexes_total=len(required),
                missed_apex_stations_m=[c["apex_station_m"] for c in required if not c["apexed"]],
                auto_apexes_hit=sum(c["apexed"] for c in auto_required),
                auto_apexes_total=len(auto_required),
                upright_median_throttle=round(statistics.median(upright), 3) if upright else None,
                full_throttle_share=round(sum(t >= 0.99 for t in throttle) / len(throttle), 3)
                if throttle else None,
                corners=found, automatic_corners=automatic)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--window-m", type=float, default=WINDOW_M)
    parser.add_argument("--apex-fraction", type=float, default=APEX_FRACTION)
    parser.add_argument("--corners", help="comma list of required apex stations (m); default: "
                        "SIGNED_OFF for the track, else found from the centreline")
    parser.add_argument("--kinks", default="", help="with --corners: reported-only stations")
    parser.add_argument("--min-corner-deg", type=float, default=MIN_CORNER_DEG)
    parser.add_argument("--min-curvature", type=float, default=MIN_CURVATURE)
    parser.add_argument("--merge-gap-m", type=float, default=MERGE_GAP_M)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    stations = lambda text: [float(x) for x in text.split(",") if x.strip()]  # noqa: E731
    result = report(args.track, args.recording,
                    corner_stations=stations(args.corners) if args.corners else None,
                    kink_stations=stations(args.kinks), window_m=args.window_m,
                    apex_fraction=args.apex_fraction, min_corner_deg=args.min_corner_deg,
                    min_curvature=args.min_curvature, merge_gap_m=args.merge_gap_m)
    if args.json:
        print(json.dumps(result, indent=1))
        return
    print(f"{args.track}: {result['apexes_hit']}/{result['apexes_total']} apexes "
          f"({result['corner_source']} corners; automatic {result['auto_apexes_hit']}/"
          f"{result['auto_apexes_total']}), upright median throttle "
          f"{result['upright_median_throttle']}, full throttle {result['full_throttle_share']}")
    for title, corner_list in (("corners", result["corners"]),
                               ("automatic corners", result["automatic_corners"])):
        if corner_list is result["automatic_corners"] and result["corner_source"] == "automatic":
            break
        print(f" {title}:")
        for c in corner_list:
            side = "L" if c["direction"] > 0 else "R"
            status = ("APEX" if c["apexed"] else "miss") if c["required"] else "kink"
            turn = f"{c['turn_deg']:6.1f} deg" if "turn_deg" in c else "          "
            arc = "  long arc" if c.get("long_arc") else ""
            print(f"  {c['apex_station_m']:7.1f} m  {side} {turn}  r {c['apex_radius_m']:6.1f} m"
                  f"  inside reach {c['inside_reach']}  {status}{arc}")


if __name__ == "__main__":
    main()
