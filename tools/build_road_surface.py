# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1"]
# ///
"""Build an experimental differentiable road, without modifying shipped geometry.

Coefficients are descending powers of local station in meters. Station remains
an interpolation parameter, not exact arc length. Output is deliberately an
artifact until projection, mesh boundaries and unilateral contact are integrated.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline, PPoly

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("R", "Rs", "Ru", "Rss", "Rsu", "Ruu")


def build(track: dict, source_sha256: str = "") -> dict:
    rows = track["samples"]
    length = float(track["length_m"])
    stations = np.array([r["s"] for r in rows] + [length])
    xyz = np.array([r["p"] for r in rows] + [rows[0]["p"]])
    bank = np.tan([r["bank"] for r in rows] + [rows[0]["bank"]])
    splines = [
        make_interp_spline(stations, xyz[:, axis], k=5, bc_type="periodic")
        for axis in range(3)
    ]
    splines.append(make_interp_spline(stations, bank, k=3, bc_type="periodic"))
    breaks = sorted(
        {0.0, length}
        | {float(x) for s in splines for x in PPoly.from_spline(s).x if 0 < x < length}
    )
    # Each union interval contains one polynomial from every channel. Taylor
    # coefficients at its left endpoint reproduce that polynomial exactly.
    coefficients = [
        [
            [
                float(s(a, nu=n)) / math.factorial(n) if n <= s.k else 0.0
                for n in range(5, -1, -1)
            ]
            for s in splines
        ]
        for a in breaks[:-1]
    ]
    return {
        "schema_version": 1,
        "period_m": length,
        "breaks": breaks,
        "coefficients": coefficients,
        "road_lift_m": 0.04,
        "metadata": {
            "track_sha256": source_sha256,
            "centerline_degree": 5,
            "bank_slope_degree": 3,
            "coefficient_order": "descending powers of station minus interval start; channels x,y,z,tan(bank)",
            "status": "experimental, not used by shipped physics or mesh",
            "license": "ODbL 1.0 derived track data",
            "limitations": "Interpolates already smoothed historical observations. Differentiability is not surveyed curvature accuracy. Station is not exact arc length.",
        },
    }


def evaluate(data: dict, station: float, lateral: float) -> dict:
    s = station % data["period_m"]
    i = min(
        np.searchsorted(data["breaks"], s, side="right") - 1,
        len(data["coefficients"]) - 1,
    )
    x = s - data["breaks"][i]
    c = np.asarray(data["coefficients"][i])
    derivatives = [
        np.array([np.polyval(np.polyder(row, n), x) for row in c]) for n in range(4)
    ]
    center, d1, d2, d3 = [d[:3] for d in derivatives]
    v, a, j = d1[[0, 2]], d2[[0, 2]], d3[[0, 2]]
    q = np.linalg.norm(v)
    if not np.isfinite(q) or q <= 1e-9:
        raise ValueError("Degenerate planar tangent")
    qp = np.dot(v, a) / q
    qpp = (np.dot(a, a) + np.dot(v, j)) / q - np.dot(v, a) ** 2 / q**3
    rotate = lambda w: np.array([w[1], 0.0, -w[0]])
    left = rotate(v) / q
    left1 = rotate(a) / q - rotate(v) * qp / q**2
    left2 = (
        rotate(j) / q
        - 2 * rotate(a) * qp / q**2
        - rotate(v) * qpp / q**2
        + 2 * rotate(v) * qp**2 / q**3
    )
    across = left + np.array([0.0, derivatives[0][3], 0.0])
    across1 = left1 + np.array([0.0, derivatives[1][3], 0.0])
    across2 = left2 + np.array([0.0, derivatives[2][3], 0.0])
    rs = d1 + lateral * across1
    cross = np.cross(rs, across)
    result = dict(
        zip(
            FIELDS,
            [
                center + lateral * across + np.array([0.0, data["road_lift_m"], 0.0]),
                rs,
                across,
                d2 + lateral * across2,
                across1,
                np.zeros(3),
            ],
        )
    )
    result["normal"] = cross / np.linalg.norm(cross)
    return result


def audit(data: dict, track: dict) -> dict:
    rows = track["samples"]
    positions = np.array([r["p"] for r in rows])
    stations = np.array([r["s"] for r in rows] + [track["length_m"]])
    max_offset = 0.0
    min_jacobian = math.inf
    curvature = []
    normal_y = []
    extrema = {}
    center_curvature = []
    for i, row in enumerate(rows):
        for fraction in np.linspace(0, 1, 9, endpoint=False):
            s = stations[i] * (1 - fraction) + stations[i + 1] * fraction
            p = (
                positions[i] * (1 - fraction)
                + positions[(i + 1) % len(rows)] * fraction
            )
            center = evaluate(data, s, 0)["R"] - np.array(
                [0.0, data["road_lift_m"], 0.0]
            )
            max_offset = max(max_offset, float(np.linalg.norm(center - p)))
            width = (
                row["width"] * (1 - fraction)
                + rows[(i + 1) % len(rows)]["width"] * fraction
            )
            for lateral in np.linspace(-width / 2, width / 2, 5):
                r = evaluate(data, s, lateral)
                min_jacobian = min(min_jacobian, float(np.cross(r["Rs"], r["Ru"])[1]))
                normal_y.append(float(r["normal"][1]))
                value = float(np.dot(r["normal"], r["Rss"]) / np.dot(r["Rs"], r["Rs"]))
                curvature.append(value)
                if lateral == 0:
                    center_curvature.append(value)
                for label, better in [
                    (
                        "minimum",
                        value < extrema.get("minimum", {}).get("curvature", math.inf),
                    ),
                    (
                        "maximum",
                        value > extrema.get("maximum", {}).get("curvature", -math.inf),
                    ),
                ]:
                    if better:
                        extrema[label] = {
                            "station": float(s),
                            "lateral": float(lateral),
                            "curvature": value,
                            "bank_slope": float(r["Ru"][1]),
                            "Rs_norm": float(np.linalg.norm(r["Rs"])),
                        }
    for entry in extrema.values():
        s, u = entry["station"], entry["lateral"]
        r, center = evaluate(data, s, u), evaluate(data, s, 0.0)
        i = min(
            np.searchsorted(data["breaks"], s, side="right") - 1,
            len(data["coefficients"]) - 1,
        )
        bank_second = float(
            np.polyval(np.polyder(data["coefficients"][i][3], 2), s - data["breaks"][i])
        )
        denominator = float(r["Rs"] @ r["Rs"])
        center_part = float(r["normal"] @ center["Rss"]) / denominator
        bank_part = r["normal"][1] * u * bank_second / denominator
        v, a = center["Rs"][[0, 2]], center["Rss"][[0, 2]]
        entry.update(
            {
                "planar_center_curvature_per_m": float(
                    (v[0] * a[1] - v[1] * a[0]) / np.linalg.norm(v) ** 3
                ),
                "bank_slope_second_derivative": bank_second,
                "normal_curvature_components": {
                    "center_second_derivative": center_part,
                    "bank_second_derivative": float(bank_part),
                    "lateral_frame_second_derivative": entry["curvature"]
                    - center_part
                    - bank_part,
                },
            }
        )
    return {
        "sample_count": len(curvature),
        "max_center_displacement_from_old_chord_m": max_offset,
        "min_planar_jacobian": min_jacobian,
        "normal_y_range": [min(normal_y), max(normal_y)],
        "constant_lateral_normal_curvature_per_m_range": [
            min(curvature),
            max(curvature),
        ],
        "curvature_extrema": extrema,
        "centerline_normal_curvature_per_m_range": [
            min(center_curvature),
            max(center_curvature),
        ],
        "scope": "Nine stations per original interval, five lateral positions per station. Sampled bounds, not an exhaustive interval proof.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=ROOT / "godot/data/track.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/road-surface/road-surface.json"
    )
    args = parser.parse_args()
    raw = args.track.read_bytes()
    track = json.loads(raw)
    data = build(track, hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    report = audit(data, track)
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    cases = []
    parity_stations = [
        -0.1,
        0.0,
        1.25,
        100.5,
        1830.25,
        data["period_m"] - 0.01,
        data["period_m"] + 0.1,
    ]
    parity_stations += [row["station"] for row in report["curvature_extrema"].values()]
    parity_stations += [data["breaks"][500] + delta for delta in [-1e-7, 0.0, 1e-7]]
    for s in parity_stations:
        laterals = [-5.0, 0.0, 5.0] + [
            row["lateral"]
            for row in report["curvature_extrema"].values()
            if row["station"] == s
        ]
        for u in laterals:
            r = evaluate(data, s, u)
            velocity = 20 * r["Rs"] / np.linalg.norm(r["Rs"]) + 2 * r["Ru"]
            metric = np.array(
                [
                    [r["Rs"] @ r["Rs"], r["Rs"] @ r["Ru"]],
                    [r["Rs"] @ r["Ru"], r["Ru"] @ r["Ru"]],
                ]
            )
            rates = np.linalg.solve(metric, [r["Rs"] @ velocity, r["Ru"] @ velocity])
            acceleration = float(
                r["normal"]
                @ (
                    r["Rss"] * rates[0] ** 2
                    + 2 * r["Rsu"] * rates[0] * rates[1]
                    + r["Ruu"] * rates[1] ** 2
                )
            )
            cases.append(
                {
                    "station": s,
                    "lateral": u,
                    "expected": {k: v.tolist() for k, v in r.items()},
                    "contact_velocity": velocity.tolist(),
                    "normal_acceleration": acceleration,
                }
            )
    args.output.with_suffix(".parity.json").write_text(
        json.dumps({"cases": cases}) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
