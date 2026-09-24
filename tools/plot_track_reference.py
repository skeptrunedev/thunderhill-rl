# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot the existing simulator progress reference, without creating a racing line."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot(track_path, output):
    data = json.loads(track_path.read_text())
    samples = data["samples"]
    points = np.array([s["p"] for s in samples])
    left = np.array([s["left"] for s in samples])
    width = np.array([s["width"] for s in samples])

    # Track JSON is east, up, south. Plot east/right and north/up.
    def xy(values):
        return values[:, [0, 2]] * [1, -1]

    center = xy(points)
    edge_a = xy(points + left * width[:, None] / 2)
    edge_b = xy(points - left * width[:, None] / 2)
    polygons = [
        [
            edge_a[i],
            edge_a[(i + 1) % len(points)],
            edge_b[(i + 1) % len(points)],
            edge_b[i],
        ]
        for i in range(len(points))
    ]
    fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
    ax.add_collection(PolyCollection(polygons, facecolor="#a5adb4", edgecolor="none"))
    closed = np.vstack([center, center[0]])
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color="#007cba",
        linewidth=1.6,
        label="Existing centerline used to measure progress",
    )
    stations = np.array([s["s"] for s in samples])
    for fraction in [0, 0.25, 0.5, 0.75]:
        i = int(np.argmin(abs(stations - fraction * data["length_m"])))
        point = center[i]
        following = center[(i + 15) % len(center)]
        ax.annotate(
            "",
            xy=following,
            xytext=point,
            arrowprops=dict(arrowstyle="->", color="#007cba", lw=2),
        )
        ax.scatter(*point, s=24, color="#153c50", zorder=5)
        ax.annotate(
            f"{fraction:.0%}" + (" progress origin" if fraction == 0 else ""),
            xy=point,
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
        )
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.set_xlabel("East from simulator origin (meters)")
    ax.set_ylabel("North from simulator origin (meters)")
    ax.grid(alpha=0.14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_title("Thunderhill East: current progress reference", fontsize=18, pad=34)
    ax.text(
        0.5,
        1.025,
        "Existing geometry only. No optimized racing line has been selected.",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.supxlabel(
        f"Simulator lap length: {data['length_m']:,.1f} m. Gray shows modeled track width.\n"
        "Reward measures legal forward progress / lap length, not closeness to the blue line.",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=ROOT / "godot/data/track.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.track, args.output)
