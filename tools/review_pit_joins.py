# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "matplotlib==3.10.8", "pyproj==3.7.2"]
# ///
"""Source registered inspection panels for adopting historical pit joins."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build_pit_envelope import OUT, ROOT, digest, frame
from pyproj import Transformer, datadir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-track", type=Path, default=OUT / "pit-baseline-track.json"
    )
    args = parser.parse_args()
    baseline_hash = digest(args.baseline_track)
    candidate = json.loads((OUT / "pit-envelope-candidate.json").read_text())
    registration = candidate["endpoint_registration"]
    image_path = ROOT / "artifacts/reference/ortho/east-2022.png"
    assert digest(image_path) == registration["image_sha256"]
    for name, expected in registration["datum_grid_hashes"].items():
        if digest(ROOT / "artifacts/reference/datum" / name) != expected:
            raise ValueError("Datum grid hash mismatch")
    datadir.append_data_dir(str(ROOT / "artifacts/reference/datum"))
    transform = Transformer.from_pipeline(registration["datum_operation"])
    if candidate["sources"]["track"]["sha256"] != baseline_hash:
        raise ValueError("Baseline track differs from candidate evidence")
    track = json.loads(args.baseline_track.read_text())
    road = json.loads((OUT / "road-surface.json").read_text())
    if road["metadata"]["track_sha256"] != baseline_hash:
        raise ValueError("Road frame differs from baseline track")
    if digest(OUT / "road-surface.json") != candidate["sources"]["road"]["sha256"]:
        raise ValueError("Road frame differs from candidate evidence")
    origin = track["origin"]
    extent = registration["extent"]
    image = plt.imread(image_path)

    def project(points):
        a = np.asarray(points)
        x, y = transform.transform(
            a[:, 0] + origin["easting"], origin["northing"] - a[:, 1]
        )
        return np.column_stack([x, y])

    manifest_path = ROOT / "data/reference/pit-envelope.json"
    adopted = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if adopted and adopted["baseline_track_sha256"] != baseline_hash:
        raise ValueError("Baseline track differs from adopted manifest")
    for start, end in [
        (4450, 4606),
        (0, 100),
        (100, 250),
        (250, 430),
        (430, 550),
        (550, 700),
    ]:
        stations = np.arange(start, end + 1)
        centers = project([frame(road, s)[0] for s in stations])
        low = centers.min(axis=0) - 25
        high = centers.max(axis=0) + 25
        dx = (extent["xmax"] - extent["xmin"]) / image.shape[1]
        dy = (extent["ymax"] - extent["ymin"]) / image.shape[0]
        c0 = max(0, int((low[0] - extent["xmin"]) / dx))
        c1 = min(image.shape[1], int((high[0] - extent["xmin"]) / dx) + 1)
        r0 = max(0, int((extent["ymax"] - high[1]) / dy))
        r1 = min(image.shape[0], int((extent["ymax"] - low[1]) / dy) + 1)
        mid = (low + high) / 2
        ex = [
            extent["xmin"] + c0 * dx - mid[0],
            extent["xmin"] + c1 * dx - mid[0],
            extent["ymax"] - r1 * dy - mid[1],
            extent["ymax"] - r0 * dy - mid[1],
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 14), layout="constrained")
        for ax in axes:
            ax.imshow(image[r0:r1, c0:c1], extent=ex, interpolation="nearest")
        for s in range(start, end + 1, 10):
            c, _, l = frame(road, s)
            xy = project([c])[0] - mid
            axes[1].annotate(str(s), xy, color="yellow", fontsize=8)
            for u, color in [(-6, "red"), (-3, "cyan"), (6, "red"), (8, "cyan")]:
                xy = project([c + u * l])[0] - mid
                axes[1].plot(*xy, ".", color=color, markersize=3)
        if adopted:
            for key, color in [("left_xz_m", "lime"), ("right_xz_m", "magenta")]:
                rows = [
                    a
                    for a in adopted["anchors"]
                    if start - 30 <= a["station_m"] <= end + 30
                ]
                if rows:
                    xy = project([a[key] for a in rows]) - mid
                    axes[1].plot(xy[:, 0], xy[:, 1], ".-", color=color)
        for ax in axes:
            ax.set_xlim(ex[0], ex[1])
            ax.set_ylim(ex[2], ex[3])
        axes[0].set_title("2022 source only")
        axes[1].set_title("Station frame: red ±6 m, cyan −3/+8 m; adopted lime/magenta")
        fig.savefig(OUT / f"pit-join-review-{start}-{end}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
