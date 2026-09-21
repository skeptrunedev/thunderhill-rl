# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "Pillow==12.1.1", "pyproj==3.7.2"]
# ///
"""Overlay explicit reviewed/provisional curb placements on pinned historical NAIP.

This is a manual review aid, not automatic detection or a curb survey. The
polygons reproduce the placement manifest and track.gd's horizontal edge frame.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from terrain_datum import terrain_transform

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/curb-placement-audit")
    args = parser.parse_args()
    output = args.output
    if (output / "report.json").exists():
        raise FileExistsError(output / "report.json")
    layout_path = ROOT / "godot/data/curb-placement.json"
    layout = json.loads(layout_path.read_text())
    track_path = ROOT / "godot/data/track.json"
    metadata_path = ROOT / "data/reference/ortho-measurements.json"
    image_path = ROOT / "artifacts/reference/ortho/east-2022.png"
    track = json.loads(track_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if digest(image_path) != metadata["image_sha256"]:
        raise ValueError("Historical aerial image does not match reviewed source")
    transform, datum = terrain_transform()
    image = Image.open(image_path).convert("RGB")
    extent = metadata["extent"]
    spacing = np.array([(extent["xmax"] - extent["xmin"]) / image.width,
                        (extent["ymax"] - extent["ymin"]) / image.height])
    points = np.array([row["p"] for row in track["samples"]])
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    left = np.column_stack([tangent[:, 2], -tangent[:, 0]])
    left /= np.linalg.norm(left, axis=1)[:, None]
    widths = np.array([row["width"] for row in track["samples"]])
    if layout["schema_version"] != 1 or layout["track_sha256"] != digest(track_path) or layout["sample_count"] != len(points):
        raise ValueError("Curb placement source mismatch")
    width = float(layout["width_m"])
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Invalid curb width")
    origin = track["origin"]

    def pixel(local):
        east, north = transform.transform(origin["easting"] + local[:, 0],
                                          origin["northing"] - local[:, 1])
        return np.column_stack([(east - extent["xmin"]) / spacing[0],
                                (extent["ymax"] - north) / spacing[1]])

    runs = layout["runs"]
    seen = set()
    for run in runs:
        if run["side"] not in [-1, 1] or not run["segments"]:
            raise ValueError("Invalid curb run")
        for index in run["segments"]:
            if not isinstance(index, int) or not 0 <= index < len(points) or (index, run["side"]) in seen:
                raise ValueError("Invalid or overlapping curb segment")
            seen.add((index, run["side"]))
    output.mkdir(parents=True, exist_ok=True)
    overview = image.copy()
    overview_draw = ImageDraw.Draw(overview)
    records = []
    for run in runs:
        indices = run["segments"]
        side = run["side"]
        polygons = []
        for i in indices:
            j = (i + 1) % len(points)
            a = points[i, [0, 2]] + side * left[i] * widths[i] * .5
            b = a + side * left[i] * width
            d = points[j, [0, 2]] + side * left[j] * widths[j] * .5
            c = d + side * left[j] * width
            polygons.append(pixel(np.array([a, b, c, d])))
        all_pixels = np.concatenate(polygons)
        lo = np.maximum(np.floor(all_pixels.min(axis=0) - 25), 0).astype(int)
        hi = np.minimum(np.ceil(all_pixels.max(axis=0) + 25), image.size).astype(int)
        box = (*lo, *hi)
        plain = image.crop(box)
        marked = plain.copy()
        draw = ImageDraw.Draw(marked)
        for polygon in polygons:
            draw.polygon([tuple(p - lo) for p in polygon], outline="red", width=1)
            overview_draw.polygon([tuple(p) for p in polygon], outline="red", width=1)
        label = run["id"]
        center = all_pixels.mean(axis=0)
        overview_draw.text(tuple(center + [4, 4]), label, fill="yellow",
                           stroke_width=1, stroke_fill="black")
        start = float(track["samples"][indices[0]]["s"])
        end = float(track["samples"][(indices[-1] + 1) % len(points)]["s"])
        plain = plain.resize((plain.width * 3, plain.height * 3), Image.Resampling.NEAREST)
        marked = marked.resize(plain.size, Image.Resampling.NEAREST)
        sheet = Image.new("RGB", (plain.width * 2, plain.height + 35), "#19232b")
        sheet.paste(plain, (0, 35))
        sheet.paste(marked, (plain.width, 35))
        ImageDraw.Draw(sheet).text((8, 10),
            f"{label}: s={start:.1f}..{end:.1f}m side={side:+d} | NAIP / explicit footprint", fill="white")
        filename = f"{label}.png"
        sheet.save(output / filename)
        records.append({"id": label, "start_m": start, "end_m": end, "side": side,
                        "segments": indices, "image": filename,
                        "status": run["status"]})
    overview.save(output / "overview.png")
    report = {"track_sha256": digest(track_path),
              "track_script_sha256": digest(ROOT / "godot/scripts/track.gd"),
              "aerial_sha256": digest(image_path),
              "metadata_sha256": digest(metadata_path), "datum": datum,
              "pixel_spacing_m": spacing.tolist(), "curb_placement_sha256": digest(layout_path),
              "provisional_width_m": width, "runs": records,
              "limitations": "Historical 0.6m aerial; paint, curb and adjoining pavement may be ambiguous. No height inference."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"runs": len(records), "output": str(output)}))


if __name__ == "__main__":
    main()
