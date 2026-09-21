# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1", "pyproj==3.7.2", "shapely==2.1.2"]
# ///
"""Extract restrained historical pavement tones, never friction or geometry."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from build_terrain_color import SOURCE_HASH, sha256, srgb_to_linear
from terrain_datum import terrain_transform

ROOT = Path(__file__).resolve().parents[1]


def pavement_samples(rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite RGB strip")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB outside unit range")
    encoded = rgb @ np.array([0.2126, 0.7152, 0.0722])
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    accepted = (encoded > 0.16) & (encoded < 0.58) & (chroma < 0.055)
    luminance = srgb_to_linear(rgb) @ np.array([0.2126, 0.7152, 0.0722])
    return luminance * accepted, accepted.astype(float)


def resample_pavement(rgb, row, col):
    numerator, weights = pavement_samples(rgb)
    return tuple(
        map_coordinates(field, [row, col], order=1, mode="nearest")
        for field in (numerator, weights)
    )


def tone_gains(numerator, weights):
    if numerator.shape != weights.shape or numerator.ndim != 2:
        raise ValueError("Expected equally sized numerator and validity fields")
    if not np.isfinite(numerator).all() or not np.isfinite(weights).all():
        raise ValueError("Nonfinite pavement data")
    accepted = weights > 0.8
    if not accepted.any():
        raise ValueError("No accepted pavement pixels")
    median = np.median(numerator[accepted] / weights[accepted])
    options = dict(sigma=(1.5, 1.0), mode=("wrap", "nearest"))
    weight = gaussian_filter(weights, **options)
    filtered = (gaussian_filter(numerator, **options) + 0.02 * median) / (weight + 0.02)
    return np.clip(filtered / median, 0.65, 1.35), accepted


def road_strip(track, rows=2048, columns=64):
    """Uniform station pixel centers and transverse centers match road UVs."""
    points = np.array([sample["p"] for sample in track["samples"]], dtype=float)[
        :, [0, 2]
    ]
    stations = np.array([sample["s"] for sample in track["samples"]])
    widths = np.array([sample["width"] for sample in track["samples"]])
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    left = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    s = (np.arange(rows) + 0.5) * track["length_m"] / rows
    i = np.searchsorted(stations, s, side="right") - 1
    j = (i + 1) % len(stations)
    end = np.where(j == 0, track["length_m"], stations[j])
    t = ((s - stations[i]) / (end - stations[i]))[:, None]
    # Match the render mesh's A B C / A C D diagonal, not a bilinear quad.
    # Its edge positions are rounded to float32 before subdivision.
    right = (points - left * widths[:, None] * 0.5).astype(np.float32).astype(float)
    left_edge = (points + left * widths[:, None] * 0.5).astype(np.float32).astype(float)
    a, b = right[i, None, :], left_edge[i, None, :]
    c, d = left_edge[j, None, :], right[j, None, :]
    across = (np.arange(columns) + 0.5) / columns
    u = ((1 - t) * widths[i, None] + t * widths[j, None]) * across[None, :]
    on_abc = u >= t * widths[j, None]
    wb = (u - t * widths[j, None]) / widths[i, None]
    abc = a * (1 - t - wb)[:, :, None] + b * wb[:, :, None] + c * t[:, :, None]
    wc = u / widths[j, None]
    acd = a * (1 - t)[:, :, None] + c * wc[:, :, None] + d * (t - wc)[:, :, None]
    return np.where(on_abc[:, :, None], abc, acd)


def main():
    source = ROOT / "artifacts/reference/ortho/east-2022.png"
    if sha256(source) != SOURCE_HASH:
        raise ValueError("Historical aerial hash changed")
    track_path = ROOT / "godot/data/track.json"
    track = json.loads(track_path.read_text())
    export_path = source.parent / "export.json"
    export = json.loads(export_path.read_text())
    extent = export["response"]["extent"]
    if extent["spatialReference"]["wkid"] != 26910:
        raise ValueError("Unexpected source datum")
    image = np.asarray(Image.open(source).convert("RGB"), dtype=float) / 255
    if image.shape[:2] != (export["response"]["height"], export["response"]["width"]):
        raise ValueError("Aerial dimensions disagree with export metadata")
    xz = road_strip(track)
    transform, datum = terrain_transform()
    east, north = transform.transform(
        xz[:, :, 0] + track["origin"]["easting"],
        track["origin"]["northing"] - xz[:, :, 1],
        errcheck=True,
    )
    col = (east - extent["xmin"]) / (extent["xmax"] - extent["xmin"]) * image.shape[
        1
    ] - 0.5
    row = (extent["ymax"] - north) / (extent["ymax"] - extent["ymin"]) * image.shape[
        0
    ] - 0.5
    if (
        col.min() < 0
        or row.min() < 0
        or col.max() >= image.shape[1] - 1
        or row.max() >= image.shape[0] - 1
    ):
        raise ValueError("Road strip leaves aerial source")
    numerator, weights = resample_pavement(image, row, col)
    gains, accepted = tone_gains(numerator, weights)
    target = ROOT / "godot/assets/materials/pavement_tone.png"
    Image.fromarray(np.rint(gains * 0.5 * 255).astype(np.uint8)).save(target)
    metadata = {
        "source_sha256": SOURCE_HASH,
        "source_export_sha256": sha256(export_path),
        "source_request": export["request"],
        "track_sha256": sha256(track_path),
        "generator_sha256": sha256(Path(__file__)),
        "horizontal_datum": datum,
        "output_sha256": sha256(target),
        "dimensions": [64, 2048],
        "accepted_pixel_fraction": float(accepted.mean()),
        "gain_range": [0.65, 1.35],
        "encoding": "Linear numerical data, red channel times two. U is lateral road fraction, V is station divided by lap length. Pixel centers sampled; station filtering wraps.",
        "license": "Public domain USDA NAIP; ODbL OpenStreetMap derived sampling coordinates",
        "limitations": [
            "Historical July 2022 appearance predates repave",
            "RGB rejection is not semantic segmentation; shadows may survive",
            "Appearance only, not measured reflectance, road geometry or friction",
            "Outer road strip should fade to neutral to avoid boundary contamination",
        ],
    }
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "accepted": metadata["accepted_pixel_fraction"],
                "gain_percentiles": np.percentile(gains, [0, 10, 50, 90, 100]).tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
