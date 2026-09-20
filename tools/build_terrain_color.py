# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1"]
# ///
"""Build restrained terrain color gains from historical public domain NAIP.

Run: uv run tools/build_terrain_color.py
This is an artistic color variation layer, not measured surface reflectance.
The RGB classifier is deliberately conservative but is not semantic segmentation.
Rejected pixels never contribute their color to the normalized convolution.
Regions without nearby accepted pixels converge to the accepted terrain median.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = "e49486d2d4f9c40b8d83a00e9ff02dd86407e5f09454f44cc835e1cd9f96573a"
SETTINGS = {
    "encoded_luminance_range": [0.24, 0.76],
    "red_minus_blue_min": 0.035,
    "red_minus_green_range": [0.006, 0.12],
    "green_minus_blue_min": 0.018,
    "smoothing_sigma_m": 6.0,
    "gaussian_truncate_sigma": 4.0,
    "median_prior_weight": 0.02,
    "target_output_pixel_m": 4.0,
    "gain_range": [0.6, 1.4],
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def dry_terrain_mask(rgb):
    """RGB is encoded sRGB in [0, 1]; thresholds are artistic selection rules."""
    red, green, blue = np.moveaxis(rgb, -1, 0)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    low, high = SETTINGS["encoded_luminance_range"]
    rg_low, rg_high = SETTINGS["red_minus_green_range"]
    return ((luminance >= low) & (luminance <= high)
            & (red - blue >= SETTINGS["red_minus_blue_min"])
            & (red - green >= rg_low) & (red - green <= rg_high)
            & (green - blue >= SETTINGS["green_minus_blue_min"]))


def terrain_gains(rgb, pixel_m):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite RGB image")
    if np.min(rgb) < 0 or np.max(rgb) > 1 or min(pixel_m) <= 0:
        raise ValueError("RGB and pixel spacing are outside valid ranges")
    valid = dry_terrain_mask(rgb)
    if not valid.any():
        raise ValueError("No accepted dry terrain pixels; review source and classifier")
    linear = srgb_to_linear(rgb)
    median = np.median(linear[valid], axis=0)
    sigma = tuple(SETTINGS["smoothing_sigma_m"] / p for p in pixel_m)
    blur_options = dict(sigma=sigma, mode="reflect",
                        truncate=SETTINGS["gaussian_truncate_sigma"])
    weights = gaussian_filter(valid.astype(float), **blur_options)
    prior = SETTINGS["median_prior_weight"]
    smooth = np.stack([
        (gaussian_filter(linear[:, :, c] * valid, **blur_options) + prior * median[c])
        / (weights + prior) for c in range(3)
    ], axis=-1)
    gains = np.clip(smooth / median, *SETTINGS["gain_range"])
    return gains, valid, median


def build():
    source_dir = ROOT / "artifacts/reference/ortho"
    source = source_dir / "east-2022.png"
    if sha256(source) != SOURCE_HASH:
        raise ValueError("NAIP image hash changed; review source before rebuilding")
    export = json.loads((source_dir / "export.json").read_text())
    extent = export["response"]["extent"]
    if extent["spatialReference"]["wkid"] != 26910:
        raise ValueError("Expected NAD83 UTM zone 10N export")
    catalog = json.loads((source_dir / "catalog.json").read_text())
    source_attributes = catalog["features"][0]["attributes"]
    if source_attributes["Name"] != "m_3912230_sw_10_060_20220715":
        raise ValueError("Unexpected NAIP source tile")
    image = Image.open(source).convert("RGB")
    if image.size != (export["response"]["width"], export["response"]["height"]):
        raise ValueError("Image size does not match export extent")
    size = [extent["xmax"] - extent["xmin"], extent["ymax"] - extent["ymin"]]
    spacing = [size[1] / image.height, size[0] / image.width]
    gains, valid, median = terrain_gains(np.asarray(image, dtype=float) / 255, spacing)
    width, height = [int(np.ceil(s / SETTINGS["target_output_pixel_m"])) for s in size]
    # Filter float gains before quantization; this PNG contains linear numeric data.
    reduced = np.stack([np.asarray(Image.fromarray(gains[:, :, c].astype(np.float32))
                                  .resize((width, height), Image.Resampling.BOX))
                        for c in range(3)], axis=-1)
    encoded = np.rint(np.clip(reduced * 0.5, 0, 1) * 255).astype(np.uint8)
    output_dir = ROOT / "godot/assets/materials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "terrain_macro.png"
    Image.fromarray(encoded).save(output)
    origin = json.loads((ROOT / "godot/data/track.json").read_text())["origin"]
    metadata = {
        "schema_version": 1,
        "purpose": "Artistic broad dry terrain color gains, not measured reflectance or land cover",
        "source": {"image_sha256": SOURCE_HASH, "catalog": source_attributes,
                   "export_request": export["request"], "extent": extent,
                   "license": "Public domain", "attribution": "USDA NAIP; USGS The National Map",
                   "license_evidence": "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer?f=pjson"},
        "local_origin_xz": [extent["xmin"] - origin["easting"], origin["northing"] - extent["ymax"]],
        "local_size_xz": size,
        "mapping": "uv = (world_xz - local_origin_xz) / local_size_xz; PNG top is north",
        "settings": SETTINGS,
        "accepted_pixel_fraction": float(valid.mean()),
        "accepted_terrain_median_linear_rgb": median.tolist(),
        "algorithm": "Select tan RGB pixels, convert to linear, Gaussian normalized convolution with 0.02 median prior, divide by accepted median, clamp gains, area downsample, encode gain/2",
        "encoding": "RGB8 linear numerical data; sample without source_color/sRGB conversion, decode texture.rgb * 2.0; quantization error at most 1/255 in gain",
        "output_dimensions": [width, height],
        "output_pixel_m": [size[0] / width, size[1] / height],
        "output_sha256": sha256(output),
        "limitations": ["Historical July 2022 appearance predates repave",
                        "Color selection is not a reviewed semantic terrain mask; tan objects and weak shadows may survive",
                        "Median prior gives neutral gains where accepted terrain is absent, including orchard and pit interiors",
                        "Source lighting and fixed display stretch are not physically removed",
                        "Use only as restrained terrain material modulation, never for road boundaries, geometry or friction",
                        "Outside image extent blend to neutral gain; never repeat the image"],
    }
    (output_dir / "terrain_macro.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ["accepted_pixel_fraction", "output_dimensions", "output_pixel_m", "output_sha256"]}))


if __name__ == "__main__":
    build()
