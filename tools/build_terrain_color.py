# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1", "pyproj==3.7.2", "shapely==2.1.2"]
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
import shapely
from shapely import STRtree
from PIL import Image
from pyproj.enums import TransformDirection
from scipy.ndimage import gaussian_filter, map_coordinates
from terrain_datum import terrain_transform

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
    return (
        (luminance >= low)
        & (luminance <= high)
        & (red - blue >= SETTINGS["red_minus_blue_min"])
        & (red - green >= rg_low)
        & (red - green <= rg_high)
        & (green - blue >= SETTINGS["green_minus_blue_min"])
    )


def terrain_gains(rgb, pixel_m, smoothing_sigma_m=6.0):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite RGB image")
    if np.min(rgb) < 0 or np.max(rgb) > 1 or min(pixel_m) <= 0:
        raise ValueError("RGB and pixel spacing are outside valid ranges")
    if not np.isfinite(smoothing_sigma_m) or smoothing_sigma_m <= 0:
        raise ValueError("Expected positive finite smoothing sigma")
    valid = dry_terrain_mask(rgb)
    if not valid.any():
        raise ValueError("No accepted dry terrain pixels; review source and classifier")
    linear = srgb_to_linear(rgb)
    median = np.median(linear[valid], axis=0)
    sigma = tuple(smoothing_sigma_m / p for p in pixel_m)
    blur_options = {
        "sigma": sigma,
        "mode": "reflect",
        "truncate": SETTINGS["gaussian_truncate_sigma"],
    }
    weights = gaussian_filter(valid.astype(float), **blur_options)
    prior = SETTINGS["median_prior_weight"]
    smooth = np.stack(
        [
            (
                gaussian_filter(linear[:, :, c] * valid, **blur_options)
                + prior * median[c]
            )
            / (weights + prior)
            for c in range(3)
        ],
        axis=-1,
    )
    gains = np.clip(smooth / median, *SETTINGS["gain_range"])
    return gains, valid, median


def terrain_detail_gains(rgb, pixel_m):
    """Retain accepted aerial detail relative to the existing broad color layer."""
    broad, _, _ = terrain_gains(rgb, pixel_m)
    fine, _, _ = terrain_gains(rgb, pixel_m, smoothing_sigma_m=0.6)
    return np.clip(fine / broad, 0.75, 1.25)


def shoulder_coverage(road_rings, lower, size, dimensions):
    """Visual grass coverage only, using exact horizontal road boundary distance."""
    segments = []
    for ring in road_rings:
        vertices = np.asarray(ring, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("Invalid road boundary ring")
        if not np.isfinite(vertices).all():
            raise ValueError("Nonfinite road boundary")
        if not np.array_equal(vertices[0], vertices[-1]):
            vertices = np.vstack((vertices, vertices[0]))
        pairs = np.stack((vertices[:-1], vertices[1:]), axis=1)
        segments.extend(pairs[np.any(pairs[:, 0] != pairs[:, 1], axis=1)])
    if not segments:
        raise ValueError("No road boundary segments")
    tree = STRtree(shapely.linestrings(np.asarray(segments)))
    width, height = dimensions
    pixel = np.asarray(size) / [width, height]
    result = np.empty((height, width), dtype=float)
    # Bounded row batches keep point geometries modest for the full circuit map.
    for start in range(0, height, 64):
        stop = min(start + 64, height)
        x, z = np.meshgrid(
            lower[0] + (np.arange(width) + 0.5) * pixel[0],
            lower[1] + (np.arange(start, stop) + 0.5) * pixel[1],
        )
        indices, distances = tree.query_nearest(
            shapely.points(np.column_stack((x.ravel(), z.ravel()))),
            return_distance=True, all_matches=False,
        )
        if not np.array_equal(np.sort(indices[0]), np.arange(x.size)):
            raise ValueError("Nearest boundary query must return every pixel exactly once")
        ordered = np.empty(x.size, dtype=float)
        ordered[indices[0]] = distances
        t = np.clip((ordered - 0.5) / 3.0, 0, 1)
        result[start:stop] = (t * t * (3 - 2 * t)).reshape(x.shape)
    return result


def local_raster_bounds(extent, origin, transform):
    """Inverse transform a densified perimeter into an enclosing local rectangle."""
    t = np.linspace(0, 1, 257)
    e = extent["xmin"] + t * (extent["xmax"] - extent["xmin"])
    n = extent["ymin"] + t * (extent["ymax"] - extent["ymin"])
    east, north = transform.transform(
        np.concatenate(
            [e, e, np.full_like(t, extent["xmin"]), np.full_like(t, extent["xmax"])]
        ),
        np.concatenate(
            [np.full_like(t, extent["ymin"]), np.full_like(t, extent["ymax"]), n, n]
        ),
        direction=TransformDirection.INVERSE,
        errcheck=True,
    )
    x, z = east - origin["easting"], origin["northing"] - north
    return np.array([x.min(), z.min()]), np.array(
        [x.max() - x.min(), z.max() - z.min()]
    )


def resample_local_gains(
    gains, extent, origin, transform, lower, size, dimensions, samples=4
):
    """Integrate transformed bilinear samples using midpoint area quadrature.

    Each sample uses the full datum transform. Pixel centers include half a
    source pixel offset. Exterior samples get neutral gains. Runtime additionally
    blends the outer 30 metres of this enclosing rectangle to neutral.
    """
    width, height = dimensions
    result = np.zeros((height, width, 3), dtype=float)
    pixel = np.asarray(size) / [width, height]
    source_h, source_w = gains.shape[:2]
    source_pixel = [
        (extent["xmax"] - extent["xmin"]) / source_w,
        (extent["ymax"] - extent["ymin"]) / source_h,
    ]
    for iy in range(samples):
        for ix in range(samples):
            x, z = np.meshgrid(
                lower[0] + (np.arange(width) + (ix + 0.5) / samples) * pixel[0],
                lower[1] + (np.arange(height) + (iy + 0.5) / samples) * pixel[1],
            )
            east, north = transform.transform(
                x + origin["easting"], origin["northing"] - z, errcheck=True
            )
            col = (east - extent["xmin"]) / source_pixel[0] - 0.5
            row = (extent["ymax"] - north) / source_pixel[1] - 0.5
            inside = (
                (east >= extent["xmin"])
                & (east <= extent["xmax"])
                & (north >= extent["ymin"])
                & (north <= extent["ymax"])
            )
            for channel in range(3):
                value = map_coordinates(
                    gains[:, :, channel], [row, col], order=1, mode="nearest"
                )
                result[:, :, channel] += np.where(inside, value, 1.0)
    return result / samples**2


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
    origin = json.loads((ROOT / "godot/data/track.json").read_text())["origin"]
    transform, datum = terrain_transform()
    lower, local_size = local_raster_bounds(extent, origin, transform)
    width, height = (
        np.ceil(local_size / SETTINGS["target_output_pixel_m"]).astype(int).tolist()
    )
    reduced = resample_local_gains(
        gains, extent, origin, transform, lower, local_size, (width, height)
    )
    encoded = np.rint(np.clip(reduced * 0.5, 0, 1) * 255).astype(np.uint8)
    output_dir = ROOT / "godot/assets/materials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "terrain_macro.png"
    Image.fromarray(encoded).save(output)
    metadata = {
        "schema_version": 2,
        "horizontal_datum": datum,
        "purpose": "Artistic broad dry terrain color gains, not measured reflectance or land cover",
        "source": {
            "image_sha256": SOURCE_HASH,
            "catalog": source_attributes,
            "export_request": export["request"],
            "extent": extent,
            "license": "Public domain",
            "attribution": "USDA NAIP; USGS The National Map",
            "license_evidence": "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer?f=pjson",
        },
        "local_origin_xz": lower.tolist(),
        "local_size_xz": local_size.tolist(),
        "mapping": "uv = (world_xz - local_origin_xz) / local_size_xz; PNG top is north",
        "settings": SETTINGS,
        "accepted_pixel_fraction": float(valid.mean()),
        "accepted_terrain_median_linear_rgb": median.tolist(),
        "algorithm": "Select tan RGB pixels, convert to linear, Gaussian normalized convolution with 0.02 median prior, divide by accepted median, clamp gains, transform local EPSG6339 output samples to source EPSG26910, bilinear interpolation with 4 by 4 midpoint area quadrature, encode gain/2",
        "encoding": "RGB8 linear numerical data; sample without source_color/sRGB conversion, decode texture.rgb * 2.0; quantization error at most 1/255 in gain",
        "output_dimensions": [width, height],
        "output_pixel_m": [local_size[0] / width, local_size[1] / height],
        "output_sha256": sha256(output),
        "limitations": [
            "Historical July 2022 appearance predates repave",
            "Color selection is not a reviewed semantic terrain mask; tan objects and weak shadows may survive",
            "Median prior gives neutral gains where accepted terrain is absent, including orchard and pit interiors",
            "Source lighting and fixed display stretch are not physically removed",
            "Use only as restrained terrain material modulation, never for road boundaries, geometry or friction",
            "Outside image extent blend to neutral gain; never repeat the image",
        ],
    }
    (output_dir / "terrain_macro.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    detail_dimensions = np.ceil(local_size).astype(int).tolist()
    detail_gains = terrain_detail_gains(np.asarray(image, dtype=float) / 255, spacing)
    detail_rgb = resample_local_gains(
        detail_gains, extent, origin, transform, lower, local_size, detail_dimensions
    )
    surface_path = ROOT / "godot/data/surface.json"
    surface = json.loads(surface_path.read_text())
    coverage = shoulder_coverage(
        surface["road_rings"], lower, local_size, detail_dimensions
    )
    detail_rgba = np.concatenate((detail_rgb * 0.5, coverage[:, :, None]), axis=2)
    detail_path = output_dir / "terrain_detail.png"
    Image.fromarray(np.rint(np.clip(detail_rgba, 0, 1) * 255).astype(np.uint8)).save(detail_path)
    detail_metadata = {
        "schema_version": 1,
        "purpose": "Historical aerial detail and estimated visual shoulder blend; not friction or land cover classification",
        "source": metadata["source"],
        "horizontal_datum": datum,
        "track_sha256": sha256(ROOT / "godot/data/track.json"),
        "surface_sha256": sha256(surface_path),
        "local_origin_xz": lower.tolist(),
        "local_size_xz": local_size.tolist(),
        "mapping": metadata["mapping"],
        "output_dimensions": detail_dimensions,
        "output_pixel_m": (local_size / detail_dimensions).tolist(),
        "settings": {
            "fine_smoothing_sigma_m": 0.6,
            "broad_smoothing_sigma_m": 6.0,
            "detail_gain_range": [0.75, 1.25],
            "grass_blend_distance_m": [0.5, 3.5],
            "target_output_pixel_m": 1.0,
            "color_filter": SETTINGS,
        },
        "algorithm": "Accepted tan pixels only; ratio of fine to broad normalized convolution gains; same datum transform and 4 by 4 quadrature as macro. Alpha uses smoothstep of exact horizontal distance to surface road ring segments at output pixel centers.",
        "encoding": "RGBA8 linear data; RGB decodes as texture.rgb * 2.0; alpha is visual dry grass coverage (0 soil, 1 grass). No sRGB conversion.",
        "output_sha256": sha256(detail_path),
        "limitations": metadata["limitations"] + [
            "Shoulder transition distances are artistic material choices, not surveyed vegetation boundaries",
            "Boundary distance is unsigned; map is intended for offroad terrain shading only",
        ],
    }
    (output_dir / "terrain_detail.json").write_text(json.dumps(detail_metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "accepted_pixel_fraction",
                    "output_dimensions",
                    "output_pixel_m",
                    "output_sha256",
                ]
            }
        )
    )


if __name__ == "__main__":
    build()
