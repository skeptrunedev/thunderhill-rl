# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "rasterio==1.4.4", "scipy==1.17.1", "pyproj==3.7.2"]
# ///
"""Sample actual surrounding USGS terrain for the distant game horizon.

Run uv run tools/build_horizon.py. This never synthesizes hill elevations.
USGS sources are read through HTTP ranges and sampled at 32m. The seamless
1/3 arc second DEM supplies areas outside the fine lidar project footprint.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates
from scipy.ndimage import map_coordinates
from pyproj.enums import TransformDirection
from terrain_datum import terrain_transform

ROOT = Path(__file__).resolve().parents[1]


def sample_window(raster, east, north):
    """Bilinear sample pixel centers; never extend edge elevations beyond coverage."""
    east, north = np.asarray(east), np.asarray(north)
    if east.shape != north.shape or not (np.isfinite(east).all() and np.isfinite(north).all()):
        raise ValueError("Invalid DEM sample coordinates")
    if raster.transform.b != 0 or raster.transform.d != 0:
        raise ValueError("Rotated DEM rasters are unsupported")
    cols = (east - raster.transform.c) / raster.transform.a - .5
    rows = (north - raster.transform.f) / raster.transform.e - .5
    if (np.any(cols < 0) or np.any(cols > raster.width - 1)
            or np.any(rows < 0) or np.any(rows > raster.height - 1)):
        raise ValueError("Transformed horizon grid leaves the source DEM")
    left, top = int(np.floor(cols.min())), int(np.floor(rows.min()))
    right = min(raster.width - 1, int(np.ceil(cols.max())))
    bottom = min(raster.height - 1, int(np.ceil(rows.max())))
    window = rasterio.windows.Window(left, top, right - left + 1, bottom - top + 1)
    raw = raster.read(1, window=window, masked=True)
    values = map_coordinates(raw.astype(float).filled(np.nan),
                            [(rows - top).ravel(), (cols - left).ravel()],
                            order=1, mode="constant", cval=np.nan).reshape(east.shape)
    return values, raw, rasterio.windows.bounds(window, raster.transform)


def main():
    track = json.loads((ROOT / "godot/data/track.json").read_text())
    detail = json.loads((ROOT / "godot/data/terrain.json").read_text())
    sources = json.loads((ROOT / "data/reference/geometry-sources.json").read_text())
    source = next(p for q in sources["catalog_queries"] for p in q["products"] if p["format"] == "GeoTIFF" and "NorthCoastRanges_B23" in p["title"])
    origin = track["origin"]
    datum, registration = terrain_transform()
    step, margin = 32.0, 2000.0
    detail_bounds = [float(detail["x0"]), float(detail["z0"]), float(detail["x0"] + (detail["nx"] - 1) * detail["step"]), float(detail["z0"] + (detail["nz"] - 1) * detail["step"])]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="60"):
        with rasterio.open(source["downloadURL"]) as raster:
            assert raster.crs.to_epsg() == 26910, "Unexpected source raster projection"
            # Convert the tile envelope into the game datum before local clipping.
            # A one metre inset exceeds the interpolation footprint; sampling below
            # independently checks every transformed point against pixel centers.
            west, south, east, north = datum.transform_bounds(
                *raster.bounds, direction=TransformDirection.INVERSE,
                densify_pts=21, errcheck=True)
            x0 = np.ceil(max(detail_bounds[0] - margin, west + 1 - origin["easting"]) / step) * step
            x1 = np.floor(min(detail_bounds[2] + margin, east - 1 - origin["easting"]) / step) * step
            z0 = np.ceil(max(detail_bounds[1] - margin, origin["northing"] - north + 1) / step) * step
            z1 = np.floor(min(detail_bounds[3] + margin, origin["northing"] - south - 1) / step) * step
            xs = np.arange(x0, x1 + .1, step)
            zs = np.arange(z0, z1 + .1, step)
            xx, zz = np.meshgrid(xs, zs)
            world_x, world_y = datum.transform(
                xx + origin["easting"], origin["northing"] - zz, errcheck=True)
            heights, raw, bounds = sample_window(raster, world_x, world_y)
            heights -= origin["elevation_m"]
            raw_hash = hashlib.sha256(np.asarray(raw).tobytes()).hexdigest()
    # Use the documented seamless 1/3 arc second product only where the
    # high resolution lidar project has no data. These are real elevations.
    fine_valid = np.isfinite(heights)
    coarse_source = {
        "title": "USGS 1/3 Arc Second n40w123 20250520",
        "source_id": "686dcf9bd4be026f4a016af7",
        "publication_date": "2025-07-08",
        "url": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/historical/n40w123/USGS_13_n40w123_20250520.tif",
        "metadata_url": "https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/1-3_arc-second/CA_NorthCoastRanges_B23/USGS_13_n40w123_20250520.xml",
    }
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="60"):
        with rasterio.open(coarse_source["url"]) as coarse:
            cx, cy = transform_coordinates("EPSG:26910", coarse.crs, world_x.ravel().tolist(), world_y.ravel().tolist())
            low_detail, coarse_raw, coarse_bounds = sample_window(
                coarse, np.asarray(cx).reshape(heights.shape), np.asarray(cy).reshape(heights.shape))
            low_detail -= origin["elevation_m"]
            assert np.isfinite(low_detail).all(), "Seamless terrain source has unavailable elevations"
            coarse_source["source_crs"] = str(coarse.crs)
            coarse_source["source_window_bounds"] = list(coarse_bounds)
            coarse_source["source_window_float32_sha256"] = hashlib.sha256(np.asarray(coarse_raw).tobytes()).hexdigest()
            coarse_source["used_sample_count"] = int((~fine_valid).sum())
            heights = np.where(fine_valid,heights,low_detail).ravel()
    assert np.isfinite(heights).all()
    assert len(heights) == len(xs) * len(zs)
    assert x0 < detail_bounds[0] < detail_bounds[2] < x1
    assert z0 < detail_bounds[1] < detail_bounds[3] < z1
    result = {"schema_version": 1, "nx": len(xs), "nz": len(zs), "step": step, "x0": float(x0), "z0": float(z0), "heights": np.round(heights, 3).tolist(), "detail_bounds": {"x0": detail_bounds[0], "z0": detail_bounds[1], "x1": detail_bounds[2], "z1": detail_bounds[3]}, "metadata": {"source": source["title"], "source_url": source["downloadURL"], "source_metadata_url": source["vendorMetaUrl"], "source_crs": "EPSG:26910", "vertical_datum": "NAVD88", "horizontal_registration": registration, "fine_source_collection_year": 2023, "coarse_source": coarse_source, "fine_source_used_sample_count": int(fine_valid.sum()), "source_window_bounds_utm": list(bounds), "source_window_shape": list(raw.shape), "source_window_float32_sha256": raw_hash, "render_sampling_m": step, "requested_margin_m": margin, "actual_margins_m": [detail_bounds[0]-float(x0), detail_bounds[1]-float(z0), float(x1)-detail_bounds[2], float(z1)-detail_bounds[3]], "coverage_rule": "Use actual 1m lidar DEM where available, documented seamless 1/3 arc second USGS DEM outside its footprint", "modifications": "Bilinear sampling of real USGS 1m and seamless 1/3 arc second elevation rasters at 32m spacing, subtraction of game elevation origin; no synthetic hills", "license": "USGS 3DEP public domain", "overlap_instruction": "Root renderer must exclude detailed patch interior and stitch boundary to detailed terrain; horizon samples are independent historical elevations, not road collision."}}
    target = ROOT / "godot/data/horizon.json"
    target.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    assert target.stat().st_size < 1_000_000
    print(json.dumps({"grid": [len(xs),len(zs)], "bytes": target.stat().st_size, "local_bounds": [float(x0),float(z0),float(x1),float(z1)], "elevation_range_m": [float(heights.min()+origin["elevation_m"]),float(heights.max()+origin["elevation_m"])], "source_window_shape": list(raw.shape), "raw_source_window_sha256": raw_hash}, indent=2))


if __name__ == "__main__":
    main()
