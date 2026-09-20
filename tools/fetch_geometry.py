# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.4.4", "numpy==2.4.3"]
# ///
"""Acquire pinned reference geometry. Run: uv run tools/fetch_geometry.py.

Raw OSM and derived OSM geometry retain ODbL, not the repository MIT license.
USGS terrain is public domain. Outputs stay in ignored artifacts/reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform, transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reference/geometry")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/reference/geometry-sources.json").read_text())
    osm_url = manifest["osm"]["url"]
    with urllib.request.urlopen(osm_url, timeout=60) as response:
        raw = response.read()
    osm = json.loads(raw)
    way = next(x for x in osm["elements"] if x["type"] == "way")
    if way["version"] != manifest["osm"]["version"]:
        raise RuntimeError("OSM way changed. Review and update the source manifest before acquisition.")
    nodes = {x["id"]: x for x in osm["elements"] if x["type"] == "node"}
    snapshot = [[n, nodes[n]["version"], nodes[n]["lon"], nodes[n]["lat"]] for n in way["nodes"]]
    snapshot_hash = hashlib.sha256(json.dumps(snapshot, separators=(",", ":")).encode()).hexdigest()
    if snapshot_hash != manifest["osm"]["ordered_node_snapshot_sha256"]:
        raise RuntimeError("OSM nodes changed. Review and update the source manifest before acquisition.")
    (args.output / "east-osm.json").write_bytes(raw)
    geojson = {"type": "FeatureCollection", "license": "ODbL 1.0", "attribution": "OpenStreetMap contributors", "source": osm_url, "features": [{"type": "Feature", "properties": {"osm_way": way["id"], "osm_version": way["version"], "status": "Approximate centerline, pre repave, course variant unverified"}, "geometry": {"type": "LineString", "coordinates": [[nodes[n]["lon"], nodes[n]["lat"]] for n in way["nodes"]]}}]}
    (args.output / "east-centerline.geojson").write_text(json.dumps(geojson, indent=2) + "\n")
    dem = next(p for q in manifest["catalog_queries"] for p in q["products"] if p["format"] == "GeoTIFF" and "NorthCoastRanges_B23" in p["title"])
    print("Reading USGS terrain window", flush=True)
    # HTTP range access avoids storing the full 337 MB regional tile.
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="60"):
        with rasterio.open(dem["downloadURL"]) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *manifest["area_bbox_wgs84"])
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            terrain = src.read(1, window=window, masked=True)
            if terrain.count() == 0:
                raise RuntimeError("Terrain window has no valid elevation values")
            profile = src.profile.copy()
            profile.update(width=terrain.shape[1], height=terrain.shape[0], transform=src.window_transform(window), compress="deflate")
            target = args.output / "east-terrain-1m.tif"
            with rasterio.open(target, "w", **profile) as dst:
                dst.write(terrain.filled(src.nodata), 1)
            report = {"source": dem, "source_crs": str(src.crs), "transform": list(profile["transform"]), "shape": list(terrain.shape), "grid_spacing_m": list(src.res), "valid_pixels": int(terrain.count()), "total_pixels": int(terrain.size), "elevation_min_m": float(terrain.min()), "elevation_max_m": float(terrain.max()), "vertical_datum": "NAVD88", "acquisition": "2023, before 2026 repave", "license": "USGS public domain", "modifications": "Rectangular window crop only, no resampling or elevation adjustment", "osm_license": "ODbL 1.0, separate from MIT", "osm_response_sha256": hashlib.sha256(raw).hexdigest(), "terrain_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    with rasterio.open(target) as crop:
        coordinates = geojson["features"][0]["geometry"]["coordinates"]
        xs, ys = transform("EPSG:4326", crop.crs, [c[0] for c in coordinates], [c[1] for c in coordinates])
        route_elevations = [float(v[0]) for v in crop.sample(zip(xs, ys))]
        report["osm_node_elevation_range_m"] = [min(route_elevations), max(route_elevations)]
        elevations = crop.read(1)
        gy, gx = np.gradient(elevations)
        shade = np.clip((1 - gx * 0.7 + gy * 0.5) / np.sqrt(1 + gx * gx + gy * gy), 0.2, 1.3) / 1.3
        relief = float(elevations.max() - elevations.min())
        normalized = (elevations - elevations.min()) / relief if relief > 0 else np.zeros_like(elevations)
        rgb = np.stack([180 + normalized * 50, 150 + normalized * 40, 90 + normalized * 30]) * shade
        route = rasterize([({"type": "LineString", "coordinates": list(zip(xs, ys))}, 1)], out_shape=elevations.shape, transform=crop.transform, all_touched=True)
        rgb[:, route == 1] = np.array([20, 40, 240])[:, None]
        # A georeferenced inspection image, not a material texture or new survey.
        with rasterio.open(args.output / "terrain-preview.png", "w", driver="PNG", height=elevations.shape[0], width=elevations.shape[1], count=3, dtype="uint8", transform=crop.transform, crs=crop.crs) as preview:
            preview.write(rgb.astype("uint8"))
    report["preview_license"] = "USGS public domain terrain; overlay contains OpenStreetMap contributors data under ODbL 1.0"
    (args.output / "acquisition-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
