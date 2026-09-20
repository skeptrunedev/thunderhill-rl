# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "pillow==11.3.0", "pyproj==3.7.2"]
# ///
"""Reproduce historical aerial cross section candidates, not surveyed track limits.

Uses public USGS NAIP imagery and the locally acquired ODbL OSM route.
Outputs remain in ignored artifacts/reference/ortho. Inspect every accepted section.
"""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reference/ortho"
BASE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"


def get_json(path, params):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    if "error" in data:
        raise RuntimeError(data["error"])
    return url, data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog_url, catalog = get_json("/query", {"f": "json", "where": "OBJECTID=23435", "outFields": "*", "returnGeometry": "false"})
    source = catalog["features"][0]["attributes"]
    if source["Name"] != "m_3912230_sw_10_060_20220715":
        raise RuntimeError("Catalog identifier changed; review imagery source")
    url, export = get_json("/exportImage", {"f": "json", "bbox": "556880,4376248,558008,4377699", "bboxSR": 26910, "imageSR": 26910, "size": "1880,2419", "format": "png", "interpolation": "RSP_NearestNeighbor", "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [23435]})})
    with urllib.request.urlopen(export["href"], timeout=60) as response:
        raw = response.read()
    (OUT / "east-2022.png").write_bytes(raw)
    im = Image.open(OUT / "east-2022.png").convert("RGB")
    pixels = np.array(im)
    extent = export["extent"]
    dx = (extent["xmax"] - extent["xmin"]) / im.width
    dy = (extent["ymax"] - extent["ymin"]) / im.height
    def pixel(xy):
        return np.column_stack(((xy[:, 0] - extent["xmin"]) / dx, (extent["ymax"] - xy[:, 1]) / dy))
    route_path = ROOT / "artifacts/reference/geometry/east-centerline.geojson"
    coords = json.loads(route_path.read_text())["features"][0]["geometry"]["coordinates"]
    project = Transformer.from_crs(4326, 26910, always_xy=True)
    route = np.array([project.transform(*p) for p in coords])
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    def point(s):
        return np.array([np.interp(s % cumulative[-1], cumulative, route[:, i]) for i in range(2)])
    sections = []
    overview = im.copy()
    draw = ImageDraw.Draw(overview)
    tiles = []
    for n, station in enumerate(np.arange(100, cumulative[-1], 200)):
        center = point(station)
        tangent = point(station + 8) - point(station - 8)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        offsets = np.arange(-25, 25.01, 0.3)
        xy = center + offsets[:, None] * normal
        uv = pixel(xy)
        ij = np.floor(uv).astype(int)
        if (ij < 0).any() or (ij[:, 0] >= im.width).any() or (ij[:, 1] >= im.height).any():
            raise RuntimeError("Profile falls outside imagery")
        rgb = pixels[ij[:, 1], ij[:, 0]].astype(float)
        # Neutral pavement vs brown terrain. Candidate edge only, not semantic truth.
        score = (rgb[:, 0] - rgb[:, 2]) / np.maximum(rgb[:, 0] + rgb[:, 2], 1)
        bands = []
        for threshold in (0.05, 0.07, 0.09):
            road = score < threshold
            starts = np.flatnonzero(road & ~np.r_[False, road[:-1]])
            ends = np.flatnonzero(road & ~np.r_[road[1:], False])
            candidates = [(a, b) for a, b in zip(starts, ends) if offsets[b] >= -3 and offsets[a] <= 3]
            if not candidates:
                bands.append(None)
                continue
            a, b = max(candidates, key=lambda pair: pair[1] - pair[0])
            bands.append({"threshold": threshold, "left_offset_m": round(float(offsets[a] - 0.15), 2), "right_offset_m": round(float(offsets[b] + 0.15), 2), "width_m": round(float(offsets[b] - offsets[a] + 0.3), 2)})
        section = {"id": f"S{n+1:02}", "station_m_from_osm_first_node": float(station), "utm_center": center.tolist(), "utm_normal": normal.tolist(), "candidates": bands, "status": "unreviewed historical color boundary, not legal track boundary"}
        sections.append(section)
        p = pixel(center[None, :])[0]
        draw.line([tuple(uv[0]), tuple(uv[-1])], fill="cyan", width=2)
        draw.text(tuple(p + [5, 5]), section["id"], fill="yellow", stroke_width=1, stroke_fill="black")
        # Rotated strip: horizontal axis follows the cross section, vertical along road.
        uu = np.linspace(-25, 25, 400)
        vv = np.linspace(12.5, -12.5, 200)
        grid = center + uu[None, :, None] * normal + vv[:, None, None] * tangent
        pp = pixel(grid.reshape(-1, 2)).astype(int)
        if (pp[:, 0] < 0).any() or (pp[:, 1] < 0).any() or (pp[:, 0] >= im.width).any() or (pp[:, 1] >= im.height).any():
            raise RuntimeError("Section image falls outside imagery")
        tile = Image.new("RGB", (400, 240), "#1b222c")
        tile.paste(Image.fromarray(pixels[pp[:, 1], pp[:, 0]].reshape(200, 400, 3)), (0, 25))
        td = ImageDraw.Draw(tile)
        td.text((8, 7), f'{section["id"]}  s={station:.0f}m  strip width=50m', fill="white")
        td.line((0, 125, 399, 125), fill="cyan")
        if bands[1]:
            for key in ("left_offset_m", "right_offset_m"):
                x = int((bands[1][key] + 25) / 50 * 399)
                td.line((x, 100, x, 150), fill="red", width=2)
        td.text((8, 228), 'widths: ' + ', '.join(str(b["width_m"]) if b else "none" for b in bands), fill="white")
        tiles.append(tile)
    overview.thumbnail((1100, 1500))
    overview.save(OUT / "section-overview.jpg", quality=95)
    for page, start in enumerate(range(0, len(tiles), 12), 1):
        group = tiles[start:start+12]
        sheet = Image.new("RGB", (1200, 240 * ((len(group)+2)//3)), "#1b222c")
        for i, tile in enumerate(group):
            sheet.paste(tile, ((i%3)*400, (i//3)*240))
        sheet.save(OUT / f"sections-{page}.jpg", quality=95)
    report = {"source": source, "catalog_url": catalog_url, "export_request": url, "extent": extent, "image_sha256": hashlib.sha256(raw).hexdigest(), "sample_spacing_m": [dx, dy], "route_sha256": hashlib.sha256(route_path.read_bytes()).hexdigest(), "route_license": "ODbL 1.0, OpenStreetMap contributors", "imagery_license": "USGS/USDA NAIP public domain", "method": "Neutral color transition across approximate OSM tangent, three thresholds; manual review required", "limitations": ["2022 imagery predates repave", "0.3m sampling oversamples 0.6m imagery and does not improve source accuracy", "curbs, shadows, junctions and gravel may confound detection", "imagery registration and OSM tangent errors are not included in threshold spread"], "sections": sections}
    (OUT / "width-candidates.json").write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([{k:s[k] for k in ("id", "station_m_from_osm_first_node", "candidates")} for s in sections], indent=2))


if __name__ == "__main__":
    main()
