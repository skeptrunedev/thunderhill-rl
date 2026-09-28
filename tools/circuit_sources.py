"""Pinned source acquisition for the MotoGP circuit builds (tools/build_circuit.py).

Every network response is cached under the ignored artifacts/circuits/<id>/raw/
tree and hashed. A product's lock is the SHA256 of its sorted (request, response
SHA256) list, pinned in data/circuits/<id>.json. A changed response fails the build
until someone reviews it and reruns with --update-lock, as fetch_geometry.py does
for the Thunderhill OSM way.

Source kinds (all return rasters on an exact north-up grid in the circuit CRS):
  wms           OGC WMS 1.3.0 GetMap in the circuit CRS; the server resamples.
  xyz           Web Mercator {z}/{x}/{y} tiles, mosaicked and warped bilinearly.
  arcgis_image  ArcGIS ImageServer exportImage in the circuit CRS.
  cog           One or more remote GeoTIFFs read by HTTP range, warped bilinearly.
  wcs           OGC WCS 2.0.1 GetCoverage GeoTIFF, warped bilinearly.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "thunderhill-rl circuit builder (https://github.com/skeptrunedev/thunderhill-rl)"
WEB_MERCATOR_HALF = 20037508.342789244


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class LockMismatch(RuntimeError):
    pass


class Product:
    """Collects (request key, response hash) pairs for one pinned product."""

    def __init__(self, name: str, cache: Path):
        self.name, self.cache = name, cache
        self.entries: dict[str, str] = {}

    def digest(self) -> str:
        rows = sorted(self.entries.items())
        return sha256_bytes(json.dumps(rows, separators=(",", ":")).encode())

    def get(self, key: str, url: str, *, data: bytes | None = None, suffix: str = "",
            expect: tuple[str, ...] = ()) -> bytes:
        """Cached HTTP GET (or POST with data). The key names the cache file."""
        path = self.cache / self.name / (key + suffix)
        if path.exists():
            raw = path.read_bytes()
        else:
            raw = http(url, data=data, expect=expect)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(raw)
            tmp.replace(path)
        self.entries[key] = sha256_bytes(raw)
        return raw


def http(url: str, *, data: bytes | None = None, expect: tuple[str, ...] = (),
         attempts: int = 5) -> bytes:
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
                kind = response.headers.get("Content-Type", "")
            if expect and not any(kind.startswith(e) for e in expect):
                raise RuntimeError(f"Unexpected content type {kind!r} from {url}: {raw[:300]!r}")
            return raw
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}: {last}")


def check_lock(product: Product, pinned: str | None, update: bool) -> str:
    actual = product.digest()
    if pinned is not None and pinned != actual and not update:
        raise LockMismatch(
            f"{product.name}: source responses changed (pinned {pinned}, got {actual}). "
            "Review the new data, then rerun with --update-lock.")
    return actual


# OpenStreetMap -----------------------------------------------------------------

def fetch_osm_ways(product: Product, ways: list[dict]) -> dict:
    """Current full way documents from the OSM API; versions must match the manifest."""
    elements = {}
    for row in ways:
        raw = product.get(f"way-{row['id']}", f"https://api.openstreetmap.org/api/0.6/way/{row['id']}/full.json",
                          suffix=".json")
        document = json.loads(raw)
        for element in document["elements"]:
            elements[(element["type"], element["id"])] = element
        way = elements[("way", row["id"])]
        if way["version"] != row["version"]:
            raise LockMismatch(f"OSM way {row['id']} is version {way['version']}, manifest pins "
                               f"{row['version']}. Review the edit, then update the manifest.")
    return elements


# Raster helpers ----------------------------------------------------------------

def decode_image(raw: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw))
    return np.asarray(image.convert("RGB"))


def grid_transform(bounds, resolution):
    x0, y0, x1, y1 = bounds
    return from_origin(x0, y1, resolution, resolution), (round((y1 - y0) / resolution),
                                                         round((x1 - x0) / resolution))


def warp_into(source: np.ndarray, source_transform, source_crs, bounds, resolution, crs,
              *, nodata=None, bands_last=True) -> np.ndarray:
    transform, (height, width) = grid_transform(bounds, resolution)
    array = np.moveaxis(source, -1, 0) if (bands_last and source.ndim == 3) else source
    if array.ndim == 2:
        array = array[None]
    out = np.full((array.shape[0], height, width), np.nan if nodata is not None else 0,
                  dtype=np.float32)
    for band in range(array.shape[0]):
        reproject(array[band].astype(np.float32), out[band], src_transform=source_transform,
                  src_crs=source_crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear, src_nodata=nodata,
                  dst_nodata=np.nan if nodata is not None else 0)
    return out


# Imagery -------------------------------------------------------------------------

def imagery_tile(spec: dict, product: Product, crs: str, bounds, pixels: int, key: str) -> np.ndarray:
    """An RGB uint8 image exactly covering bounds (x0, y0, x1, y1) in the circuit CRS."""
    kind = spec["kind"]
    x0, y0, x1, y1 = bounds
    if kind == "wms":
        query = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
                 "LAYERS": spec["layer"], "STYLES": spec.get("style", ""), "CRS": crs,
                 "BBOX": f"{x0},{y0},{x1},{y1}", "WIDTH": pixels, "HEIGHT": pixels,
                 "FORMAT": spec.get("format", "image/jpeg"), **spec.get("params", {})}
        url = spec["url"] + ("&" if "?" in spec["url"] else "?") + urllib.parse.urlencode(query)
        raw = product.get(key, url, suffix=".img", expect=("image/",))
        image = decode_image(raw)
        if image.shape[:2] != (pixels, pixels):
            raise RuntimeError(f"WMS returned {image.shape} for {key}")
        return image
    if kind == "arcgis_image":
        epsg = crs.split(":")[1]
        query = {"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": epsg, "imageSR": epsg,
                 "size": f"{pixels},{pixels}", "format": spec.get("format", "jpg"),
                 "interpolation": "RSP_BilinearInterpolation", "f": "image",
                 **spec.get("params", {})}
        url = spec["url"].rstrip("/") + "/exportImage?" + urllib.parse.urlencode(query)
        raw = product.get(key, url, suffix=".img", expect=("image/",))
        image = decode_image(raw)
        if image.shape[:2] != (pixels, pixels):
            raise RuntimeError(f"ImageServer returned {image.shape} for {key}")
        return image
    resolution = (x1 - x0) / pixels
    if kind == "xyz":
        mosaic, transform = xyz_mosaic(spec, product, crs, bounds, resolution)
        out = warp_into(mosaic, transform, "EPSG:3857", bounds, resolution, crs)
        return np.clip(np.round(np.moveaxis(out, 0, -1)), 0, 255).astype(np.uint8)
    if kind == "cog":
        out = cog_window(spec, product, crs, bounds, resolution, key)
        return np.clip(np.round(np.moveaxis(out, 0, -1) * spec.get("scale", 1.0)), 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown imagery kind {kind}")


def xyz_mosaic(spec, product, crs, bounds, resolution):
    """Web Mercator tiles at the zoom whose pixel is no coarser than resolution."""
    to_merc = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    x0, y0, x1, y1 = bounds
    xs, ys = to_merc.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    zoom = spec["zoom"]
    size = spec.get("tile_size", 256)
    span = 2 * WEB_MERCATOR_HALF / 2 ** zoom
    margin = span * 0.1
    tx0 = int(math.floor((min(xs) - margin + WEB_MERCATOR_HALF) / span))
    tx1 = int(math.floor((max(xs) + margin + WEB_MERCATOR_HALF) / span))
    ty0 = int(math.floor((WEB_MERCATOR_HALF - (max(ys) + margin)) / span))
    ty1 = int(math.floor((WEB_MERCATOR_HALF - (min(ys) - margin)) / span))
    mosaic = np.zeros(((ty1 - ty0 + 1) * size, (tx1 - tx0 + 1) * size, 3), np.uint8)

    def one(tile):
        tx, ty = tile
        url = spec["url"].format(z=zoom, x=tx, y=ty)
        raw = product.get(f"{zoom}-{tx}-{ty}", url, suffix=".img", expect=("image/",))
        return tile, decode_image(raw)

    tiles = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    with ThreadPoolExecutor(4) as pool:
        for (tx, ty), image in pool.map(one, tiles):
            if image.shape[:2] != (size, size):
                raise RuntimeError(f"Tile {tx},{ty} has shape {image.shape}")
            mosaic[(ty - ty0) * size:(ty - ty0 + 1) * size, (tx - tx0) * size:(tx - tx0 + 1) * size] = image
    transform = from_origin(-WEB_MERCATOR_HALF + tx0 * span, WEB_MERCATOR_HALF - ty0 * span,
                            span / size, span / size)
    return mosaic, transform


def cog_window(spec, product, crs, bounds, resolution, key):
    """Read and warp windows of remote GeoTIFFs. Each window's bytes are cached."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    bands = []
    for index, url in enumerate(spec["urls"]):
        cache_key = f"{key}-{index}"
        path = product.cache / product.name / (cache_key + ".tif")
        if not path.exists():
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="120",
                              CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff"):
                with rasterio.open(url) as src:
                    left, bottom, right, top = transform_bounds(crs, src.crs, *bounds, densify_pts=21)
                    pad = 4 * max(abs(src.res[0]), abs(src.res[1]))
                    window = from_bounds(left - pad, bottom - pad, right + pad, top + pad,
                                         transform=src.transform).round_offsets().round_lengths()
                    data = src.read(window=window, boundless=True,
                                    fill_value=src.nodata if src.nodata is not None else 0)
                    profile = {"driver": "GTiff", "width": data.shape[2], "height": data.shape[1],
                               "count": data.shape[0], "dtype": data.dtype, "crs": src.crs,
                               "transform": src.window_transform(window), "nodata": src.nodata,
                               "compress": "deflate"}
            path.parent.mkdir(parents=True, exist_ok=True)
            with MemoryFile() as memory:
                with memory.open(**profile) as dst:
                    dst.write(data)
                path.write_bytes(memory.read())
        raw = path.read_bytes()
        # Pin decoded samples, not container bytes, which vary with GDAL versions.
        with MemoryFile(raw) as memory, memory.open() as src:
            data = src.read()
            product.entries[cache_key] = sha256_bytes(data.tobytes() + str(src.transform).encode())
            nodata = src.nodata if spec.get("nodata", True) else None
            warped = warp_into(data, src.transform, src.crs, bounds, resolution, crs,
                               nodata=nodata, bands_last=False)
        bands.append(warped[spec.get("band", 0)] if len(spec["urls"]) > 1 else warped)
    if len(spec["urls"]) > 1:
        return np.stack(bands)
    return bands[0]


# Elevation ---------------------------------------------------------------------

def elevation(spec: dict, product: Product, crs: str, bounds, resolution: float) -> np.ndarray:
    """Float32 heights on the exact grid (rows north to south); NaN outside coverage."""
    kind = spec["kind"]
    x0, y0, x1, y1 = bounds
    if kind == "cog":
        return cog_window(spec, product, crs, bounds, resolution, "dem")[0]
    if kind == "wcs":
        rows = []
        # Split into requests of at most spec["max_pixels"] on a side.
        native = spec["resolution_m"]
        step = spec.get("max_pixels", 2000) * native
        tiles = []
        y = y0
        while y < y1:
            x = x0
            while x < x1:
                tiles.append((x, y, min(x + step, x1), min(y + step, y1)))
                x += step
            y += step
        out = np.full(grid_transform(bounds, resolution)[1], np.nan, np.float32)
        for index, (a, b, c, d) in enumerate(tiles):
            pad = 3 * native
            query = [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
                     ("COVERAGEID", spec["coverage"]), ("FORMAT", "image/tiff"),
                     ("SUBSET", f"{spec.get('axis_x', 'x')}({a - pad},{c + pad})"),
                     ("SUBSET", f"{spec.get('axis_y', 'y')}({b - pad},{d + pad})"),
                     *spec.get("params", {}).items()]
            if spec.get("subsetting_crs"):
                query.append(("SUBSETTINGCRS", spec["subsetting_crs"]))
            url = spec["url"] + ("&" if "?" in spec["url"] else "?") + urllib.parse.urlencode(query)
            raw = product.get(f"dem-{index}", url, suffix=".tif", expect=("image/tiff", "application/octet-stream", "image/geotiff"))
            with MemoryFile(raw) as memory, memory.open() as src:
                data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                data[data < -1000] = np.nan
                warped = warp_into(data, src.transform, src.crs or crs, bounds, resolution, crs,
                                   nodata=np.nan)[0]
            out = np.where(np.isnan(out), warped, out)
        rows.append(out)
        return rows[0]
    if kind == "arcgis_image":
        epsg = crs.split(":")[1]
        size = round((x1 - x0) / resolution), round((y1 - y0) / resolution)
        query = {"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": epsg, "imageSR": epsg,
                 "size": f"{size[0]},{size[1]}", "format": "tiff", "pixelType": "F32",
                 "interpolation": "RSP_BilinearInterpolation", "f": "image",
                 **spec.get("params", {})}
        url = spec["url"].rstrip("/") + "/exportImage?" + urllib.parse.urlencode(query)
        raw = product.get("dem", url, suffix=".tif", expect=("image/",))
        with MemoryFile(raw) as memory, memory.open() as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            if data.shape != (size[1], size[0]):
                raise RuntimeError(f"ImageServer DEM shape {data.shape}")
            return data
    raise ValueError(f"Unknown elevation kind {kind}")
