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
import threading
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
            # Overlapping mosaics fetch the same tile from several threads; each
            # writes its own temporary file and the atomic replace keeps one.
            tmp = path.with_suffix(f"{path.suffix}.{threading.get_ident()}.part")
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

def _query_url(base: str, query) -> str:
    return base + ("&" if "?" in base else "?") + urllib.parse.urlencode(query)


def _native_request(spec: dict, src_crs: str, box, width: int, height: int, *, dem: bool = False) -> str:
    """URL for one image of box (x0, y0, x1, y1) in src_crs at width x height pixels."""
    kind = spec["kind"]
    x0, y0, x1, y1 = box
    if kind in ("wms", "wms_dem"):
        version = spec.get("version", "1.3.0")
        axis_swap = version == "1.3.0" and src_crs in ("EPSG:4326", "EPSG:4258")
        bbox = f"{y0},{x0},{y1},{x1}" if axis_swap else f"{x0},{y0},{x1},{y1}"
        query = {"SERVICE": "WMS", "VERSION": version, "REQUEST": "GetMap", "LAYERS": spec["layer"],
                 "STYLES": spec.get("style", ""), ("CRS" if version == "1.3.0" else "SRS"): src_crs,
                 "BBOX": bbox, "WIDTH": width, "HEIGHT": height,
                 "FORMAT": spec.get("format", "image/geotiff" if dem else "image/jpeg"),
                 **spec.get("params", {})}
        return _query_url(spec["url"], query)
    if kind == "arcgis_image":
        epsg = src_crs.split(":")[1]
        query = {"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": epsg, "imageSR": epsg,
                 "size": f"{width},{height}", "interpolation": "RSP_BilinearInterpolation", "f": "image",
                 **({"format": "tiff", "pixelType": "F32"} if dem else {"format": spec.get("format", "jpg")}),
                 **spec.get("params", {})}
        return spec["url"].rstrip("/") + "/exportImage?" + urllib.parse.urlencode(query)
    if kind == "ogcapi_map":
        query = {"f": spec.get("format", "png"), "bbox": f"{x0},{y0},{x1},{y1}",
                 "bbox-crs": spec.get("crs_uri", src_crs), "crs": spec.get("crs_uri", src_crs),
                 "width": width, "height": height, **spec.get("params", {})}
        return _query_url(spec["url"], query)
    if kind == "wcs":
        query = [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
                 ("COVERAGEID", spec["coverage"]), ("FORMAT", spec.get("format", "image/tiff")),
                 ("SUBSET", f"{spec.get('axis_x', 'x')}({x0},{x1})"),
                 ("SUBSET", f"{spec.get('axis_y', 'y')}({y0},{y1})"), *spec.get("params", {}).items()]
        return _query_url(spec["url"], query)
    if kind == "wcs10":
        query = {"SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage", "COVERAGE": spec["coverage"],
                 "CRS": src_crs, "BBOX": f"{x0},{y0},{x1},{y1}", "WIDTH": width, "HEIGHT": height,
                 "FORMAT": spec.get("format", "GeoTIFF"), **spec.get("params", {})}
        return _query_url(spec["url"], query)
    raise ValueError(f"Unknown request kind {kind}")


def _source_box(crs: str, src_crs: str, bounds, margin: float):
    x0, y0, x1, y1 = bounds
    box = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    if src_crs == crs:
        return box
    from rasterio.warp import transform_bounds
    return transform_bounds(crs, src_crs, *box, densify_pts=21)


def imagery_tile(spec: dict, product: Product, crs: str, bounds, pixels: int, key: str) -> np.ndarray:
    """An RGB uint8 image exactly covering bounds (x0, y0, x1, y1) in the circuit CRS.

    Services that render in the circuit CRS return the exact tile. Otherwise the
    source is requested in its own CRS around the tile and warped bilinearly.
    """
    kind = spec["kind"]
    resolution = (bounds[2] - bounds[0]) / pixels
    if kind == "xyz":
        mosaic, transform = xyz_mosaic(spec, product, crs, bounds, resolution)
        out = warp_into(mosaic, transform, "EPSG:3857", bounds, resolution, crs)
        return np.clip(np.round(np.moveaxis(out, 0, -1)), 0, 255).astype(np.uint8)
    if kind == "cog":
        out = cog_window(spec, product, crs, bounds, resolution, key)
        return np.clip(np.round(np.moveaxis(out[:3], 0, -1) * spec.get("scale", 1.0)), 0, 255).astype(np.uint8)
    src_crs = spec.get("crs", crs)
    if src_crs == crs:
        raw = product.get(key, _native_request(spec, crs, bounds, pixels, pixels), suffix=".img", expect=("image/",))
        image = decode_image(raw)
        if image.shape[:2] != (pixels, pixels):
            raise RuntimeError(f"{kind} returned {image.shape} for {key}")
        return image
    box = _source_box(crs, src_crs, bounds, 8 * resolution)
    native = spec.get("native_units_per_pixel")
    scale = native if native else resolution
    width = min(4096, int(math.ceil((box[2] - box[0]) / scale)))
    height = min(4096, int(math.ceil((box[3] - box[1]) / scale)))
    raw = product.get(key, _native_request(spec, src_crs, box, width, height), suffix=".img", expect=("image/",))
    image = decode_image(raw)
    transform = from_origin(box[0], box[3], (box[2] - box[0]) / image.shape[1], (box[3] - box[1]) / image.shape[0])
    out = warp_into(image, transform, src_crs, bounds, resolution, crs)
    return np.clip(np.round(np.moveaxis(out, 0, -1)), 0, 255).astype(np.uint8)


def xyz_mosaic(spec, product, crs, bounds, resolution):
    """Web Mercator tiles at the manifest zoom, mosaicked around bounds."""
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


def _remote_window(url: str, crs: str, bounds, path: Path):
    """Cache a window of a remote GeoTIFF (HTTP range reads) around bounds."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    if path.exists():
        return
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="120",
                      GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3",
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


def cog_window(spec, product, crs, bounds, resolution, key):
    """Warp windows of one or more remote GeoTIFFs onto the grid; later URLs fill gaps.

    The pinned hash covers decoded samples and georeferencing, not container
    bytes, which vary with the GDAL version that wrote the cache.
    """
    out = None
    for index, url in enumerate(spec["urls"]):
        cache_key = f"{key}-{index}"
        path = product.cache / product.name / (cache_key + ".tif")
        _remote_window(url, crs, bounds, path)
        with rasterio.open(path) as src:
            data = src.read()
            product.entries[cache_key] = sha256_bytes(data.tobytes() + str(src.transform).encode())
            nodata = src.nodata if src.nodata is not None else (0 if spec.get("zero_is_nodata") else None)
            warped = warp_into(data.astype(np.float32), src.transform, src.crs, bounds, resolution, crs,
                               nodata=nodata if nodata is not None else np.nan, bands_last=False)
        out = warped if out is None else np.where(np.isnan(out), warped, out)
    return out


# Elevation ---------------------------------------------------------------------

def _chunks(bounds, step):
    x0, y0, x1, y1 = bounds
    ys = np.arange(y0, y1, step)
    xs = np.arange(x0, x1, step)
    return [(x, y, min(x + step, x1), min(y + step, y1)) for y in ys for x in xs]


def _decode_dem(raw: bytes, fallback_crs: str):
    with MemoryFile(raw) as memory, memory.open() as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == np.float32(src.nodata)] = np.nan
        data[(data < -500) | (data > 9000)] = np.nan
        return data, src.transform, src.crs or fallback_crs


def _merge(out, warped):
    return warped if out is None else np.where(np.isnan(out), warped, out)


def elevation(spec: dict, product: Product, crs: str, bounds, resolution: float) -> np.ndarray:
    """Float32 heights on the exact grid (rows north to south); NaN outside coverage."""
    kind = spec["kind"]
    if kind == "cog":
        return cog_window(spec, product, crs, bounds, resolution, "dem")[0]
    if kind == "zip_geotiff":
        return _zip_geotiffs(spec, product, crs, bounds, resolution)
    if kind == "gsi_dem":
        return _gsi_dem(spec, product, crs, bounds, resolution)
    src_crs = spec.get("crs", crs)
    native = spec["resolution_m"]
    out = None
    for index, chunk in enumerate(_chunks(bounds, spec.get("max_pixels", 2000) * native)):
        box = _source_box(crs, src_crs, chunk, 3 * native)
        width = max(2, int(math.ceil((box[2] - box[0]) / spec.get("native_units_per_pixel", native))))
        height = max(2, int(math.ceil((box[3] - box[1]) / spec.get("native_units_per_pixel", native))))
        url = _native_request(spec, src_crs, box, width, height, dem=True)
        raw = product.get(f"dem-{index}", url, suffix=".tif",
                          expect=("image/tiff", "image/geotiff", "application/octet-stream", "image/tif"))
        data, transform, data_crs = _decode_dem(raw, src_crs)
        if kind in ("wms_dem", "arcgis_image", "wcs10") and data.shape == (height, width):
            # Some servers omit georeferencing in rendered images; the request defines it.
            transform = from_origin(box[0], box[3], (box[2] - box[0]) / width, (box[3] - box[1]) / height)
        out = _merge(out, warp_into(data, transform, data_crs, bounds, resolution, crs, nodata=np.nan)[0])
    return out


def _zip_geotiffs(spec, product, crs, bounds, resolution):
    """Zipped GeoTIFF tiles (with world files); spec['urls'] lists every tile to use."""
    import zipfile
    out = None
    for index, url in enumerate(spec["urls"]):
        raw = product.get(f"zip-{index}", url, suffix=".zip")
        folder = product.cache / product.name / f"zip-{index}"
        if not folder.exists():
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                archive.extractall(folder)
        for tif in sorted(folder.rglob("*.tif")):
            with rasterio.open(tif) as src:
                data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    data[data == np.float32(src.nodata)] = np.nan
                data_crs = src.crs or spec["crs"]
                out = _merge(out, warp_into(data, src.transform, data_crs, bounds, resolution, crs, nodata=np.nan)[0])
    return out


def _gsi_dem(spec, product, crs, bounds, resolution):
    """GSI elevation PNG tiles: h = 0.01 * (65536 R + 256 G + B), signed 24 bit; (128,0,0) is nodata."""
    to_merc = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    x0, y0, x1, y1 = bounds
    xs, ys = to_merc.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    zoom, size = spec["zoom"], 256
    span = 2 * WEB_MERCATOR_HALF / 2 ** zoom
    tx0 = int(math.floor((min(xs) - span * 0.1 + WEB_MERCATOR_HALF) / span))
    tx1 = int(math.floor((max(xs) + span * 0.1 + WEB_MERCATOR_HALF) / span))
    ty0 = int(math.floor((WEB_MERCATOR_HALF - (max(ys) + span * 0.1)) / span))
    ty1 = int(math.floor((WEB_MERCATOR_HALF - (min(ys) - span * 0.1)) / span))
    mosaic = np.full(((ty1 - ty0 + 1) * size, (tx1 - tx0 + 1) * size), np.nan, np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            try:
                raw = product.get(f"{zoom}-{tx}-{ty}", spec["url"].format(z=zoom, x=tx, y=ty), suffix=".png",
                                  expect=("image/",))
            except RuntimeError:
                continue  # 404: no survey coverage in this tile
            rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.int64)
            value = rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
            height = np.where(value < 2 ** 23, value, value - 2 ** 24) * 0.01
            height = np.where(value == 2 ** 23, np.nan, height).astype(np.float32)
            mosaic[(ty - ty0) * size:(ty - ty0 + 1) * size, (tx - tx0) * size:(tx - tx0 + 1) * size] = height
    transform = from_origin(-WEB_MERCATOR_HALF + tx0 * span, WEB_MERCATOR_HALF - ty0 * span, span / size, span / size)
    return warp_into(mosaic, transform, "EPSG:3857", bounds, resolution, crs, nodata=np.nan)[0]
