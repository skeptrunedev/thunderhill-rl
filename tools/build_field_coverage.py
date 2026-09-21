# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "scipy==1.17.1", "Pillow==12.1.1", "pyproj==3.7.2", "shapely==2.1.2"]
# ///
"""Rasterize reviewed aerial field annotations into linear material control data.

Run: uv run tools/build_field_coverage.py --output-dir artifacts/turn2-straw-map-v1
No image colors, invented mowing stripes, geometry, or friction are generated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from build_terrain_color import SOURCE_HASH, local_raster_bounds, sha256
from PIL import Image
from pyproj.enums import TransformDirection
from terrain_datum import terrain_transform

ROOT = Path(__file__).resolve().parents[1]


def local_polygon(vertices, extent, dimensions, origin, transform):
    """Convert source pixel center vertices with the pinned inverse datum operation."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("Expected at least three polygon pixel coordinates")
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite polygon coordinates")
    width, height = dimensions
    if np.any(vertices < 0) or np.any(vertices > np.array([width - 1, height - 1])):
        raise ValueError("Polygon vertices leave source image")
    polygon = shapely.Polygon(vertices)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Invalid or empty polygon")
    # Densify before projection so long annotation edges use the full transform.
    vertices = np.asarray(shapely.segmentize(polygon, 1.0).exterior.coords)
    east = (
        extent["xmin"]
        + (vertices[:, 0] + 0.5) * (extent["xmax"] - extent["xmin"]) / width
    )
    north = (
        extent["ymax"]
        - (vertices[:, 1] + 0.5) * (extent["ymax"] - extent["ymin"]) / height
    )
    east, north = transform.transform(
        east, north, direction=TransformDirection.INVERSE, errcheck=True
    )
    polygon = shapely.Polygon(
        np.column_stack((east - origin["easting"], origin["northing"] - north))
    )
    if (
        not polygon.is_valid
        or not np.isfinite(polygon.bounds).all()
        or polygon.area <= 0
    ):
        raise ValueError("Invalid transformed polygon")
    return polygon


def rasterize(regions, lower, size, dimensions):
    """Evaluate interior distance at local output pixel centers; outside is zero."""
    width, height = dimensions
    pixel = np.asarray(size) / dimensions
    output = np.zeros((height, width, 4), dtype=float)
    for polygon, coverage, pale, feather in regions:
        if not np.isfinite([coverage, pale, feather]).all() or not (
            0 <= coverage <= 1 and 0 <= pale <= 1 and feather > 0
        ):
            raise ValueError("Invalid material weights or interior feather")
        for start in range(0, height, 64):
            stop = min(start + 64, height)
            x, z = np.meshgrid(
                lower[0] + (np.arange(width) + 0.5) * pixel[0],
                lower[1] + (np.arange(start, stop) + 0.5) * pixel[1],
            )
            points = shapely.points(x, z)
            inside = shapely.contains(polygon, points)
            distance = shapely.distance(polygon.boundary, points)
            t = np.where(inside, np.clip(distance / feather, 0, 1), 0)
            alpha = t * t * (3 - 2 * t)
            block = output[start:stop]
            select = alpha > 0
            block[select, 0] = coverage
            block[select, 1] = pale
            block[select, 3] = alpha[select]
    return output


def build(output_dir, annotation_path):
    annotations = json.loads(annotation_path.read_text())
    source = ROOT / annotations["source_image"]
    if annotations["source_sha256"] != SOURCE_HASH or sha256(source) != SOURCE_HASH:
        raise ValueError("NAIP source hash changed")
    export_path = ROOT / "artifacts/reference/ortho/export.json"
    export = json.loads(export_path.read_text())
    extent = export["response"]["extent"]
    if extent["spatialReference"]["wkid"] != 26910:
        raise ValueError("Expected EPSG26910 source export")
    dimensions = (export["response"]["width"], export["response"]["height"])
    with Image.open(source) as image:
        if image.size != dimensions:
            raise ValueError("Source image dimensions disagree with export")
    track_path = ROOT / "godot/data/track.json"
    origin = json.loads(track_path.read_text())["origin"]
    transform, datum = terrain_transform()
    lower, size = local_raster_bounds(extent, origin, transform)
    pixel_m = annotations["target_output_pixel_m"]
    if not np.isfinite(pixel_m) or not 0.25 <= pixel_m <= 4:
        raise ValueError("Output spacing must be finite and between 0.25 and 4 metres")
    output_dimensions = np.ceil(size / pixel_m).astype(int).tolist()
    regions = []
    ids = set()
    for region in annotations["regions"]:
        if not region["id"] or region["id"] in ids:
            raise ValueError("Region identifiers must be unique and nonempty")
        ids.add(region["id"])
        polygon = local_polygon(
            region["polygon_pixels"], extent, dimensions, origin, transform
        )
        if any(polygon.intersection(previous[0]).area > 0 for previous in regions):
            raise ValueError(
                "Overlapping region interiors require explicit material priority"
            )
        regions.append(
            (
                polygon,
                region["target_grass_coverage"],
                region["pale_straw_response"],
                region["interior_feather_m"],
            )
        )
    if not regions:
        raise ValueError("No annotated regions")
    surface_path = ROOT / "godot/data/surface.json"
    road = shapely.GeometryCollection()
    # Boundary rings include infield holes; even odd filling preserves them.
    for ring in json.loads(surface_path.read_text())["road_rings"]:
        road = road.symmetric_difference(shapely.Polygon(ring))
    road_clearance = [float(polygon.distance(road)) for polygon, *_ in regions]
    if min(road_clearance) <= 0:
        raise ValueError("Field annotation intersects road geometry")
    values = rasterize(regions, lower, size, output_dimensions)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "field_coverage.png"
    Image.fromarray(np.rint(values * 255).astype(np.uint8)).save(output)
    metadata = {
        "schema_version": 1,
        "purpose": annotations["purpose"],
        "source": {
            "image": annotations["source_image"],
            "image_sha256": SOURCE_HASH,
            "export_sha256": sha256(export_path),
            "export_request": export["request"],
            "extent": extent,
            "license": "Public domain",
            "attribution": "USDA NAIP; USGS The National Map",
        },
        "annotation_path": str(annotation_path.relative_to(ROOT)),
        "annotation_sha256": sha256(annotation_path),
        "horizontal_datum": datum,
        "track_sha256": sha256(track_path),
        "surface_sha256": sha256(surface_path),
        "local_origin_xz": lower.tolist(),
        "local_size_xz": size.tolist(),
        "output_dimensions": output_dimensions,
        "output_pixel_m": (size / output_dimensions).tolist(),
        "mapping": "uv = (world_xz - local_origin_xz) / local_size_xz; PNG top is north; outside rectangle has zero alpha",
        "encoding": "RGBA8 linear numerical data without sRGB conversion. R target grass coverage; G pale straw response weight; B unused zero; A regional blend weight. Outside annotations RGBA is zero.",
        "algorithm": "Source pixel center annotation polygons densified to one pixel then inverse transformed using pinned datum grids. Interior boundary distance at local output pixel centers controls smoothstep alpha; no exterior feather and no invented stripes.",
        "regions": [
            {
                "id": annotation["id"],
                "local_area_m2": polygon.area,
                "road_clearance_m": clearance,
                "target_grass_coverage": coverage,
                "pale_straw_response": pale,
                "interior_feather_m": feather,
            }
            for annotation, (polygon, coverage, pale, feather), clearance in zip(
                annotations["regions"], regions, road_clearance
            )
        ],
        "output_sha256": sha256(output),
        "limitations": [
            "Historical July 2022 appearance predates repave and current footage",
            "Polygon boundaries and material weights are artistic estimates, not measured vegetation coverage or reflectance",
            "Only the reviewed Turn 2 interior is annotated; other regions remain unchanged",
            "Existing aerial detail may supply mowing variation; this map adds no synthetic stripe pattern",
            "Material appearance only; never use for friction, physics, or track geometry",
        ],
    }
    (output_dir / "field_coverage.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output_sha256": metadata["output_sha256"],
                "dimensions": output_dimensions,
                "regions": metadata["regions"],
                "nonzero_alpha_pixels": int(np.count_nonzero(values[:, :, 3])),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/turn2-straw-map-v1"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data/reference/turn2-straw-regions.json",
    )
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.annotations.resolve())
