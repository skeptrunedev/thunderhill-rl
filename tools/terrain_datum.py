"""Use the reviewed horizontal datum operation with verified local grid files."""

import hashlib
import json
from pathlib import Path

from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]


def terrain_transform():
    manifest = ROOT / "data/reference/datum-grids.json"
    pin = json.loads(manifest.read_text())
    if (pin["source_crs"], pin["target_crs"]) != ("EPSG:6339", "EPSG:26910"):
        raise ValueError("Unexpected terrain datum pair")
    operation = pin["operation"]
    hashes = {}
    for grid in pin["grids"]:
        path = ROOT / "artifacts/reference/datum" / grid["name"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != grid["sha256"]:
            raise ValueError("Terrain datum grid hash mismatch: " + grid["name"])
        token = "grids=" + grid["name"]
        if operation.count(token) != 1:
            raise ValueError("Datum grid is not used exactly once")
        operation = operation.replace(token, "grids=" + str(path))
        hashes[grid["name"]] = digest
    transform = Transformer.from_pipeline(operation)
    return transform, {
        "source_crs": pin["source_crs"],
        "target_crs": pin["target_crs"],
        "operation": pin["operation"],
        "grid_sha256": hashes,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "vertical": "Horizontal transformation only; NAVD88 heights are unchanged",
    }
