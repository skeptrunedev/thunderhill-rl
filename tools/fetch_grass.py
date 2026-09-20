# /// script
# requires-python = ">=3.11"
# dependencies = ["Pillow==12.1.1"]
# ///
"""Acquire pinned CC0 grass, preserving the author's geometry and atlas UVs.

Only three complete source tufts are extracted; no arbitrary atlas planes or
mesh simplification. Source display layout translations are intentionally omitted.
"""
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://dl.polyhaven.org/file/ph-assets/Models/"
SOURCES = [
    ("gltf/1k/grass_medium_01/grass_medium_01_1k.gltf", "28dd8247f3faa749bf87a0b84be01e7e"),
    ("gltf/8k/grass_medium_01/grass_medium_01.bin", "e0527a0561c2dae5b7b5c1b478d8c6bf"),
    ("png/1k/grass_medium_01/grass_medium_01_dry_diff_1k.png", "6f5ae8d4f152542de30aad6b6a3ea327"),
    ("png/1k/grass_medium_01/grass_medium_01_alpha_1k.png", "6b5b251ef1ca9a69ab9f49def66d758a"),
]


def main():
    raw_dir = ROOT / "artifacts/reference/grass"
    output = ROOT / "godot/assets/grass"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    provenance = []
    for suffix, md5 in SOURCES:
        path = raw_dir / suffix.split("/")[-1]
        if not path.exists():
            with urllib.request.urlopen(BASE + suffix, timeout=90) as response:
                path.write_bytes(response.read())
        raw = path.read_bytes()
        assert hashlib.md5(raw).hexdigest() == md5, f"Publisher hash mismatch: {path}"
        provenance.append(dict(url=BASE+suffix, publisher_md5=md5, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    gltf = json.loads((raw_dir / "grass_medium_01_1k.gltf").read_text())
    binary = (raw_dir / "grass_medium_01.bin").read_bytes()

    def accessor(index):
        a = gltf["accessors"][index]
        view = gltf["bufferViews"][a["bufferView"]]
        count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        code = {5126: "f", 5123: "H", 5125: "I"}[a["componentType"]]
        fmt = "<" + code*count
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", struct.calcsize(fmt))
        rows = [struct.unpack_from(fmt, binary, offset + i*stride) for i in range(a["count"])]
        return [r[0] for r in rows] if count == 1 else [[round(v, 8) for v in row] for row in rows]

    selected = ["grass_medium_01_tall_a_LOD0", "grass_medium_01_tiny_a_LOD0", "grass_medium_01_tiny_e_LOD0"]
    meshes = []
    for name in selected:
        node = next(n for n in gltf["nodes"] if n["name"] == name)
        p = gltf["meshes"][node["mesh"]]["primitives"][0]
        attributes = p["attributes"]
        indices = accessor(p["indices"])
        meshes.append(dict(name=name, positions=accessor(attributes["POSITION"]), normals=accessor(attributes["NORMAL"]), uv=accessor(attributes["TEXCOORD_0"]), indices=indices, triangles=len(indices)//3))
    image = Image.open(raw_dir / "grass_medium_01_dry_diff_1k.png").convert("RGBA")
    alpha = Image.open(raw_dir / "grass_medium_01_alpha_1k.png").convert("L")
    assert image.size == alpha.size == (1024, 1024)
    image.putalpha(alpha)
    image.save(output / "dry_grass_rgba.png")
    metadata = dict(source="https://polyhaven.com/a/grass_medium_01", authors=["Rico Cilliers", "Rob Tuytel"], license="CC0-1.0", license_url="https://polyhaven.com/license", sources=provenance, transformation="Three unchanged author tuft meshes, positions/normals/UVs rounded to 8 decimals, scene display translations omitted. Dry diffuse RGB combined with publisher alpha. Godot reverses glTF winding; placement and scale are artistic estimates, not surveyed Thunderhill vegetation.", texture_sha256=hashlib.sha256((output / "dry_grass_rgba.png").read_bytes()).hexdigest())
    (output / "grass.json").write_text(json.dumps(dict(meshes=meshes, metadata=metadata), separators=(",", ":"))+"\n")
    print(json.dumps({"meshes": [(m["name"],m["triangles"]) for m in meshes], "rgba_bytes": (output / "dry_grass_rgba.png").stat().st_size}))


if __name__ == "__main__":
    main()
