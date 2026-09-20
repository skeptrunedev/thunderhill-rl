# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow==11.3.0"]
# ///
"""Acquire CC0 material candidates and a contact sheet.

Run: uv run tools/fetch_materials.py
Use --refresh-manifest explicitly to resolve current publisher metadata and hashes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/reference/material-candidates.json"
LIMIT = 50_000_000


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ThunderhillReferenceAudit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise RuntimeError("Download exceeds candidate acquisition budget")
    return raw


def resolve():
    url = "https://ambientcg.com/api/v2/full_json?id=Asphalt010&include=downloadData"
    data = json.loads(fetch(url))["foundAssets"][0]
    downloads = data["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
    chosen = next(item for item in downloads if item["attribute"] == "1K-JPG")
    entries = [{"asset": "Asphalt010", "map": "archive", "url": chosen["downloadLink"],
                "filename": chosen["fileName"], "metadata_url": url,
                "source": "https://ambientcg.com/a/Asphalt010", "author": "ambientCG / Lennart Demes",
                "license": "CC0-1.0", "license_url": "https://docs.ambientcg.com/license/"}]
    for asset, author in [("brown_mud_dry", "Rob Tuytel"), ("withered_grass", "Charlotte Baglioni")]:
        url = "https://api.polyhaven.com/files/" + asset
        data = json.loads(fetch(url))
        for channel in ("Diffuse", "nor_gl", "Rough"):
            selected = data[channel]["1k"]["jpg"]
            entries.append({"asset": asset, "map": channel, "url": selected["url"],
                            "filename": selected["url"].rsplit("/", 1)[1], "metadata_url": url,
                            "source": "https://polyhaven.com/a/" + asset, "author": author,
                            "license": "CC0-1.0", "license_url": "https://polyhaven.com/license",
                            "publisher_md5": selected["md5"]})
    return {"status": "Candidate materials, not a verified Thunderhill surface match", "resolution": "1K", "files": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reference/materials")
    args = parser.parse_args()
    manifest = resolve() if args.refresh_manifest else json.loads(MANIFEST.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    total = 0
    diffuse = []
    for entry in manifest["files"]:
        target = args.output / entry["filename"]
        raw = target.read_bytes() if target.exists() else fetch(entry["url"])
        total += len(raw)
        if total > LIMIT:
            raise RuntimeError("Total download size exceeds 50 MB")
        sha = hashlib.sha256(raw).hexdigest()
        if not args.refresh_manifest and sha != entry["sha256"]:
            raise RuntimeError("SHA256 mismatch for " + entry["filename"])
        if "publisher_md5" in entry and hashlib.md5(raw).hexdigest() != entry["publisher_md5"]:
            raise RuntimeError("Publisher checksum mismatch for " + entry["filename"])
        target.write_bytes(raw)
        entry.update(sha256=sha, bytes=len(raw))
        if entry["map"] == "archive":
            extracted = []
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for member in archive.namelist():
                    if member.endswith(("_Color.jpg", "_NormalGL.jpg", "_Roughness.jpg")):
                        contents = archive.read(member)
                        name = Path(member).name
                        (args.output / name).write_bytes(contents)
                        extracted.append({"filename": name, "sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)})
                        if member.endswith("_Color.jpg"):
                            diffuse.append((entry["asset"], args.output / name))
            if len(extracted) != 3:
                raise RuntimeError("Expected color, OpenGL normal, and roughness maps in archive")
            entry["extracted_maps"] = extracted
        elif entry["map"] == "Diffuse":
            diffuse.append((entry["asset"], target))
        print(entry["filename"], len(raw), sha, flush=True)
    sheet = Image.new("RGB", (384 * len(diffuse), 420), "#242424")
    draw = ImageDraw.Draw(sheet)
    for index, (name, path) in enumerate(diffuse):
        with Image.open(path) as source:
            sheet.paste(source.convert("RGB").resize((384, 384)), (384 * index, 36))
        draw.text((384 * index + 12, 12), name, fill="white")
    sheet.save(args.output / "candidate-sheet.jpg", quality=95)
    if args.refresh_manifest:
        manifest["download_bytes"] = total
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified download bytes:", total)


if __name__ == "__main__":
    main()
