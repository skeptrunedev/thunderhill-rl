# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "opencv-python-headless>=4.10"]
# ///
"""Acquire a pinned CC0 sky study and derive lighting direction from its HDR.

These are lighting references, not photographs of Thunderhill. Some candidates
include unrelated landscapes and must not be installed without visual review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/kloofendal_48d_partly_cloudy_puresky_2k.hdr"
MD5 = "2eba3a4d7eeb23cbfbeca364c97e7980"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["aristea_wreck_puresky", "kloppenheim_05_puresky", "kloppenheim_03_puresky", "cloud_layers", "cayley_lookout"])
    args = parser.parse_args()
    asset = args.candidate or "kloofendal_48d_partly_cloudy_puresky"
    pins = {"aristea_wreck_puresky": "e764c66f871ab0987f3fac422edc841d", "kloppenheim_05_puresky": "adb05080152dc9ee44ca41d6452748e4", "kloppenheim_03_puresky": "06abf490739e537e9339d619a2a3c941", "cloud_layers": "4f0d6e3d46dd31e8bff222ba3b9c9eed", "cayley_lookout": "050097d1e383a128b1330d6197972ec0"}
    url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/{asset}_2k.hdr" if args.candidate else URL
    expected_md5 = pins[asset] if args.candidate else MD5
    path = ROOT / "artifacts/sky-studies" / f"{asset}.hdr"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ThunderhillReferenceAudit/1.0"}), timeout=90) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    assert hashlib.md5(raw).hexdigest() == expected_md5, "Sky differs from pinned publisher hash"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None and image.shape == (1024, 2048, 3)
    luminance = image @ np.array([0.0722, 0.7152, 0.2126])
    y, x = np.unravel_index(np.argmax(luminance), luminance.shape)
    # Matches sky.gdshader's explicit atan(x,z) / acos(y) panorama mapping.
    azimuth = ((x + .5) / image.shape[1] - .5) * np.pi * 2
    polar = (y + .5) / image.shape[0] * np.pi
    toward_sun = [float(np.sin(polar)*np.sin(azimuth)), float(np.cos(polar)), float(np.sin(polar)*np.cos(azimuth))]
    metadata = {"source": "https://polyhaven.com/a/" + asset,
                "download_url": url, "authors": ["Greg Zaal"] if asset in {"cloud_layers", "cayley_lookout"} else ["Greg Zaal", "Jarod Guest"],
                "license": "CC0-1.0", "license_url": "https://polyhaven.com/license",
                "publisher_md5": expected_md5, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "dimensions": [2048, 1024], "sun_pixel_xy": [int(x),int(y)],
                "toward_sun": toward_sun,
                "status": "Generic sky reference; not Thunderhill weather or calibrated radiometry"}
    metadata["local_path"] = str(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
