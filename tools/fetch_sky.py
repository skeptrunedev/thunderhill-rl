# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "opencv-python-headless>=4.10"]
# ///
"""Acquire pinned CC0 sky and derive its lighting direction from the actual HDR.

This is a lighting reference, not a photograph of Thunderhill. No terrain from
another location is present in this pure sky asset.
"""
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
    path = ROOT / "godot/assets/sky/kloofendal_2k.hdr"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    assert hashlib.md5(raw).hexdigest() == MD5, "Sky differs from pinned publisher hash"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None and image.shape == (1024, 2048, 3)
    luminance = image @ np.array([0.0722, 0.7152, 0.2126])
    y, x = np.unravel_index(np.argmax(luminance), luminance.shape)
    # Matches sky.gdshader's explicit atan(x,z) / acos(y) panorama mapping.
    azimuth = ((x + .5) / image.shape[1] - .5) * np.pi * 2
    polar = (y + .5) / image.shape[0] * np.pi
    toward_sun = [float(np.sin(polar)*np.sin(azimuth)), float(np.cos(polar)), float(np.sin(polar)*np.cos(azimuth))]
    metadata = {"source": "https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky",
                "download_url": URL, "authors": ["Greg Zaal", "Jarod Guest"],
                "license": "CC0-1.0", "license_url": "https://polyhaven.com/license",
                "publisher_md5": MD5, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "dimensions": [2048, 1024], "sun_pixel_xy": [int(x),int(y)],
                "toward_sun": toward_sun,
                "status": "Generic midday sky reference; not Thunderhill weather or calibrated radiometry"}
    (ROOT / "godot/data/sky.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
