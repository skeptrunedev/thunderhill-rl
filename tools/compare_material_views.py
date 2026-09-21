# /// script
# requires-python = ">=3.11"
# dependencies = ["Pillow==12.1.1"]
# ///
"""Create labeled reference versus game inspection sheets without color grading.

JSON pairs contain label, reference/game image paths, optional reference/game
crop rectangles, and a note identifying framing or lighting limitations.
Third party frames and generated sheets belong in ignored artifacts/.
"""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for number, pair in enumerate(spec["pairs"]):
        images, evidence = [], {}
        for side in ("reference", "game"):
            path = Path(pair[side])
            image = Image.open(path).convert("RGB")
            bounds = pair.get(side + "_crop", [0, 0, image.width, image.height])
            x0, y0, x1, y1 = bounds
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError(f"Crop outside source: {path}")
            evidence[side] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_size": image.size, "crop": bounds}
            image = image.crop(bounds)
            # Fit without upsampling; show native crop detail, not invented pixels.
            image.thumbnail((1280, 720), Image.Resampling.LANCZOS)
            images.append(image)
            evidence[side]["display_size"] = image.size
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        sheet = Image.new("RGB", (width * 2 + 36, height + 100), "#14191d")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), pair["label"], fill="white")
        for index, (side, image) in enumerate(zip(("Video reference", "Game render"), images)):
            x = 12 + index * (width + 12)
            draw.text((x, 32), side, fill="#aebeca")
            sheet.paste(image, (x, 54))
        draw.text((12, height + 65), pair["note"], fill="#e2bd8b")
        target = args.output / f"{number:02}.png"
        if target.exists():
            raise FileExistsError(target)
        sheet.save(target)
        report.append({"label": pair["label"], "note": pair["note"], "output": str(target), **evidence})
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(report)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
