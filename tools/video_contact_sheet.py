# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless>=4.10", "pillow>=11"]
# ///
"""Create timestamped inspection sheets from a local reference video.

Keep third party footage and derived sheets in ignored artifacts/, not game assets.
Run: uv run tools/video_contact_sheet.py VIDEO --output artifacts/reference/video
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive")
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("start must be finite and nonnegative")
    if args.end is not None and (not math.isfinite(args.end) or args.end <= args.start):
        parser.error("end must be finite and greater than start")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        raise RuntimeError("Video has no usable frame timing metadata")
    duration = count / fps
    if args.start >= duration or (args.end is not None and args.end > duration):
        cap.release()
        parser.error("requested time range exceeds video duration")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    index = []
    timestamp = args.start
    end = args.end if args.end is not None else duration
    while timestamp < end:
        frame_number = min(round(timestamp * fps), count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode requested frame at {timestamp:.3f}s")
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        filename = f"frame-{timestamp:07.2f}.jpg"
        rgb.save(args.output / filename, quality=92)
        rgb.thumbnail((400, 225))
        tile = Image.new("RGB", (400, 253), "#161b22")
        tile.paste(rgb, ((400 - rgb.width) // 2, 0))
        ImageDraw.Draw(tile).text((10, 234), f"{int(timestamp // 60):02}:{timestamp % 60:05.2f}", fill="white")
        frames.append(tile)
        index.append({"requested_seconds": timestamp, "frame_number": frame_number, "image": filename})
        timestamp += args.interval
    cap.release()
    for page, start in enumerate(range(0, len(frames), 12), 1):
        group = frames[start:start + 12]
        sheet = Image.new("RGB", (1200, 253 * ((len(group) + 2) // 3)), "#161b22")
        for i, tile in enumerate(group):
            sheet.paste(tile, ((i % 3) * 400, (i // 3) * 253))
        sheet.save(args.output / f"sheet-{page:02}.jpg", quality=92)
    with args.video.open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    (args.output / "index.json").write_text(json.dumps({"video": args.video.name, "fps": fps, "duration_seconds": duration, "source_sha256": source_sha256, "start_seconds": args.start, "end_seconds": end, "frames": index}, indent=2) + "\n")
    print(f"Inspected {len(frames)} sample frames across {duration:.2f} seconds; output: {args.output}")


if __name__ == "__main__":
    main()
