"""Extract a state playback clip without resimulating or renumbering its ticks.

The source manifest, policy identity and physics configuration are preserved.
The preceding recorded state becomes the clip's initial state. Clips are for
visual playback, not control benchmarks, which require a tick zero start.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def clip_replay(source: Path, output: Path, start: float, end: float) -> dict:
    if not all(map(math.isfinite, (start, end))) or not 0 <= start < end:
        raise ValueError("Require finite 0 <= start < end seconds")
    if output.exists():
        raise FileExistsError(output)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with source.open() as stream:
        manifest = json.loads(next(stream))
        if manifest.get("type") != "episode":
            raise ValueError("Source must begin with an episode manifest")
        previous = manifest["initial_state"]
        if start < previous["elapsed"] - 1e-8:
            raise ValueError("Source begins after requested clip start")
        last_seen_elapsed = previous["elapsed"]
        initial = None
        selected = []
        for line in stream:
            row = json.loads(line)
            if row.get("type") != "transition":
                continue
            state = row["state"]
            elapsed = state["elapsed"]
            if not math.isfinite(elapsed) or elapsed <= previous["elapsed"]:
                raise ValueError("Source elapsed time must increase")
            if state["tick"] != previous["tick"] + 1:
                raise ValueError("Source ticks must be contiguous")
            if row["tick"] != state["tick"] or row["previous_tick"] != previous["tick"]:
                raise ValueError("Transition ticks disagree with state")
            last_seen_elapsed = elapsed
            if elapsed > end:
                break
            if elapsed >= start:
                if initial is None:
                    initial = previous
                selected.append(row)
            previous = state
    if not selected:
        raise ValueError("Requested interval contains no recorded transitions")
    if last_seen_elapsed < end - 1e-8:
        raise ValueError("Source ends before requested clip endpoint")
    manifest["initial_state"] = initial
    provenance = {
        "source_file": source.name,
        "source_sha256": digest,
        "requested_start_seconds": start,
        "requested_end_seconds": end,
        "initial_elapsed_seconds": initial["elapsed"],
        "final_elapsed_seconds": selected[-1]["state"]["elapsed"],
        "duration_seconds": selected[-1]["state"]["elapsed"] - initial["elapsed"],
        "initial_tick": initial["tick"],
        "final_tick": selected[-1]["state"]["tick"],
        "transitions": len(selected),
        "purpose": "Authoritative state playback, not resimulation or training data",
    }
    manifest["playback_clip"] = provenance
    # Exclusive creation prevents accidental replacement of source or old clips.
    with output.open("x") as stream:
        for row in [manifest, *selected]:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(clip_replay(args.source, args.output, args.start, args.end), indent=2))


if __name__ == "__main__":
    main()
