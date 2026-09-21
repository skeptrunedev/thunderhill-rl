"""Extract authoritative state playback, optionally with exact model decisions.

Decision sidecars must accompany the original complete episode recording. Their
end ticks are joined to the recorded transitions before clipping. Native decision
events are preserved when clipping a recording that already contains them.
Model and generation labels are caller supplied presentation metadata, not
verified model provenance. Existing labels survive clipping without changes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def decision_rows(stream, manifest, decisions):
    """Place each accepted call before its first transition, not at its end tick."""
    if decisions is None:
        yield from (json.loads(line) for line in stream)
        return
    if "playback_clip" in manifest:
        raise ValueError("Join decisions to the original recording before clipping")
    with decisions.open() as source:
        calls = [json.loads(line) for line in source]
    # A rejected final completion has no applied controls and no transition range.
    if calls and "error" in calls[-1] and "tick" not in calls[-1]:
        calls.pop()
    previous = manifest["initial_state"]
    end_tick = previous["tick"]
    for index, call in enumerate(calls):
        if (call.get("episode_id") != manifest.get("episode_id")
                or call.get("action_index") != index
                or type(call.get("tick")) is not int
                or call["tick"] <= end_tick
                or not isinstance(call.get("completion"), str)
                or not isinstance(call.get("controls"), dict)):
            raise ValueError(f"Decision provenance or ordering mismatch at action {index}")
        end_tick = call["tick"]
    index = 0
    pending = True
    for line in stream:
        row = json.loads(line)
        if row.get("type") == "model_decision":
            raise ValueError("Recording already contains decisions; omit the sidecar")
        if row.get("type") != "transition":
            yield row
            continue
        if index >= len(calls):
            raise ValueError("Decision sidecar does not cover recorded transitions")
        call = calls[index]
        if (row.get("episode_id") != manifest.get("episode_id")
                or row.get("previous_tick") != previous["tick"]
                or row.get("tick") != previous["tick"] + 1
                or row.get("requested_controls") != call["controls"]):
            raise ValueError(f"Decision and recorded controls disagree at tick {row.get('tick')}")
        if pending:
            yield {"type": "model_decision", "episode_id": call["episode_id"],
                   "tick": row["previous_tick"], "elapsed": previous["elapsed"],
                   "action_index": call["action_index"], "text": call["completion"],
                   "controls": call["controls"]}
            pending = False
        yield row
        previous = row["state"]
        if row["tick"] == call["tick"]:
            index += 1
            pending = True
    if index != len(calls):
        raise ValueError("Recording ends before decision endpoint")


def clip_replay(source: Path, output: Path, start: float, end: float,
                decisions: Path | None = None, *, model_name: str | None = None,
                generation: int | None = None) -> dict:
    if not all(map(math.isfinite, (start, end))) or not 0 <= start < end:
        raise ValueError("Require finite 0 <= start < end seconds")
    if (model_name is None) != (generation is None):
        raise ValueError("Supply model_name and generation together")
    if model_name is not None and (
            not isinstance(model_name, str) or not model_name.strip()
            or type(generation) is not int or generation < 0):
        raise ValueError("Require a nonempty model name and nonnegative integer generation")
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
        events = []
        for row in decision_rows(stream, manifest, decisions):
            if row.get("type") == "model_decision":
                if (not isinstance(row.get("text"), str)
                        or type(row.get("tick")) is not int
                        or row["tick"] > previous["tick"]
                        or (events and row["tick"] < events[-1]["tick"])
                        or ("episode_id" in row
                            and row["episode_id"] != manifest.get("episode_id"))):
                    raise ValueError("Invalid or future model decision event")
                events.append(row)
                continue
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
            if start <= elapsed <= end:
                if initial is None:
                    initial = previous
                selected.append(row)
            previous = state
    if not selected:
        raise ValueError("Requested interval contains no recorded transitions")
    if last_seen_elapsed < end - 1e-8:
        raise ValueError("Source ends before requested clip endpoint")
    manifest["initial_state"] = initial
    if model_name is not None:
        manifest["policy_display"] = {"model_name": model_name, "generation": generation}
    # Keep only the call active at the initial state, then each call whose first
    # transition appears in the clip. Equal command text still means new calls.
    active = [event for event in events if event["tick"] <= initial["tick"]]
    visible = active[-1:] + [event for event in events
                             if initial["tick"] < event["tick"] < selected[-1]["tick"]]
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
        "model_decisions": len(visible),
        "purpose": "Authoritative state playback, not resimulation or training data",
    }
    if decisions is not None:
        provenance["decisions_file"] = decisions.name
        provenance["decisions_sha256"] = hashlib.sha256(decisions.read_bytes()).hexdigest()
    manifest["playback_clip"] = provenance
    # Exclusive creation prevents accidental replacement of source or old clips.
    with output.open("x") as stream:
        stream.write(json.dumps(manifest, separators=(",", ":")) + "\n")
        event_index = 0
        for row in selected:
            while event_index < len(visible) and visible[event_index]["tick"] <= row["previous_tick"]:
                stream.write(json.dumps(visible[event_index], separators=(",", ":")) + "\n")
                event_index += 1
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--decisions", type=Path, help="Original evaluate_lap decisions JSONL")
    parser.add_argument("--model-name", help="Caller supplied display name, paired with --generation")
    parser.add_argument("--generation", type=int,
                        help="Caller supplied experiment generation, paired with --model-name")
    args = parser.parse_args()
    if (args.model_name is None) != (args.generation is None):
        parser.error("--model-name and --generation must be supplied together")
    print(json.dumps(clip_replay(args.source, args.output, args.start, args.end,
                                args.decisions, model_name=args.model_name,
                                generation=args.generation), indent=2))


if __name__ == "__main__":
    main()
