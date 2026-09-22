"""Load and verify a frozen prefix from an actual learned policy evaluation."""

import hashlib
import json
from pathlib import Path

from lap_policy import parse_action
from lap_rollout import recorded_transitions


def load_prefix(path, count, adapter_hash, road, tokenizer):
    """Bind model text to its original recorded controls and physical states."""
    path = Path(path).resolve()
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("adapter_sha256") != adapter_hash:
        raise ValueError(
            "Prefix source adapter does not match verified source checkpoint"
        )
    if summary.get("teacher_used_at_inference") is not False:
        raise ValueError("Prefix source must explicitly exclude teacher controls")
    recordings = summary.get("recordings", [])
    if len(recordings) != 1:
        raise ValueError("Prefix source must have exactly one simulator recording")
    recording = Path(recordings[0])
    selected = []
    with path.open() as source:
        for _ in range(count):
            line = source.readline()
            if not line:
                raise ValueError("Requested prefix exceeds source decisions")
            selected.append(json.loads(line))
    with recording.open() as source:
        header = json.loads(next(source))
        if (
            header.get("type") != "episode"
            or header["track_sha256"] != road.track_sha256
            or header["initial_state"]["tick"] != 0
            or header["start_station"] != 0
        ):
            raise ValueError("Prefix source recording header mismatch")
        # Old episode headers do not contain the exact projected initial road
        # position. A start station of zero can project just across the wrap.
        # The first prompt is verified against a real reset during replay.
        transitions = recorded_transitions(source)
        previous = None
        prefix = []
        tick = 0
        for index, decision in enumerate(selected):
            if (
                decision.get("action_index") != index
                or decision.get("episode_id") != header["episode_id"]
                or decision.get("adapter_sha256") != adapter_hash
                or decision.get("tick") != tick + 12
                or (
                    previous is not None
                    and decision.get("prompt") != road.prompt(previous)
                )
            ):
                raise ValueError(
                    f"Prefix source decision provenance mismatch at {index}"
                )
            if (
                tokenizer.decode(decision["completion_ids"], skip_special_tokens=True)
                != decision["completion"]
            ):
                raise ValueError("Prefix text differs from generated token IDs")
            controls = parse_action(decision["completion"])
            if controls != decision.get("controls"):
                raise ValueError("Prefix source controls differ from generated text")
            before_tick = tick
            for _ in range(12):
                row = next(transitions, None)
                if row is None:
                    raise ValueError("Prefix source recording ends early")
                if (
                    row.get("type") != "transition"
                    or row["episode_id"] != header["episode_id"]
                    or row["policy_id"] != header["policy_id"]
                    or row["previous_tick"] != tick
                    or row["tick"] != tick + 1
                    or row["requested_controls"] != controls
                    or not row["track"]["lap_valid"]
                    or not row["track"]["on_track"]
                    or row["terminated"]
                    or row["truncated"]
                ):
                    raise ValueError(
                        f"Prefix source transition mismatch at tick {tick + 1}"
                    )
                previous = row
                tick += 1
            prefix.append(
                {
                    "phase": "prefix",
                    "before_tick": before_tick,
                    "after_tick": tick,
                    "prompt": decision["prompt"],
                    "completion": decision["completion"],
                    "completion_ids": decision["completion_ids"],
                    "controls": controls,
                    "adapter_sha256": adapter_hash,
                    "loss_mask": [0] * len(decision["completion_ids"]),
                    "source_after_state": previous["state"],
                    "source_after_track": previous["track"],
                }
            )
    provenance = {
        "decision_path": str(path),
        "decision_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recording_path": str(recording),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source_episode_id": header["episode_id"],
        "source_policy_id": header["policy_id"],
        "adapter_sha256": adapter_hash,
        "actions": count,
        "last_tick": tick,
        "track_sha256": road.track_sha256,
        "all_source_controls_and_states_verified": True,
    }
    return prefix, provenance
