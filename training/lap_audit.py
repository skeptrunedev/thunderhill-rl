"""Bind a learned policy's decisions to the complete simulator recording."""

from __future__ import annotations

import json
from pathlib import Path

from lap_policy import parse_action


def audit_lap(
    recording_paths,
    *,
    episode_id,
    policy_id,
    track_sha256,
    final_observation,
    decisions,
):
    """Raise on broken provenance; return success=False for an honestly failed lap.

    Decisions must be the generated decision rows, including any final invalid
    completion. A successful control holds for twelve ticks except when the
    simulator terminates or truncates within that interval.
    """

    def require(condition, message):
        if not condition:
            raise ValueError(f"Lap recording audit: {message}")

    matches = []
    for candidate in recording_paths:
        path = Path(candidate)
        with path.open() as source:
            first = source.readline()
        require(bool(first), f"empty recording {path}")
        header = json.loads(first)
        if header.get("episode_id") == episode_id:
            matches.append((path, header))
    require(len(matches) == 1, "expected exactly one recording for the episode")
    path, header = matches[0]
    require(header.get("type") == "episode", "missing episode header")
    require(header.get("policy_id") == policy_id, "header policy mismatch")
    require(header.get("track_sha256") == track_sha256, "track hash mismatch")
    require(header["initial_state"]["tick"] == 0, "episode must start at tick zero")
    require(header.get("start_station") == 0, "lap must start at station zero")
    require(final_observation.get("episode_id") == episode_id, "final episode mismatch")
    require(final_observation.get("policy_id") == policy_id, "final policy mismatch")
    require(final_observation.get("rollout_valid") is True, "invalid simulator rollout")

    actions = []
    invalid_completion = False
    adapter_hash = None
    previous_tick = 0
    for index, decision in enumerate(decisions):
        require(decision.get("episode_id") == episode_id, "decision episode mismatch")
        require(decision.get("action_index") == index, "decision order mismatch")
        digest = decision.get("adapter_sha256")
        require(
            isinstance(digest, str) and len(digest) == 64, "missing adapter fingerprint"
        )
        if adapter_hash is None:
            adapter_hash = digest
        require(digest == adapter_hash, "adapter changed during episode")
        if "error" in decision:
            require(index == len(decisions) - 1, "invalid completion must be last")
            require(
                "tick" not in decision and "controls" not in decision,
                "invalid completion was executed",
            )
            try:
                parse_action(decision["completion"])
            except ValueError:
                invalid_completion = True
            else:
                require(False, "valid completion marked invalid")
            continue
        controls = parse_action(decision["completion"])
        require(
            controls == decision.get("controls"),
            "generated and submitted controls differ",
        )
        end_tick = decision["tick"]
        require(0 < end_tick - previous_tick <= 12, "invalid action tick interval")
        actions.append((previous_tick, end_tick, controls))
        previous_tick = end_tick
    require(
        previous_tick == final_observation["tick"], "final tick differs from decisions"
    )

    gates = []
    offtrack = 0
    count = 0
    action_index = 0
    last = None
    all_lap_valid = True
    with path.open() as source:
        next(source)
        for line in source:
            row = json.loads(line)
            require(row.get("type") == "transition", "unexpected recording row")
            require(row.get("episode_id") == episode_id, "transition episode mismatch")
            require(row.get("policy_id") == policy_id, "transition policy mismatch")
            require(
                row["previous_tick"] == count and row["tick"] == count + 1,
                "noncontiguous physics ticks",
            )
            require(
                last is None or not (last["terminated"] or last["truncated"]),
                "transitions after episode end",
            )
            count += 1
            while action_index < len(actions) and count > actions[action_index][1]:
                action_index += 1
            require(action_index < len(actions), "transition has no model decision")
            start, end, controls = actions[action_index]
            require(start < count <= end, "transition outside decision interval")
            require(
                row["requested_controls"] == controls,
                "recorded controls differ from generated controls",
            )
            if count == end and end - start < 12:
                require(
                    row["terminated"] or row["truncated"],
                    "short action without episode end",
                )
            gates.extend(
                event["gate"] for event in row["events"] if event["type"] == "gate"
            )
            offtrack += not row["track"]["on_track"]
            all_lap_valid = all_lap_valid and row["track"]["lap_valid"]
            last = row
    require(count == final_observation["tick"], "recording does not reach final tick")
    if last:
        require(
            last["state"] == final_observation["state"],
            "final state differs from recording",
        )
        for key in (
            "terminated",
            "truncated",
            "termination_reason",
            "truncation_reason",
        ):
            require(last[key] == final_observation[key], f"final {key} mismatch")
        for key in set(last["track"]) & set(final_observation["track"]):
            require(
                last["track"][key] == final_observation["track"][key],
                f"final track {key} mismatch",
            )
    else:
        require(
            header["initial_state"] == final_observation["state"],
            "empty episode state changed",
        )
    success = bool(
        last
        and not invalid_completion
        and final_observation["terminated"]
        and final_observation["termination_reason"] == "lap_completed"
        and final_observation["track"]["completed_laps"] == 1
        and final_observation["track"]["lap_valid"]
        and all_lap_valid
        and not final_observation["state"]["crashed"]
        and not final_observation["truncated"]
        and gates == [*range(1, 32), 0]
        and offtrack == 0
    )
    return {
        "success": success,
        "recordings": [str(path)],
        "recorded_transitions": count,
        "gates": gates,
        "offtrack_ticks": offtrack,
        "actions": len(actions),
        "adapter_sha256": adapter_hash,
        "recording_provenance_verified": True,
    }
