"""Bind a learned policy's decisions to the complete simulator recording."""

from __future__ import annotations

import json
import math
from pathlib import Path

from lap_policy import parse_action
from lap_rollout import recorded_transitions


# Godot 4.7 core/io/json.cpp:374 uses String::to_float; core/string/ustring.cpp:
# 2306-2337 counts leading zeros toward its 18-digit mantissa limit. Thus
# 0.000136217883843274 parses as 0.00013621788384327 (2.95e-14 relative).
# Full precision stringify cannot recover discarded input digits. This bound
# admits that measured serialization loss, not arbitrary small absolute errors.
# Values suffering worse precision loss fail closed instead of widening it.
RECORDED_CONTROL_REL_TOL = 1e-13
_CONTROL_FIELDS = {"steer", "throttle", "front_brake", "rear_brake", "shift"}


def recorded_controls_match(recorded, requested):
    """Compare only the simulator receipt with the exact submitted controls."""
    for controls in (recorded, requested):
        if not isinstance(controls, dict) or set(controls) != _CONTROL_FIELDS:
            return False
        for name, value in controls.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
            if name == "shift":
                if value not in (-1, 0, 1):
                    return False
            elif not (-1 if name == "steer" else 0) <= value <= 1:
                return False
    return recorded["shift"] == requested["shift"] and all(
        math.isclose(recorded[name], requested[name],
                     rel_tol=RECORDED_CONTROL_REL_TOL, abs_tol=0.0)
        for name in _CONTROL_FIELDS - {"shift"}
    )


def audit_lap(
    recording_paths,
    *,
    episode_id,
    policy_id,
    track_sha256,
    final_observation,
    decisions,
    parse_completion=parse_action,
    initial_speed_m_s=0.0, scenario_setup_ticks=0, scenario_setup_kind="neutral_coast",
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
    require(header["initial_state"].get("speed", 0) == initial_speed_m_s, "initial speed differs from scenario")
    require(scenario_setup_kind in ("neutral_coast", "speed_hold"), "invalid setup kind")
    expected_setup = 600 if scenario_setup_kind == "speed_hold" else 180
    require(scenario_setup_ticks in (0, expected_setup) and (not scenario_setup_ticks or initial_speed_m_s > 0), "invalid scenario setup")
    require(header.get("start_station") == 0, "lap must start at station zero")
    require(final_observation.get("episode_id") == episode_id, "final episode mismatch")
    require(final_observation.get("policy_id") == policy_id, "final policy mismatch")
    require(final_observation.get("rollout_valid") is True, "invalid simulator rollout")

    actions = []
    invalid_completion = False
    adapter_hash = None
    identity_kind = None
    previous_tick = 0
    for index, decision in enumerate(decisions):
        require(decision.get("episode_id") == episode_id, "decision episode mismatch")
        require(decision.get("action_index") == index, "decision order mismatch")
        digest = decision.get("adapter_sha256")
        kind = decision.get("policy_identity_kind", "adapter_weights_sha256")
        require(kind in {"adapter_weights_sha256", "api_request_config_sha256"}, "unknown policy identity kind")
        if identity_kind is None:
            identity_kind = kind
        require(kind == identity_kind, "policy identity kind changed during episode")
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
                parse_completion(decision["completion"])
            except ValueError:
                invalid_completion = True
            else:
                require(False, "valid completion marked invalid")
            continue
        controls = parse_completion(decision["completion"])
        setup = previous_tick < scenario_setup_ticks
        require(decision.get("action_source", "model") == ("scenario_setup" if setup else "model"), "scenario setup source mismatch")
        if setup:
            if scenario_setup_kind == "neutral_coast":
                require(not any(controls.values()), "scenario setup must coast with neutral controls")
            require(not decision.get("completion_ids") and not decision.get("behavior_logprobs"), "setup cannot contain training targets")
        require(
            controls == decision.get("controls")
            and recorded_controls_match(decision.get("controls"), controls),
            "generated and submitted controls differ",
        )
        end_tick = decision["tick"]
        require(0 < end_tick - previous_tick <= 12, "invalid action tick interval")
        actions.append((previous_tick, end_tick, controls))
        previous_tick = end_tick
    require(
        previous_tick == final_observation["tick"], "final tick differs from decisions"
    )

    setup_tracker = None
    if scenario_setup_ticks and scenario_setup_kind == "speed_hold":
        from driving_trajectory import TrajectoryTracker, encode_controller_controls, decode_controller_controls
        setup_tracker = TrajectoryTracker()
    gates = []
    offtrack = 0
    count = 0
    action_index = 0
    last = None
    all_lap_valid = True
    with path.open() as source:
        next(source)
        for row in recorded_transitions(source):
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
            if setup_tracker is not None and start < scenario_setup_ticks and count == start + 1:
                state = last["state"] if last else header["initial_state"]
                setup_tracker.replan([[(i + 1) * .1 * initial_speed_m_s, 0., 0.] for i in range(64)], origin=[0., 0., 0.])
                expected, _ = setup_tracker.next_controls(state["speed"], position=[0., 0., 0.], forward=[1., 0., 0.])
                expected = decode_controller_controls(encode_controller_controls(expected))
                require(recorded_controls_match(controls, expected), "setup controls differ from speed hold controller")
            require(start < count <= end, "transition outside decision interval")
            require(
                recorded_controls_match(row.get("requested_controls"), controls),
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
        "scenario_setup_actions": sum(start < scenario_setup_ticks for start, _, _ in actions),
        "model_actions": sum(start >= scenario_setup_ticks for start, _, _ in actions),
        "adapter_sha256": adapter_hash,
        "recording_provenance_verified": True,
    }
