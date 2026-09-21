"""Reward and independent recording checks for short road telemetry RL rollouts."""

import json
import math
from pathlib import Path

from lap_policy import parse_action

REWARD_VERSION = "road-first-action-monte-carlo-v1"
FAILURE_PENALTY = -25.0
SYNTAX_PENALTY = -10.0


def physical_snapshot(observation):
    """Ignore episode labels while retaining all observed physical state."""
    return {key: observation[key] for key in ("tick", "state", "track")}


def rollout_reward(transitions, prefix_tick, invalid_syntax=False):
    """Sum only post prefix progress, penalizing rider and syntax failures once."""
    scored = [row for row in transitions if row["tick"] > prefix_tick]
    progress = sum(row["reward_components"]["legal_progress_m"] for row in scored)
    failed = any(
        row["state"]["crashed"] or not row["track"]["lap_valid"] for row in scored
    )
    result = {
        "version": REWARD_VERSION,
        "legal_progress_m": progress,
        "failure_penalty": FAILURE_PENALTY if failed else 0.0,
        "syntax_penalty": SYNTAX_PENALTY if invalid_syntax else 0.0,
        "scored_ticks": len(scored),
        "excluded_prefix_ticks": prefix_tick,
        "recorded_prefix_ticks": sum(row["tick"] <= prefix_tick for row in transitions),
    }
    result["total"] = progress + result["failure_penalty"] + result["syntax_penalty"]
    if not all(
        math.isfinite(result[key])
        for key in ("legal_progress_m", "failure_penalty", "syntax_penalty", "total")
    ):
        raise ValueError("Nonfinite rollout reward")
    return result


def audit_rollout(paths, record, track_sha256):
    """Reconstruct submitted controls and reward independently from game files."""
    paths = list(paths)
    matches = []
    for candidate in paths:
        path = Path(candidate)
        with path.open() as source:
            header = json.loads(next(source))
        if header.get("episode_id") == record["episode_id"]:
            matches.append((path, header))
    if len(matches) != 1:
        raise ValueError("Expected exactly one episode recording")
    path, header = matches[0]
    if (
        header.get("type") != "episode"
        or header["policy_id"] != record["policy_id"]
        or header["track_sha256"] != track_sha256
        or header["initial_state"] != record["reset_snapshot"]["state"]
    ):
        raise ValueError("Recording header provenance mismatch")
    initial_tick = header["initial_state"]["tick"]
    snapshot = record.get("snapshot_provenance")
    if snapshot:
        if (
            header.get("snapshot") != snapshot
            or snapshot.get("version") != "worker-snapshot-v1"
            or snapshot.get("source_tick") != initial_tick
            or initial_tick != record["prefix_tick"]
            or record["reset_snapshot"] != record["branch_snapshot"]
            or not record.get("prefix_source_provenance", {}).get(
                "all_source_controls_and_states_verified"
            )
        ):
            raise ValueError("Snapshot reset provenance mismatch")
        source_matches = []
        for candidate in paths:
            with Path(candidate).open() as source:
                source_header = json.loads(next(source))
                if source_header.get("episode_id") != snapshot["source_episode_id"]:
                    continue
                last_source = None
                snapshot_recorded = False
                for line in source:
                    row = json.loads(line)
                    if row.get("type") == "transition":
                        last_source = row
                    elif (
                        row.get("type") == "snapshot"
                        and row.get("provenance") == snapshot
                    ):
                        snapshot_recorded = True
            if last_source is None:
                raise ValueError("Snapshot source recording is empty")
            if not snapshot_recorded:
                raise ValueError("Snapshot creation is absent from source recording")
            source_matches.append((source_header, last_source))
        if len(source_matches) != 1:
            raise ValueError("Snapshot source recording missing or ambiguous")
        source_header, source_last = source_matches[0]
        if (
            source_header["policy_id"] != snapshot["source_policy_id"]
            or source_header["track_sha256"] != track_sha256
            or source_last["tick"] != initial_tick
            or source_last["state"] != header["initial_state"]
            or source_last["terminated"]
            or source_last["truncated"]
            or not source_last["track"]["lap_valid"]
        ):
            raise ValueError("Snapshot source state differs from restored recording")
        initial_track = header.get("initial_track", {})
        for source_key, initial_key in (
            ("progress", "last_progress"),
            ("lap_valid", "lap_valid"),
            ("legal_distance", "legal_distance"),
            ("next_gate", "next_gate"),
        ):
            if source_last["track"].get(source_key) != initial_track.get(initial_key):
                raise ValueError(f"Snapshot restored track {initial_key} differs")
    elif initial_tick != 0 or header["start_station"] != 0:
        raise ValueError("Nonzero initial state requires snapshot provenance")
    with path.open() as source:
        next(source)
        transitions = [json.loads(line) for line in source]
    decisions = record["decisions"]
    cursor = initial_tick
    invalid_syntax = False
    for index, decision in enumerate(decisions):
        if decision["before_tick"] != cursor:
            raise ValueError("Decision does not follow prior physics tick")
        if decision["phase"] not in ("prefix", "sampled", "continuation"):
            raise ValueError("Unknown decision phase")
        expected_hash = (
            record["prefix_adapter_sha256"]
            if decision["phase"] == "prefix"
            else record["current_adapter_sha256"]
        )
        if decision["adapter_sha256"] != expected_hash:
            raise ValueError("Decision checkpoint mismatch")
        if "error" in decision:
            if index != len(decisions) - 1 or "controls" in decision:
                raise ValueError("Invalid completion was executed")
            try:
                parse_action(decision["completion"])
            except ValueError:
                invalid_syntax = True
                continue
            raise ValueError("Valid completion marked as invalid")
        controls = parse_action(decision["completion"])
        if controls != decision["controls"]:
            raise ValueError("Submitted controls differ from generated controls")
        end = decision["after_tick"]
        if not cursor < end <= cursor + 12 or end > initial_tick + len(transitions):
            raise ValueError("Invalid action interval")
        for row in transitions[cursor - initial_tick : end - initial_tick]:
            if (
                row["type"] != "transition"
                or row["episode_id"] != record["episode_id"]
                or row["policy_id"] != record["policy_id"]
                or row["previous_tick"] != cursor
                or row["tick"] != cursor + 1
                or row["requested_controls"] != controls
            ):
                raise ValueError("Transition provenance mismatch")
            if (row["terminated"] or row["truncated"]) and row[
                "tick"
            ] != initial_tick + len(transitions):
                raise ValueError("Transition after terminal state")
            cursor += 1
        if end - decision["before_tick"] < 12 and not (
            transitions[end - initial_tick - 1]["terminated"]
            or transitions[end - initial_tick - 1]["truncated"]
        ):
            raise ValueError("Short action without termination")
    final = record["final_observation"]
    if (
        cursor != initial_tick + len(transitions)
        or cursor != final["tick"]
        or not final["rollout_valid"]
    ):
        raise ValueError("Recording and final tick mismatch")
    if (
        final["episode_id"] != record["episode_id"]
        or final["policy_id"] != record["policy_id"]
    ):
        raise ValueError("Final identity mismatch")
    if transitions:
        last = transitions[-1]
        for key in (
            "state",
            "terminated",
            "truncated",
            "termination_reason",
            "truncation_reason",
        ):
            if last[key] != final[key]:
                raise ValueError(f"Final {key} differs from recording")
        for key in last["track"].keys() & final["track"].keys():
            if last["track"][key] != final["track"][key]:
                raise ValueError(f"Final track {key} differs from recording")
    elif final["state"] != header["initial_state"]:
        raise ValueError("Empty snapshot rollout changed physical state")
    prefix_tick = record["prefix_tick"]
    if not snapshot:
        if prefix_tick <= 0 or prefix_tick > len(transitions):
            raise ValueError("Missing physical prefix")
        prefix = transitions[prefix_tick - 1]
        if prefix["state"] != record["branch_snapshot"]["state"]:
            raise ValueError("Branch state differs from recorded prefix")
    if any(
        row["tick"] <= prefix_tick and not row["track"]["lap_valid"]
        for row in transitions
    ):
        raise ValueError("Invalid prefix lap")
    recomputed = rollout_reward(transitions, prefix_tick, invalid_syntax)
    if recomputed != record["reward_components"]:
        raise ValueError("Reward does not match independent recording")
    return {
        "recording": str(path),
        "transitions": len(transitions),
        "reward_verified": True,
    }
