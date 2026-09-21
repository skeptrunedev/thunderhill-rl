# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Verify authoritative worker snapshots against actual headless physics branches."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from check_parallel import worker
from drive_lap import Driver


def advance(client, observation, action, action_id):
    command = {
        "op": "advance",
        "episode_id": observation["episode_id"],
        "expected_tick": observation["tick"],
        "action_id": action_id,
        "controls": action,
    }
    result = client.request(command)
    assert "error" not in result and result["rollout_valid"], result
    return result, command


def snapshot(client, observation):
    return client.request(
        {
            "op": "snapshot",
            "episode_id": observation["episode_id"],
            "expected_tick": observation["tick"],
        }
    )


def comparable(value):
    """Only episode, policy and action identity may differ across equivalent branches."""
    if isinstance(value, dict):
        return {
            key: comparable(item)
            for key, item in value.items()
            if key not in {"episode_id", "policy_id", "action_id"}
        }
    if isinstance(value, list):
        return [comparable(item) for item in value]
    return value


def check_moving_branch(client, data):
    driver = Driver(8)
    obs = client.request({"op": "reset", "policy_id": "snapshot-prefix"})
    prefix_gates = []
    for index in range(500):
        action, _ = driver.controls(obs)
        obs, old_request = advance(client, obs, action, str(index))
        assert not obs["terminated"] and not obs["truncated"]
        prefix_gates.extend(
            event["gate"]
            for row in obs["transitions"]
            for event in row["events"]
            if event["type"] == "gate"
        )
        if obs["track"]["progress"] * driver.length > 200:
            break
    else:
        raise AssertionError("Prefix did not reach gate one")
    assert prefix_gates == [1], prefix_gates
    source = client.request({"op": "observe", "episode_id": obs["episode_id"]})
    assert source["state"]["speed"] > 1
    saved = snapshot(client, source)
    assert len(saved["snapshot_id"]) == 32, saved
    assert saved["source_tick"] == source["tick"]
    assert (
        client.request({"op": "observe", "episode_id": source["episode_id"]}) == source
    )
    base_request = {
        "op": "snapshot",
        "episode_id": source["episode_id"],
        "expected_tick": source["tick"],
    }
    for bad in (
        {**base_request, "expected_tick": source["tick"] - 1},
        {**base_request, "episode_id": "other"},
        {**base_request, "state": {}},
        {"op": "reset", "initial_state": source["state"]},
    ):
        assert "error" in client.request(bad), bad
        assert (
            client.request({"op": "observe", "episode_id": source["episode_id"]})
            == source
        )
    for invalid_id in (None, 1, [], {}, "", "f" * 31, "f" * 32, "f" * 33):
        assert "error" in client.request({"op": "reset", "snapshot_id": invalid_id})
    assert "error" in client.request(
        {"op": "reset", "snapshot_id": saved["snapshot_id"], "station": 0}
    )
    assert (
        client.request({"op": "observe", "episode_id": source["episode_id"]}) == source
    )

    actions, reference = [], []
    for index in range(160):
        action, _ = driver.controls(obs)
        obs, _ = advance(client, obs, action, f"reference-{index}")
        assert not obs["terminated"] and not obs["truncated"]
        actions.append(action)
        reference.append(obs)
    gates = [
        event["gate"]
        for obs in reference
        for row in obs["transitions"]
        for event in row["events"]
        if event["type"] == "gate"
    ]
    assert gates == [2], gates
    restored_ids = []
    for repeat in range(2):
        obs = client.request(
            {
                "op": "reset",
                "snapshot_id": saved["snapshot_id"],
                "policy_id": f"snapshot-branch-{repeat}",
            }
        )
        restored_ids.append(obs["episode_id"])
        assert obs["episode_id"] != source["episode_id"]
        assert comparable(obs) == comparable(source), (obs, source)
        assert "error" in client.request(old_request), "Old episode action was accepted"
        # Same action ID is allowed in a new episode and must execute new physics,
        # rather than returning the source episode's cached response.
        for index, action in enumerate(actions):
            obs, command = advance(
                client,
                obs,
                action,
                old_request["action_id"] if index == 0 else str(index),
            )
            assert comparable(obs) == comparable(reference[index]), (repeat, index)
            assert all(
                row["episode_id"] == restored_ids[-1]
                and row["policy_id"] == f"snapshot-branch-{repeat}"
                for row in obs["transitions"]
            )
            if index == 0:
                assert client.request(command) == obs
        headers = [json.loads(path.open().readline()) for path in data.rglob("*.jsonl")]
        header = next(row for row in headers if row["episode_id"] == restored_ids[-1])
        assert header["snapshot"] == saved
        assert header["initial_state"] == source["state"]
        assert header["initial_track"]["passed_gates"] == 1
        assert header["initial_track"]["next_gate"] == 2
        assert (
            header["initial_track"]["legal_distance"]
            == source["track"]["legal_distance"]
        )
        assert (
            abs(header["start_station"] / driver.length - source["track"]["progress"])
            < 1e-12
        )
    assert len(set(restored_ids)) == 2
    # A snapshot of a branch records its immediate parent; no hidden state mutation.
    child = snapshot(client, obs)
    assert child["parent_snapshot_id"] == saved["snapshot_id"]
    for _ in range(62):
        assert "error" not in snapshot(client, obs)
    assert "limit" in snapshot(client, obs)["error"]
    assert (
        client.request({"op": "observe", "episode_id": obs["episode_id"]})["tick"]
        == obs["tick"]
    )
    return {
        "moving_snapshot_tick": source["tick"],
        "branch_actions": len(actions),
        "equivalent_branch_ticks": len(actions) * 12 * 2,
        "source_gates": prefix_gates,
        "branch_gates": gates,
        "foreign_snapshot_id": saved["snapshot_id"],
    }


def check_fresh_budget(client, foreign_id):
    obs = client.request({"op": "reset", "policy_id": "budget-prefix"})
    assert "error" in client.request({"op": "reset", "snapshot_id": foreign_id})
    obs, _ = advance(client, obs, {"throttle": 0.4}, "prefix")
    saved = snapshot(client, obs)
    branches = []
    for repeat in range(2):
        obs = client.request(
            {
                "op": "reset",
                "snapshot_id": saved["snapshot_id"],
                "policy_id": f"budget-branch-{repeat}",
            }
        )
        assert obs["tick"] == 12
        rows = []
        for index in range(4):
            obs, _ = advance(client, obs, {"throttle": 0.4}, str(index))
            rows.append(obs)
            assert obs["truncated"] == (index == 3)
        assert obs["tick"] == 60 and obs["truncation_reason"] == "episode_tick_limit"
        assert "error" in snapshot(client, obs), "Terminal snapshot accepted"
        branches.append(rows)
    assert comparable(branches[0]) == comparable(branches[1])
    return {
        "fresh_budget_ticks": 48,
        "preserved_initial_tick": 12,
        "truncation_tick": 60,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--godot", default=shutil.which("godot") or shutil.which("godot4")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.godot:
        parser.error("Provide --godot")
    with tempfile.TemporaryDirectory(prefix="thunderhill-snapshots-") as temporary:
        root = args.output.resolve() if args.output else Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        with worker(args.godot, root / "moving", 90) as (client, data):
            report = check_moving_branch(client, data)
        with worker(
            args.godot, root / "budget", 90, ("--agent-max-episode-ticks=48",)
        ) as (client, _):
            report.update(check_fresh_budget(client, report.pop("foreign_snapshot_id")))
        report["success"] = True
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
