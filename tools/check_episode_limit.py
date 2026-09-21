# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exercise exact episode budget boundaries through the real agent socket."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from check_agent import ROOT
from check_parallel import worker


def verify(client, data, limit):
    initial = client.request(
        {"op": "reset", "policy_id": "budget-check", "max_episode_ticks": 9999}
    )
    episode = initial["episode_id"]
    tick = 0
    transitions = []
    while tick < limit:
        command = {
            "op": "advance",
            "episode_id": episode,
            "expected_tick": tick,
            "action_id": str(tick),
            "controls": {"throttle": 0.3},
        }
        result = client.request(command)
        assert "error" not in result, result
        assert result["tick"] == min(tick + 12, limit)
        assert len(result["transitions"]) == min(12, limit - tick)
        assert client.request(command) == result, (
            "Duplicate changed budget or terminal event"
        )
        tick = result["tick"]
        transitions.extend(result["transitions"])
    assert result["truncated"] and not result["terminated"] and result["rollout_valid"]
    assert result["truncation_reason"] == "episode_tick_limit"
    assert result["termination_reason"] == ""
    assert sum(row["truncated"] for row in transitions) == 1
    assert (
        transitions[-1]["events"].count(
            {"type": "truncation", "reason": "episode_tick_limit"}
        )
        == 1
    )
    assert "error" in client.request(
        {**command, "action_id": "after-limit", "expected_tick": limit}
    )
    frozen = client.request({"op": "observe", "episode_id": episode})
    assert frozen["tick"] == limit and frozen["truncated"]
    fresh = client.request({"op": "reset", "policy_id": "next-budget-check"})
    assert fresh["tick"] == 0 and not fresh["truncated"] and not fresh["terminated"]
    assert fresh["truncation_reason"] == "" and fresh["termination_reason"] == ""
    rows = [
        json.loads(line)
        for path in data.rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    rows = [row for row in rows if row.get("episode_id") == episode]
    manifest = next(row for row in rows if row["type"] == "episode")
    assert manifest["episode_limits"] == {
        "version": "tick-budget-v1",
        "max_physics_ticks": limit,
    }
    recorded = [row for row in rows if row["type"] == "transition"]
    assert recorded == transitions, (
        "Recorded budget transitions differ from socket response"
    )
    return {"limit_ticks": limit, "recorded_transitions": len(recorded), "ok": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    args = parser.parse_args()
    for value, agent in (
        ("0", True),
        ("-1", True),
        ("1.5", True),
        ("bad", True),
        ("1", False),
    ):
        command = [
            args.godot,
            "--headless",
            "--path",
            str(ROOT / "godot"),
            "--",
            f"--agent-max-episode-ticks={value}",
        ]
        if agent:
            command.append("--agent-port=0")
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        assert result.returncode == 2, result.stdout + result.stderr
        expected = "positive integer" if agent else "requires agent mode"
        assert expected in result.stderr, result.stderr
    results = []
    with tempfile.TemporaryDirectory(prefix="thunderhill-budget-") as directory:
        for limit in (1, 12, 13):
            with worker(
                args.godot,
                Path(directory) / str(limit),
                90,
                (f"--agent-max-episode-ticks={limit}",),
            ) as (client, data):
                results.append(verify(client, data, limit))
    print(json.dumps({"ok": True, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
