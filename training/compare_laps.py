"""Independently audit two complete laps and their learned checkpoint lineage."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from lap_audit import audit_lap

CONDITIONS = (
    "physics_dt",
    "physics_version",
    "parameters",
    "track_sha256",
    "terrain_sha256",
    "surface_sha256",
    "obstacle_collision",
    "initial_state",
    "start_station",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def audit_evaluation(directory, adapter):
    directory = Path(directory)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    adapter_hash = digest(Path(adapter) / "adapter_model.safetensors")
    require(
        summary.get("teacher_used_at_inference") is False,
        "Teacher inference not excluded",
    )
    require(
        summary.get("adapter_sha256") == adapter_hash, "Checkpoint fingerprint mismatch"
    )
    decisions_path = directory / "decisions.jsonl"
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    final = summary["final_observation"]
    headers = []
    for filename in summary["recordings"]:
        with Path(filename).open() as stream:
            header = json.loads(next(stream))
        if header.get("episode_id") == final["episode_id"]:
            headers.append(header)
    require(len(headers) == 1, "Expected exactly one episode header")
    header = headers[0]
    require(all(key in header for key in CONDITIONS), "Missing simulator conditions")
    for key in ("track_sha256", "terrain_sha256", "surface_sha256"):
        value = header[key]
        require(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            f"Invalid {key}",
        )
    require(positive(header["physics_dt"]), "Invalid physics timestep")
    require(
        header["initial_state"].get("speed") == 0
        and header["initial_state"].get("elapsed") == 0,
        "Lap must begin from standing start",
    )
    require(not header.get("snapshot"), "Snapshot start is not a fresh lap")
    audit = audit_lap(
        summary["recordings"],
        episode_id=final["episode_id"],
        policy_id="interactive-step-lap-eval-" + adapter_hash[:12],
        track_sha256=header["track_sha256"],
        final_observation=final,
        decisions=decisions,
    )
    require(
        all(summary.get(key) == value for key, value in audit.items()),
        "Summary disagrees with independent recording audit",
    )
    require(audit["success"] is True, "A complete legal lap is required")
    seconds = audit["recorded_transitions"] * header["physics_dt"]
    for value in (summary["sim_seconds"], final["sim_time"], final["state"]["elapsed"]):
        require(
            positive(value) and math.isclose(value, seconds, rel_tol=0, abs_tol=1e-6),
            "Simulator time disagrees with recorded ticks",
        )
    return {
        "lap_seconds": seconds,
        "model": summary["model"],
        "adapter_sha256": adapter_hash,
        "summary_sha256": digest(summary_path),
        "decisions_sha256": digest(decisions_path),
        "recording_sha256": digest(audit["recordings"][0]),
        "evaluation": str(directory.resolve()),
        "audit": audit,
    }, {key: header[key] for key in CONDITIONS}


def compare(baseline, candidate, baseline_adapter, candidate_adapter, training_summary):
    before, conditions = audit_evaluation(baseline, baseline_adapter)
    after, candidate_conditions = audit_evaluation(candidate, candidate_adapter)
    require(
        isinstance(before["model"], str)
        and bool(before["model"])
        and before["model"] == after["model"],
        "Model identity changed",
    )
    require(
        conditions == candidate_conditions,
        "Simulator conditions or starting state changed",
    )
    require(
        before["adapter_sha256"] != after["adapter_sha256"], "Checkpoint must change"
    )
    require(bool(training_summary), "Training lineage is required")
    current = before["adapter_sha256"]
    lineage = []
    for path in training_summary:
        trained = json.loads(Path(path).read_text())
        require(
            all(
                trained.get(key) is True
                for key in (
                    "ok",
                    "reloaded_logits_match",
                    "all_recording_audits_passed",
                )
            ),
            "Training verification failed",
        )
        require(
            trained.get("model") == before["model"], "Training model identity changed"
        )
        steps = trained.get("optimizer_steps")
        require(
            type(steps) is int
            and steps > 0
            and positive(trained.get("max_adapter_delta")),
            "No finite learning update",
        )
        require(
            trained.get("initial_adapter_sha256") == current,
            "Broken training checkpoint lineage",
        )
        next_hash = trained.get("current_adapter_sha256")
        require(
            isinstance(next_hash, str)
            and len(next_hash) == 64
            and all(c in "0123456789abcdef" for c in next_hash)
            and next_hash != current,
            "Invalid learned checkpoint fingerprint",
        )
        lineage.append(
            {
                "summary": str(Path(path).resolve()),
                "summary_sha256": digest(path),
                "initial_adapter_sha256": current,
                "current_adapter_sha256": next_hash,
                "optimizer_steps": steps,
            }
        )
        current = next_hash
    require(
        current == after["adapter_sha256"],
        "Candidate does not match trained checkpoint",
    )
    delta = before["lap_seconds"] - after["lap_seconds"]
    return {
        "improved": delta > 0,
        "baseline": before,
        "candidate": after,
        "seconds_saved": delta,
        "percent_faster": 100 * delta / before["lap_seconds"],
        "identical_simulator_conditions": conditions,
        "training_lineage": lineage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "baseline",
        "candidate",
        "baseline-adapter",
        "candidate-adapter",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, action="append", required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = compare(**args)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["improved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
