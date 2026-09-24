"""Read authoritative simulator recordings without counting metadata as physics."""

import json


def recording_rows(source):
    """Read episode contents without silently swallowing faults or unknown records."""
    for line in source:
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("type") not in {
            "transition",
            "model_decision",
            "snapshot",
            "camera_observation",
        }:
            raise ValueError("Unexpected simulator recording row")
        yield row


def recorded_transitions(source):
    """Presentation and observation metadata never count as physics transitions."""
    for row in recording_rows(source):
        if row["type"] == "transition":
            yield row
