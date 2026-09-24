"""Simulator road telemetry and strict controls used for recording and audits.

The Alpamayo policy receives camera frames and motion history separately.
Road geometry here supports the existing episode lifecycle and diagnostics.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_VERSION = "privileged-road-telemetry-v1"
ACTION_VERSION = "bike-integer-controls-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RoadTelemetry:
    def __init__(self, track_path: Path = ROOT / "godot/data/track.json"):
        self.path = Path(track_path)
        self.geometry = json.loads(self.path.read_text())
        self.samples = self.geometry["samples"]
        self.stations = [row["s"] for row in self.samples]
        self.length = self.geometry["length_m"]
        self.track_sha256 = sha256(self.path)

    def _segment(self, station):
        station %= self.length
        index = max(0, bisect.bisect_right(self.stations, station) - 1)
        following = (index + 1) % len(self.samples)
        end = self.stations[following] if following else self.length
        fraction = (station - self.stations[index]) / (end - self.stations[index])
        return index, following, fraction

    def point(self, station: float) -> list[float]:
        index, following, fraction = self._segment(station)
        return [
            a + fraction * (b - a)
            for a, b in zip(self.samples[index]["p"], self.samples[following]["p"])
        ]

    def features(self, observation: dict) -> dict:
        state = observation["state"]
        speed = float(state["speed"])
        station = float(observation["track"]["progress"]) * self.length
        target = self.point(station + 7.0 + speed * 1.3)
        dx = target[0] - state["position"][0]
        dz = target[2] - state["position"][2]
        angle = (math.atan2(dx, -dz) - state["heading"] + math.pi) % (
            2 * math.pi
        ) - math.pi
        result = {
            "speed": round(speed, 2),
            "lean": round(state["lean"], 4),
            "angle": round(angle, 4),
            "distance": round(math.hypot(dx, dz), 2),
            "lateral": round(observation["track"]["lateral_m"], 2),
            "curves": [
                round(self.samples[self._segment(station + ahead)[0]]["curvature"], 5)
                for ahead in (0, 30, 60, 90, 120)
            ],
        }
        if not all(
            math.isfinite(v)
            for v in [*result["curves"], *[result[k] for k in result if k != "curves"]]
        ):
            raise ValueError("Nonfinite policy observation")
        return result

    def prompt(self, observation: dict) -> str:
        return self.prompt_features(self.features(observation))

    def prompt_features(self, features: dict) -> str:
        """Format the same road features surfaced by the agent harness."""
        return (
            "Ride the track safely. Simulator road guidance: speed m/s, lean rad, "
            "angle rad to lookahead center, distance m to it, lateral m, signed "
            "curves 1/m at 0,30,60,90,120m. Reply only control_bike "
            "STEER_MILLI THROTTLE_PERCENT FRONT_PERCENT REAR_PERCENT. "
            "Steer -1000..1000; pedals 0..100.\n"
            + json.dumps(features, separators=(",", ":"))
            + "\nAction:\n"
        )


def parse_action(text: str) -> dict:
    match = re.fullmatch(r"\s*control_bike (-?\d+) (\d+) (\d+) (\d+)\s*", text)
    if not match:
        raise ValueError("Expected exactly control_bike and four integer controls")
    steer, throttle, front, rear = map(int, match.groups())
    if not -1000 <= steer <= 1000 or not all(
        0 <= x <= 100 for x in (throttle, front, rear)
    ):
        raise ValueError("Control outside allowed range")
    return {
        "steer": steer / 1000,
        "throttle": throttle / 100,
        "front_brake": front / 100,
        "rear_brake": rear / 100,
        "shift": 0,
    }


# Public evaluator spelling; both names execute the same strict codec.
decode_action = parse_action


def encode_action(controls: dict) -> str:
    values = [controls[k] for k in ("steer", "throttle", "front_brake", "rear_brake")]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite controls")
    if not -1 <= values[0] <= 1 or not all(0 <= x <= 1 for x in values[1:]):
        raise ValueError("Control outside allowed range")
    return "control_bike " + " ".join(
        str(round(v * scale)) for v, scale in zip(values, (1000, 100, 100, 100))
    )
