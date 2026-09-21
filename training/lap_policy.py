"""Privileged road telemetry and direct bike action codec for the tiny lap pilot.

Road geometry is simulator supplied guidance, not camera perception. The teacher
is imported only by offline dataset generation, never by policy observation code.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
import re
import sys
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
        return (
            "Ride the track safely. Simulator road guidance: speed m/s, lean rad, "
            "angle rad to lookahead center, distance m to it, lateral m, signed "
            "curves 1/m at 0,30,60,90,120m. Reply only control_bike "
            "STEER_MILLI THROTTLE_PERCENT FRONT_PERCENT REAR_PERCENT. "
            "Steer -1000..1000; pedals 0..100.\n"
            + json.dumps(self.features(observation), separators=(",", ":"))
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


def build_dataset(
    driver_path: Path, episode_path: Path, output: Path, augment=0, max_speed=8.0
):
    road = RoadTelemetry()
    with episode_path.open() as source:
        header = json.loads(next(source))
    if header.get("type") != "episode" or header["track_sha256"] != road.track_sha256:
        raise ValueError("Episode header or track fingerprint mismatch")
    with driver_path.open() as source:
        rows = [json.loads(line) for line in source]
    if not rows:
        raise ValueError("Empty teacher recording")
    observation = {
        "state": header["initial_state"],
        "tick": header["initial_state"]["tick"],
        "episode_id": header["episode_id"],
        "track": {"progress": header["start_station"] / road.length, "lateral_m": 0.0},
    }
    teacher = None
    if augment:
        sys.path.insert(0, str(ROOT / "tools"))
        from drive_lap import Driver

        teacher = Driver(max_speed)
    output.mkdir(parents=True, exist_ok=False)
    counts = {"train": 0, "eval": 0, "excluded": 0}
    # One contiguous central holdout, with ten decisions excluded on each edge.
    lo, hi = int(len(rows) * 0.45), int(len(rows) * 0.55)
    with (
        (output / "train.jsonl").open("w") as train,
        (output / "eval.jsonl").open("w") as evaluation,
    ):
        for index, row in enumerate(rows):
            req = row["request"]
            if (
                req["episode_id"] != header["episode_id"]
                or req["expected_tick"] != observation["tick"]
            ):
                raise ValueError(f"Preaction observation mismatch at row {index}")
            split = "eval" if lo <= index < hi else "train"
            if lo - 10 <= index < lo or hi <= index < hi + 10:
                counts["excluded"] += 1
                observation = row["observation"]
                continue
            examples = [(observation, req["controls"], False)]
            if teacher and split == "train":
                for direction in (-1, 1)[:augment]:
                    varied = copy.deepcopy(observation)
                    state = varied["state"]
                    station = varied["track"]["progress"] * road.length
                    a, b = road.point(station - 0.5), road.point(station + 0.5)
                    dx, dz = b[0] - a[0], b[2] - a[2]
                    norm = math.hypot(dx, dz)
                    # Positive lateral points left, matching track projection.
                    state["position"][0] += direction * 0.5 * dz / norm
                    state["position"][2] -= direction * 0.5 * dx / norm
                    state["heading"] += direction * 0.025
                    state["speed"] = max(0, state["speed"] + direction * 0.5)
                    varied["track"]["lateral_m"] += direction * 0.5
                    controls, _ = teacher.controls(varied)
                    examples.append((varied, controls, True))
            for before, controls, augmented in examples:
                record = {
                    "prompt": road.prompt(before),
                    "completion": encode_action(controls),
                    "source_row": index,
                    "split": split,
                    "augmented": augmented,
                }
                (train if split == "train" else evaluation).write(
                    json.dumps(record) + "\n"
                )
                counts[split] += 1
            observation = row["observation"]
    manifest = {
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "privileged_geometry": True,
        "teacher_at_inference": False,
        "driver_sha256": sha256(driver_path),
        "episode_sha256": sha256(episode_path),
        "driver_path": str(driver_path.resolve()),
        "episode_path": str(episode_path.resolve()),
        "track_sha256": road.track_sha256,
        "counts": counts,
        "augmentation_per_training_row": augment,
        "augmentation_teacher_max_speed": max_speed,
        "holdout_source_rows": [lo, hi],
        "boundary_gap_rows": 10,
        "limitation": "Single episode contiguous holdout is not an independent closed loop evaluation.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--augment", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--max-speed", type=float, default=8.0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.driver, args.episode, args.output, args.augment, args.max_speed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
