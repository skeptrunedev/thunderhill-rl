"""Explicit simulator initial conditions shared by native launch and runtime.

A scene ID fully determines where and how fast an episode starts. NVIDIA
AlpaGym treats each scene ID as one Cosmos prompt and runs its n_generation
sibling rollouts from that scene, so every GRPO group shares one initial
condition while different groups start at different places on the circuit.

The warmup regulator below sets up that initial condition in the real
simulator. It is neither a policy nor a demonstration: its controls are never
given to the model as actions or labels, and warmup progress is excluded from
reward. The model only observes the resulting measured ego history, which is
steady motion at the scene speed rather than a coast-down.
"""

from __future__ import annotations

import math
import random
import re

TRACK = "thunderhill-east"
MAX_INITIAL_SPEED_M_S = 40.0
WARMUP_SECONDS = 1.5
# Conservative envelope for choosing a start speed from road curvature. The
# simulator's asphalt friction is 1.1; an untrained policy starts well inside it.
START_LATERAL_ACCELERATION_M_S2 = 5.0
START_BRAKING_M_S2 = 4.0
# The warmup rides straight with zero steer, so a start must have a straight
# enough centerline over the whole warmup distance plus a short margin.
START_STRAIGHT_TOLERANCE_M = 0.5
START_MARGIN_M = 10.0
_ROLLING = re.compile(
    r"thunderhill-east(?:-station-(?P<station>\d+\.\d+)m)?-rolling-(?P<speed>\d+\.\d+)mps"
)


def validate_initial_speed(initial_speed_m_s: float) -> float:
    if (
        type(initial_speed_m_s) not in (int, float)
        or not math.isfinite(initial_speed_m_s)
        or not 0 <= initial_speed_m_s <= MAX_INITIAL_SPEED_M_S
    ):
        raise ValueError(
            f"Initial speed must be finite and in [0, {MAX_INITIAL_SPEED_M_S:g}] m/s"
        )
    return float(initial_speed_m_s)


def scene_id(initial_speed_m_s: float = 0.0, station_m: float | None = None) -> str:
    """Start-line scenes keep their historical names; other stations are explicit."""
    speed = validate_initial_speed(initial_speed_m_s)
    if station_m is None:
        if speed == 0:
            return TRACK + "-standing"
        return TRACK + "-rolling-" + str(speed) + "mps"
    if type(station_m) not in (int, float) or not math.isfinite(station_m) or station_m < 0:
        raise ValueError("Start station must be finite and nonnegative")
    if speed == 0:
        raise ValueError("Track position starts require a rolling speed")
    return f"{TRACK}-station-{float(station_m)}m-rolling-{speed}mps"


def parse_scene_id(scene: str) -> dict:
    """Return the exact initial condition a scene ID names."""
    if scene == TRACK + "-standing":
        return dict(start_station_m=0.0, initial_speed_m_s=0.0)
    match = _ROLLING.fullmatch(scene)
    if not match:
        raise ValueError(f"Unknown Thunderhill scene: {scene!r}")
    station = float(match["station"]) if match["station"] else None
    speed = float(match["speed"])
    if scene_id(speed, station) != scene:
        raise ValueError(f"Noncanonical Thunderhill scene: {scene!r}")
    return dict(start_station_m=station or 0.0, initial_speed_m_s=speed)


def _speed_limits(road, max_speed):
    """Curvature speed limit, then a backward braking pass around the loop."""
    samples = road.samples
    limits = [
        min(max_speed, math.sqrt(START_LATERAL_ACCELERATION_M_S2 / abs(row["curvature"])))
        if row["curvature"]
        else max_speed
        for row in samples
    ]
    count = len(samples)
    for _ in range(2):  # the second lap propagates braking across the start line
        for index in range(count - 1, -1, -1):
            following = (index + 1) % count
            step = (samples[following]["s"] if following else road.length) - samples[index]["s"]
            limits[index] = min(
                limits[index],
                math.sqrt(limits[following] ** 2 + 2 * START_BRAKING_M_S2 * step),
            )
    return limits


def _straight_deviation(road, index, distance):
    """Largest centerline offset from the start heading ray over a distance."""
    samples = road.samples
    count = len(samples)
    origin = samples[index]["p"]
    ahead = samples[(index + 1) % count]["p"]
    # Same heading basis as main.gd reset_episode: the next sample's direction.
    dx, dz = ahead[0] - origin[0], ahead[2] - origin[2]
    norm = math.hypot(dx, dz)
    dx, dz = dx / norm, dz / norm
    worst = 0.0
    travelled = 0.0
    cursor = index
    while travelled <= distance:
        following = (cursor + 1) % count
        step = (samples[following]["s"] if following else road.length) - samples[cursor]["s"]
        travelled += step
        cursor = following
        point = samples[cursor]["p"]
        ox, oz = point[0] - origin[0], point[2] - origin[2]
        worst = max(worst, abs(ox * dz - oz * dx))
    return worst


def start_candidates(road, max_speed: float) -> list[dict]:
    """Every track sample where a straight, speed-held warmup is on the road."""
    max_speed = validate_initial_speed(max_speed)
    if max_speed == 0:
        raise ValueError("Track position starts require a rolling speed")
    limits = _speed_limits(road, max_speed)
    count = len(road.samples)
    candidates = []
    for index, row in enumerate(road.samples):
        warmup = max_speed * WARMUP_SECONDS + START_MARGIN_M
        # Hold a speed that is safe everywhere along the warmup and at handoff.
        speed = limits[index]
        travelled, cursor = 0.0, index
        while travelled <= warmup:
            following = (cursor + 1) % count
            travelled += (
                road.samples[following]["s"] if following else road.length
            ) - road.samples[cursor]["s"]
            cursor = following
            speed = min(speed, limits[cursor])
        speed = math.floor(speed * 10) / 10
        if speed <= 0:
            continue
        deviation = _straight_deviation(road, index, speed * WARMUP_SECONDS + START_MARGIN_M)
        if deviation <= START_STRAIGHT_TOLERANCE_M:
            candidates.append(
                dict(
                    start_station_m=float(row["s"]),
                    initial_speed_m_s=speed,
                    warmup_straight_deviation_m=round(deviation, 4),
                )
            )
    return candidates


def randomized_starts(road, *, count: int, seed: int, max_speed: float) -> list[dict]:
    """Seeded, spread-out track positions; reproducible for the same track file."""
    if type(count) is not int or count < 1:
        raise ValueError("Randomized start count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("Randomized start seed must be a nonnegative integer")
    candidates = start_candidates(road, max_speed)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} straight start stations are available")
    # Stratify over the ordered eligible stations with one random phase, so the
    # chosen starts cover the circuit instead of clustering on one straight.
    offset = random.Random(seed).random()
    chosen = [candidates[int((k + offset) * len(candidates) / count)] for k in range(count)]
    return [
        dict(start, scene_id=scene_id(start["initial_speed_m_s"], start["start_station_m"]))
        for start in chosen
    ]


def game_scene_ids(game: dict, road=None) -> list[str]:
    """Scene IDs implied by a game configuration's explicit initial conditions."""
    speed = validate_initial_speed(game.get("initial_speed_m_s", 0.0))
    spread = game.get("randomized_start")
    if spread is None:
        return [scene_id(speed)]
    if set(spread) != {"count", "seed", "track_sha256"}:
        raise ValueError("Randomized start requires count, seed and track_sha256")
    if road is None:
        from training.lap_policy import RoadTelemetry

        road = RoadTelemetry()
    if road.track_sha256 != spread["track_sha256"]:
        raise ValueError("Randomized starts were chosen for a different track file")
    return [
        start["scene_id"]
        for start in randomized_starts(
            road, count=spread["count"], seed=spread["seed"], max_speed=speed
        )
    ]


class WarmupSpeedHold:
    """Throttle/brake regulator holding measured speed during sensor warmup.

    Simulator initial-condition setup only. It never steers, is recorded with
    action_source "speed_hold_sensor_warmup", and precedes the reward baseline.
    """

    KP = 0.35  # throttle per m/s of speed error
    KI = 0.8  # throttle per m/s of error per second
    # Starting integrator value: roughly the throttle that balances engine
    # braking, drag and rolling resistance in the simulator at rolling speeds.
    CRUISE_THROTTLE = 0.15

    def __init__(self, target_m_s: float, period_s: float = 0.1):
        self.target = validate_initial_speed(target_m_s)
        if self.target == 0:
            raise ValueError("A standing warmup holds the brakes; no speed regulation")
        self.period = period_s
        self.integral = self.CRUISE_THROTTLE

    def controls(self, speed_m_s: float) -> tuple[dict, dict]:
        error = self.target - speed_m_s
        command = self.integral + self.KP * error
        # Conditional integration: do not wind up while the actuator saturates.
        if -1 < command < 1 or (command >= 1) != (error > 0):
            self.integral = min(1.0, max(-1.0, self.integral + self.KI * error * self.period))
        command = min(1.0, max(-1.0, self.integral + self.KP * error))
        brake = max(0.0, -command)
        controls = dict(
            throttle=max(0.0, command), steer=0.0, front_brake=brake, rear_brake=brake
        )
        return controls, dict(
            kind="warmup_speed_hold", target_m_s=self.target, measured_m_s=speed_m_s,
            error_m_s=error, integral=self.integral,
        )
