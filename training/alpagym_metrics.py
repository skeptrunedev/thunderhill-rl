"""Simulator metric units for NVIDIA's metric reward dispatcher.

Progress is bounded route completion, never raw metres. The full circuit is our
reference route; NVIDIA instead projects onto a recorded scene trajectory.
Motorcycle falls are an explicit extension, not mislabeled obstacle collisions.
"""

import math

REWARD_VERSION = "thunderhill-normalized-progress-safety-v2"
REWARD_SCALES = {
    "progress": 1.0,
    "collision_any": -10.0,
    "offroad": -5.0,
    "fall_without_collision": -10.0,
}


def validate_reward_terms(terms):
    actual = {}
    for term in terms:
        if term.kind != "metric" or term.metric_name in actual:
            raise ValueError("Expected unique simulator metric reward terms")
        actual[term.metric_name] = term.scale
    if actual != REWARD_SCALES:
        raise ValueError(
            "Reward contract changed; prepare a fresh normalized reward run"
        )


class EpisodeMetrics:
    def __init__(self, *, track_length_m, start_legal_distance_m):
        if not math.isfinite(track_length_m) or track_length_m <= 0:
            raise ValueError("Track length must be finite and positive")
        if not math.isfinite(start_legal_distance_m):
            raise ValueError("Initial legal distance must be finite")
        self.track_length_m = track_length_m
        self.start = start_legal_distance_m
        self.distance = 0.0
        self.collision = False
        self.offroad = False
        self.crashed = False

    def observe(self, transition):
        """Consume every physics tick, so a short excursion cannot be missed."""
        track, state = transition["track"], transition["state"]
        distance = float(track["legal_distance"]) - self.start
        if not math.isfinite(distance):
            raise ValueError("Nonfinite legal progress")
        if type(track["on_track"]) is not bool or type(state["crashed"]) is not bool:
            raise ValueError("Expected boolean simulator event flags")
        if not isinstance(state["collision_contact"], dict):
            raise ValueError("Expected simulator collision contact object")
        self.distance = distance
        self.collision |= bool(state["collision_contact"])
        self.offroad |= not track["on_track"]
        self.crashed |= state["crashed"]

    def values(self):
        return {
            "progress": min(1.0, max(0.0, self.distance / self.track_length_m)),
            "collision_any": float(self.collision),
            "offroad": float(self.offroad),
            "fall_without_collision": float(self.crashed and not self.collision),
        }

    def report(self):
        return {
            "reward_version": REWARD_VERSION,
            "progress_normalizer_m": self.track_length_m,
            "legal_progress_m": self.distance,
            "progress_reference": "full_track_length",
            "event_sampling": "every_executed_physics_tick",
            "reference_trajectory_penalty": False,
        }
