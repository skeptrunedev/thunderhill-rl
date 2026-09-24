"""Simulator metric units for NVIDIA's metric reward dispatcher.

Progress is bounded route completion, never raw metres. The full circuit is our
reference route; NVIDIA instead projects onto a recorded scene trajectory.
Motorcycle falls are an explicit extension, not mislabeled obstacle collisions.
"""

import math

REWARD_VERSION = "thunderhill-normalized-progress-safe-lap-speed-v3"
REWARD_SCALES = {
    "progress": 1.0,
    "collision_any": -10.0,
    "offroad": -5.0,
    "fall_without_collision": -10.0,
    "completed_lap_speed": 1.0,
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

    def values(self, *, elapsed_seconds, episode_seconds, lap_completed):
        if not math.isfinite(episode_seconds) or episode_seconds <= 0:
            raise ValueError("Episode budget must be finite and positive")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError("Executed episode duration must be finite and positive")
        if type(lap_completed) is not bool:
            raise ValueError("Expected simulator lap completion flag")
        # Game gates accept only forward displacements below five metres per
        # physics tick. Allow that terminal crossing discretization, never a
        # partial circuit marked complete by an inconsistent caller.
        full_distance = self.distance >= self.track_length_m - 5.0
        safe_completion = (
            lap_completed
            and full_distance
            and not (self.collision or self.offroad or self.crashed)
        )
        # A terminal bonus avoids rewarding fast failures or subtracting time
        # costs that a crash can evade. Partial progress keeps its native scale.
        speed_bonus = (
            max(0.0, 1.0 - elapsed_seconds / episode_seconds)
            if safe_completion
            else 0.0
        )
        self.speed_report = {
            "completed_lap_speed": speed_bonus,
            "completed_lap_speed_formula": "max(0, 1 - elapsed_seconds / episode_seconds) for a safe completed lap, else 0",
            "episode_budget_seconds": episode_seconds,
            "elapsed_seconds": elapsed_seconds,
            "safe_completion": safe_completion,
            "completion_distance_tolerance_m": 5.0,
            "full_lap_distance_verified": full_distance,
        }
        return {
            "progress": min(1.0, max(0.0, self.distance / self.track_length_m)),
            "collision_any": float(self.collision),
            "offroad": float(self.offroad),
            "fall_without_collision": float(self.crashed and not self.collision),
            "completed_lap_speed": speed_bonus,
        }

    def report(self):
        return {
            **self.speed_report,
            "reward_version": REWARD_VERSION,
            "progress_normalizer_m": self.track_length_m,
            "legal_progress_m": self.distance,
            "progress_reference": "full_track_length",
            "event_sampling": "every_executed_physics_tick",
            "reference_trajectory_penalty": False,
        }
