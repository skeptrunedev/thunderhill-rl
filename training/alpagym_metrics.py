"""Simulator metric units for NVIDIA's metric reward dispatcher.

Every reward term shares one unit: metres divided by the reference distance
REFERENCE_SPEED_M_S * horizon_seconds, the distance a rider averaging the
reference speed covers in the fixed episode horizon. Progress is therefore a
progress rate over a fixed horizon, not a fraction of the full circuit, and
incident costs are directly comparable to the progress they trade against.

Time not executed because the episode ended early (stall, crash, track limits)
earns zero progress. A safe completed lap is credited at its own mean lap speed
for the whole horizon, so a faster lap is always worth more. Incidents cost the
rider's kinetic energy expressed as metres of gentle braking (Fuchs et al.,
GT Sport: fixed wall penalties taught agents to brake and stand still, kinetic
penalties did not). Motorcycle falls are an explicit extension, not mislabeled
obstacle collisions.
"""

import math

REWARD_VERSION = "thunderhill-horizon-progress-rate-kinetic-incidents-v4"
# Unit only: GRPO normalizes each sibling group, so this sets readable magnitudes
# (1.0 = averaging 20 m/s for the whole horizon, a 230 s lap) and no trade-off.
REFERENCE_SPEED_M_S = 20.0
# Incident cost = v^2 / (2 a): the distance a 0.25 g brake needs to remove the
# impact speed. Braking to a stop instead of crashing is roughly distance
# neutral; the crash additionally forfeits the rest of the horizon.
INCIDENT_DECELERATION_M_S2 = 0.25 * 9.80665
REWARD_SCALES = {
    "progress": 1.0,
    "collision_cost": -1.0,
    "offroad_cost": -1.0,
    "fall_cost": -1.0,
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


def incident_cost_m(speed_m_s):
    return speed_m_s * speed_m_s / (2.0 * INCIDENT_DECELERATION_M_S2)


class EpisodeMetrics:
    def __init__(self, *, track_length_m, start_legal_distance_m, horizon_seconds,
                 start_speed_m_s):
        if not math.isfinite(track_length_m) or track_length_m <= 0:
            raise ValueError("Track length must be finite and positive")
        if not math.isfinite(start_legal_distance_m):
            raise ValueError("Initial legal distance must be finite")
        if not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
            raise ValueError("Episode horizon must be finite and positive")
        if not math.isfinite(start_speed_m_s) or start_speed_m_s < 0:
            raise ValueError("Initial speed must be finite and nonnegative")
        self.track_length_m = track_length_m
        self.start = start_legal_distance_m
        self.horizon_seconds = horizon_seconds
        self.distance = 0.0
        self.collision = False
        self.offroad = False
        self.crashed = False
        self.incidents = []
        self._previous_speed = start_speed_m_s
        self._in_contact = False
        self._on_track = True

    def _incident(self, kind, transition, speed):
        self.incidents.append({
            "kind": kind,
            "tick": transition.get("tick"),
            "speed_m_s": speed,
            "cost_m": incident_cost_m(speed),
        })

    def observe(self, transition):
        """Consume every physics tick, so a short excursion cannot be missed.

        Each incident is charged once at its onset. Obstacle contact zeroes the
        simulator speed on the contact tick, so the charged speed is the larger
        of this and the previous tick's speed.
        """
        track, state = transition["track"], transition["state"]
        distance = float(track["legal_distance"]) - self.start
        speed = float(state["speed"])
        if not math.isfinite(distance) or not math.isfinite(speed) or speed < 0:
            raise ValueError("Nonfinite legal progress or speed")
        if type(track["on_track"]) is not bool or type(state["crashed"]) is not bool:
            raise ValueError("Expected boolean simulator event flags")
        if not isinstance(state["collision_contact"], dict):
            raise ValueError("Expected simulator collision contact object")
        contact = bool(state["collision_contact"])
        impact_speed = max(self._previous_speed, speed)
        if contact and not self._in_contact:
            self._incident("collision", transition, impact_speed)
        if not track["on_track"] and self._on_track:
            self._incident("offroad", transition, impact_speed)
        if state["crashed"] and not self.crashed and not contact:
            self._incident("fall", transition, impact_speed)
        self.distance = distance
        self.collision |= contact
        self.offroad |= not track["on_track"]
        self.crashed |= state["crashed"]
        self._in_contact = contact
        self._on_track = track["on_track"]
        self._previous_speed = speed

    def cost_m(self, kind):
        return sum(row["cost_m"] for row in self.incidents if row["kind"] == kind)

    def values(self, *, elapsed_seconds, lap_completed):
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError("Executed episode duration must be finite and positive")
        if elapsed_seconds > self.horizon_seconds + 1e-6:
            raise ValueError("Executed episode exceeds its fixed reward horizon")
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
        # Early stops forfeit the unexecuted horizon (zero progress); a safe lap
        # keeps its own pace for the remainder, so lap time orders completions.
        credited_m = (
            self.track_length_m * self.horizon_seconds / elapsed_seconds
            if safe_completion
            else min(self.track_length_m, max(-self.track_length_m, self.distance))
        )
        reference_m = REFERENCE_SPEED_M_S * self.horizon_seconds
        self.summary = {
            "elapsed_seconds": elapsed_seconds,
            "horizon_seconds": self.horizon_seconds,
            "unexecuted_horizon_seconds": self.horizon_seconds - elapsed_seconds,
            "reference_distance_m": reference_m,
            "credited_progress_m": credited_m,
            "horizon_progress_rate_m_s": credited_m / self.horizon_seconds,
            "safe_completion": safe_completion,
            "completion_distance_tolerance_m": 5.0,
            "full_lap_distance_verified": full_distance,
            "incidents": list(self.incidents),
        }
        return {
            "progress": credited_m / reference_m,
            "collision_cost": self.cost_m("collision") / reference_m,
            "offroad_cost": self.cost_m("offroad") / reference_m,
            "fall_cost": self.cost_m("fall") / reference_m,
            # Event flags keep NVIDIA's metric names for charts; not reward terms.
            "collision_any": float(self.collision),
            "offroad": float(self.offroad),
            "fall_without_collision": float(self.crashed and not self.collision),
        }

    def report(self):
        return {
            **self.summary,
            "reward_version": REWARD_VERSION,
            "reward_scales": dict(REWARD_SCALES),
            "reward_unit": "metres / (reference_speed_m_s * horizon_seconds)",
            "reference_speed_m_s": REFERENCE_SPEED_M_S,
            "progress_formula": "credited_progress_m / reference_distance_m; unexecuted horizon earns 0; a safe lap is credited track_length * horizon / lap_time",
            "incident_cost_formula": "speed_m_s^2 / (2 * incident_deceleration_m_s2) at each incident onset",
            "incident_deceleration_m_s2": INCIDENT_DECELERATION_M_S2,
            "legal_progress_m": self.distance,
            "track_length_m": self.track_length_m,
            "event_sampling": "every_executed_physics_tick",
            "reference_trajectory_penalty": False,
        }
