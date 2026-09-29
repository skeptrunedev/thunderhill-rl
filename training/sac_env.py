"""Gymnasium environment: one headless Godot worker, 10 Hz low-dimensional control.

Follows Fuchs et al., "Super-Human Performance in Gran Turismo Sport Using Deep
Reinforcement Learning" (arXiv 2008.07971) and the Assetto Corsa Gym benchmark
(arXiv 2407.16680): SAC on a compact telemetry vector with track-edge course
points ahead of the vehicle, a dense progress reward, a speed-scaled cost on
leaving the track, and rolling starts spread around the whole lap.

Observation (float32, OBSERVATION_SIZE = 64), all roughly unit scale:
  0      speed / 40 m/s
  1      longitudinal acceleration / 10 m/s^2
  2      lateral acceleration / 10 m/s^2
  3      lean / 1 rad
  4      lean rate / 2 rad/s
  5      front steering angle / 0.5 rad (the steering limit)
  6-7    sin, cos of heading error against the centerline tangent
  8      lateral offset / half track width (positive = left of centerline)
  9      half track width / 10 m
  10     gear / 6
  11-12  previous action (steer, throttle-brake)
  13-52  left and right edge points at COURSE_DISTANCES_M ahead along the
         centerline, in the bike frame (forward, left) / 50 m: for each distance
         (left_fwd, left_left, right_fwd, right_left)
  53-62  centerline curvature at COURSE_DISTANCES_M ahead * 50 m
  63     on track flag

vehicle="car" (godot/scripts/car.gd, --vehicle=car) keeps this layout and
reuses the two lean slots for the car's own stability state, since body roll
is a small, nearly redundant function of lateral acceleration:
  3      sideslip angle at the centre of gravity / 1 rad
  4      yaw rate / 2 rad/s
Slot 5 is then the front road wheel angle / 0.5 rad (the car's limit is 0.42).

Action Box([-1, 1]^2): steer with the game's steering assist (the command sets a
target lean), and combined throttle (+) / brake (-); braking applies the same
fraction to both brakes, as the rolling-start speed hold does. Gears stay
automatic. For the car, steer sets the front road wheel angle as a fraction of
its steering limit (rate limited, no assist) and the brake pedal is split by
the car's brake balance.

Reward per 0.1 s step, in units of STEP_REFERENCE_M (the distance covered in one
step at the v6 reference speed of 20 m/s), reusing the reward v6 accounting in
alpagym_metrics.EpisodeMetrics tick by tick:
  r = sum_ticks legal_progress_m / 2 m                                  (reward_line="progress")
  r = sum_ticks legal_progress_m * max(0, 1 - |lateral| / half_width) / 2 m  ("centered")
The default "progress" pays centerline progress however wide the line, so a
racing line that uses the track width to straighten corners earns more per
second; "centered" (reward v6) weights credit toward the centerline, which kept
the policy on the centerline instead of hitting apexes.
      - incident_cost_m(v) / 2 m   on the step of the first incident
where incident_cost_m(v) = v^2 / (2 * 0.48 g) is the off-track run-off distance.
The first incident (offroad, obstacle contact, fall) terminates the episode. A
stall (less than STALL_MIN_PROGRESS_M of legal progress over STALL_WINDOW_S,
which also catches riding the wrong way) terminates with the v6 stall cost. The
horizon and a completed lap truncate (the task itself continues, so SAC
bootstraps through them).
"""

from __future__ import annotations

import math
import secrets
import sys
from contextlib import ExitStack
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "training", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpagym_metrics import (  # noqa: E402
    REFERENCE_SPEED_M_S,
    STALL_REFERENCE_SPEED_M_S,
    EpisodeMetrics,
    incident_cost_m,
)
from check_parallel import worker  # noqa: E402
from episode_config import MAX_INITIAL_SPEED_M_S, _speed_limits  # noqa: E402
from lap_policy import DEFAULT_TRACK, RoadTelemetry  # noqa: E402
from apex_report import (  # noqa: E402  the specialists' judge, for the opt-in shaping
    APEX_FRACTION,
    UPRIGHT_LEAN_RAD,
    WINDOW_M as APEX_WINDOW_M,
    required_corners,
)

CONTROL_PERIOD_S = 0.1  # the game advances 12 physics ticks per request
STEP_REFERENCE_M = REFERENCE_SPEED_M_S * CONTROL_PERIOD_S
COURSE_DISTANCES_M = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 125.0, 150.0)
OBSERVATION_SIZE = 13 + 4 * len(COURSE_DISTANCES_M) + len(COURSE_DISTANCES_M) + 1
STALL_WINDOW_S = 3.0
STALL_MIN_PROGRESS_M = 3.0
# Rolling starts: the curvature/braking-safe speed at the station, scaled by a
# random factor so the policy also sees slower entries, never below the speed
# where the steering assist engages.
START_SPEED_FRACTION = (0.5, 1.0)
MIN_START_SPEED_M_S = 6.0


class Track:
    """Centerline, edges and curvature interpolated by station (vectorized)."""

    def __init__(self, road: RoadTelemetry):
        samples = road.samples
        self.length = road.length
        s = np.array([row["s"] for row in samples] + [road.length])
        wrap = samples + samples[:1]
        self._s = s
        self._x = np.array([row["p"][0] for row in wrap])
        self._z = np.array([row["p"][2] for row in wrap])
        self._lx = np.array([row["left"][0] for row in wrap])
        self._lz = np.array([row["left"][2] for row in wrap])
        self._tx = np.array([row["tangent"][0] for row in wrap])
        self._tz = np.array([row["tangent"][2] for row in wrap])
        self._w = np.array([row["width"] for row in wrap])
        # Curvature is piecewise constant per sample in the game's road query.
        self._curvature = np.array([row["curvature"] for row in samples])
        self.speed_limits = np.array(_speed_limits(road, MAX_INITIAL_SPEED_M_S))

    def at(self, stations):
        s = np.mod(stations, self.length)
        interp = lambda values: np.interp(s, self._s, values)  # noqa: E731
        index = np.searchsorted(self._s, s, side="right") - 1
        return dict(
            x=interp(self._x), z=interp(self._z), lx=interp(self._lx), lz=interp(self._lz),
            tx=interp(self._tx), tz=interp(self._tz), half_width=0.5 * interp(self._w),
            curvature=self._curvature[np.minimum(index, len(self._curvature) - 1)],
        )


def evaluation_starts(count: int, track: Track | None = None) -> list[dict]:
    """Fixed rolling starts spread evenly around the lap at 80% of the safe speed:
    the deterministic evaluation every trainer and driver shares."""
    track = track or Track(RoadTelemetry())
    starts = []
    for k in range(count):
        station = k * track.length / count
        index = int(np.searchsorted(track._s, station, side="right") - 1)
        speed = float(np.clip(0.8 * track.speed_limits[index], MIN_START_SPEED_M_S,
                              MAX_INITIAL_SPEED_M_S))
        starts.append(dict(station=round(station, 2), speed=round(speed, 2)))
    return starts


VEHICLES = ("motorcycle", "car")


def observation_vector(track: Track, observation: dict, previous_action,
                       vehicle: str = "motorcycle") -> np.ndarray:
    state, road = observation["state"], observation["track"]
    if vehicle == "car":
        roll, roll_rate = state["sideslip_rad"], state["yaw_rate"]
    else:
        roll, roll_rate = state["lean"], state["lean_rate"]
    x, _, z = state["position"]
    heading = state["heading"]
    station = float(road["progress"]) * track.length
    here = track.at(np.array([station]))
    ahead = track.at(station + np.array(COURSE_DISTANCES_M))
    # Game convention: heading = atan2(forward.x, -forward.z); left = (f.z, -f.x).
    fx, fz = math.sin(heading), -math.cos(heading)
    lx, lz = fz, -fx
    track_heading = math.atan2(here["tx"][0], -here["tz"][0])
    error = heading - track_heading
    half_width = float(here["half_width"][0])
    edges = []
    for side in (1.0, -1.0):
        px = ahead["x"] + side * ahead["lx"] * ahead["half_width"] - x
        pz = ahead["z"] + side * ahead["lz"] * ahead["half_width"] - z
        edges.append((px * fx + pz * fz, px * lx + pz * lz))
    course = np.stack([edges[0][0], edges[0][1], edges[1][0], edges[1][1]], axis=1) / 50.0
    vector = np.concatenate([
        [
            state["speed"] / 40.0,
            state["longitudinal_acceleration"] / 10.0,
            state["lateral_acceleration"] / 10.0,
            roll,
            roll_rate / 2.0,
            state["steering"] / 0.5,
            math.sin(error),
            math.cos(error),
            road["lateral_m"] / half_width,
            half_width / 10.0,
            state["gear"] / 6.0,
            *previous_action,
        ],
        course.reshape(-1),
        ahead["curvature"] * 50.0,
        [float(road["on_track"])],
    ]).astype(np.float32)
    if vector.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(vector)):
        raise ValueError("Nonfinite or misshapen policy observation")
    return vector


def controls(action, pedal_gain: float = 1.0) -> dict:
    """pedal_gain > 1 reaches full throttle or brake before tanh saturates.

    SAC's tanh squashing and entropy bonus keep the policy mean away from the
    action bounds, so with gain 1 it held throttle near 0.56 on straights at
    10% grip use; with gain 1.25 a raw 0.8 is already full throttle.
    """
    steer = float(np.clip(action[0], -1.0, 1.0))
    pedal = float(np.clip(pedal_gain * action[1], -1.0, 1.0))
    brake = max(0.0, -pedal)
    return dict(steer=steer, throttle=max(0.0, pedal), front_brake=brake, rear_brake=brake)


class ThunderhillSACEnv(gym.Env):
    """A worker that fails (crash, disconnect, invalid response) is restarted and
    the interrupted episode is truncated with info["worker_restart"] = True."""

    metadata = {"render_modes": []}

    def __init__(self, *, godot: str, data_dir: str, horizon_s: float = 60.0,
                 reward_line: str = "progress", pedal_gain: float = 1.0,
                 vehicle: str = "motorcycle", track: str = DEFAULT_TRACK,
                 record_godot: bool = False, starts: list[dict] | None = None,
                 seed: int = 0, policy_id: str = "sac", apex_bonus_m: float = 0.0,
                 throttle_bonus_m: float = 0.0, apex_bonus_dense: bool = False,
                 max_start_speed_m_s: float = MAX_INITIAL_SPEED_M_S,
                 apex_bonus_stations: list[float] | None = None, apex_bonus_floor: float = 0.0):
        self.godot, self.data_dir = godot, Path(data_dir)
        self.horizon_s, self.record_godot = horizon_s, record_godot
        if reward_line not in ("progress", "centered"):
            raise ValueError(f"Unknown reward line {reward_line!r}")
        self.reward_line = reward_line
        if not 1.0 <= pedal_gain <= 2.0:
            raise ValueError("pedal_gain must be in [1, 2]")
        self.pedal_gain = pedal_gain
        if vehicle not in VEHICLES:
            raise ValueError(f"Unknown vehicle {vehicle!r}")
        self.vehicle = vehicle
        self.fixed_starts, self.policy_id = starts, policy_id
        # The worker runs --track=<id>; Track mirrors that geometry for observations.
        self.track_id = track
        road = RoadTelemetry(track=track)
        self.track = Track(road)
        # Training starts (not the evaluation's) may begin faster than the shared 40 m/s
        # cap, up to the braking-aware limit computed with this higher cap.
        self.max_start_speed = max_start_speed_m_s
        self.start_limits = (self.track.speed_limits if max_start_speed_m_s <= MAX_INITIAL_SPEED_M_S
                             else np.array(_speed_limits(road, max_start_speed_m_s)))
        # Opt-in shaping toward the specialists' bar (tools/apex_report.py), in metres
        # of progress: apex_bonus_m once per required apex as the bike leaves its
        # judging window, scaled by its best inside reach there (full at the 0.75
        # the judge requires, nothing at or outside the centreline);
        # throttle_bonus_m per step at full applied throttle while upright and off
        # the front brake (the judge's straights).
        # apex_bonus_dense pays the same total as the bike's best reach grows inside
        # the window, instead of once when it leaves it: credit at the moment.
        self.apex_bonus_m, self.throttle_bonus_m = apex_bonus_m, throttle_bonus_m
        self.apex_bonus_dense = apex_bonus_dense
        # apex_bonus_floor: reach below which an apex earns nothing. From 0 the credit
        # is nearly used up by 0.7 and the last 0.05 to the bar pays a few metres.
        self.apex_bonus_floor = apex_bonus_floor
        self._apexes = []
        if apex_bonus_m > 0:
            self._apexes = [(c["apex_station_m"], c["direction"]) for c in required_corners(track)
                            # apex_bonus_stations: only these corners pay (within 5 m)
                            if not apex_bonus_stations or any(
                                abs(c["apex_station_m"] - s) <= 5.0 for s in apex_bonus_stations)]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.tag = secrets.token_hex(3)
        self._stack, self._client, self._restarts, self._episodes = None, None, 0, 0
        self._observation, self._vector = None, None

    # Worker lifecycle -------------------------------------------------------
    def _start_worker(self):
        self.close()
        self._stack = ExitStack()
        directory = self.data_dir / f"worker-{self.tag}-{self._restarts}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        extra = () if self.record_godot else ("--agent-no-recording",)
        if self.vehicle != "motorcycle":
            extra += (f"--vehicle={self.vehicle}",)
        if self.track_id != DEFAULT_TRACK:
            extra += (f"--track={self.track_id}",)
        self._client, _ = self._stack.enter_context(
            worker(self.godot, directory, 120, extra)
        )
        self._restarts += 1

    def close(self):
        if self._stack is not None:
            self._stack.close()
        self._stack, self._client = None, None

    def _request(self, request):
        result = self._client.request(request)
        if "error" in result or not result.get("rollout_valid", False):
            raise RuntimeError(f"Invalid simulator response: {str(result)[:500]}")
        # Motorcycle telemetry predates the vehicle field.
        if result["state"].get("vehicle", "motorcycle") != self.vehicle:
            raise RuntimeError(f"Worker simulates {result['state'].get('vehicle')}, "
                               f"expected {self.vehicle}")
        return result

    # Episodes ---------------------------------------------------------------
    def _choose_start(self, options):
        if options and "start" in options:
            return options["start"]
        if self.fixed_starts:
            return self.fixed_starts[self._episodes % len(self.fixed_starts)]
        station = float(self.rng.uniform(0.0, self.track.length))
        speed = max(MIN_START_SPEED_M_S, self.start_limit(station) * self.rng.uniform(*START_SPEED_FRACTION))
        return dict(station=round(station, 2), speed=round(min(speed, self.max_start_speed), 2))

    def start_limit(self, station: float) -> float:
        """Braking-aware start speed limit at a station, capped at max_start_speed."""
        index = int(np.searchsorted(self.track._s, station, side="right") - 1)
        return float(self.start_limits[min(index, len(self.start_limits) - 1)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        start = self._choose_start(options)
        request = {"op": "reset", "policy_id": self.policy_id,
                   "station": start["station"], "initial_speed_m_s": start["speed"]}
        for attempt in range(3):
            try:
                if self._client is None:
                    self._start_worker()
                self._observation = self._request(request)
                break
            except Exception:
                if attempt == 2:
                    raise
                self._start_worker()
        observation = self._observation
        self._episodes += 1
        self._step_count = 0
        self._previous_action = (0.0, 0.0)
        self._metrics = EpisodeMetrics(
            track_length_m=self.track.length,
            start_legal_distance_m=float(observation["track"]["legal_distance"]),
            horizon_seconds=self.horizon_s,
            start_speed_m_s=float(observation["state"]["speed"]),
        )
        self._progress_window = []
        self._return, self._laps, self._lap_time = 0.0, 0, None
        self._apex_reach = [None] * len(self._apexes)  # best reach while in each window
        self._apex_bonus = self._throttle_bonus = 0.0
        self._start = start
        self._vector = observation_vector(self.track, observation, self._previous_action,
                                          self.vehicle)
        return self._vector, {"episode_id": observation["episode_id"], "start": start}

    def step(self, action):
        command = controls(action, self.pedal_gain)
        try:
            result = self._request({
                "op": "advance",
                "episode_id": self._observation["episode_id"],
                "expected_tick": self._observation["tick"],
                "action_id": str(self._step_count),
                "controls": command,
            })
            transitions = result["transitions"]
            if not transitions:
                raise RuntimeError("Advance returned no transitions")
        except Exception as error:
            episode_id = self._observation["episode_id"]
            self._start_worker()
            info = self._summary("worker_restart", episode_id)
            info.update(worker_restart=True, worker_error=str(error)[:500])
            return self._vector, 0.0, False, True, info
        metrics = self._metrics
        credited = "centered_distance" if self.reward_line == "centered" else "distance"
        before_credit, before_incidents = getattr(metrics, credited), len(metrics.incidents)
        for transition in transitions:
            metrics.observe(transition)
            for event in transition["events"]:
                if event.get("type") == "lap":
                    self._laps += 1
                    self._lap_time = float(event["time"])
        reward = (getattr(metrics, credited) - before_credit) / STEP_REFERENCE_M
        if self._apexes or self.throttle_bonus_m:
            reward += self._shaping(transitions) / STEP_REFERENCE_M
        self._observation = result
        self._step_count += 1
        self._previous_action = (command["steer"], command["throttle"] - command["front_brake"])
        self._vector = observation_vector(self.track, result, self._previous_action,
                                          self.vehicle)
        progress = sum(row["reward_components"]["legal_progress_m"] for row in transitions)
        self._progress_window = (self._progress_window + [progress])[
            -round(STALL_WINDOW_S / CONTROL_PERIOD_S):]

        reason = None
        terminated = truncated = False
        if before_incidents == 0 and metrics.incidents:
            incident = metrics.incidents[0]
            reward -= incident["cost_m"] / STEP_REFERENCE_M
            terminated, reason = True, incident["kind"]
        elif result["terminated"] and result["termination_reason"] == "lap_completed":
            truncated, reason = True, "lap_completed"
        elif result["terminated"] or result["truncated"]:
            terminated, reason = True, result["termination_reason"] or result["truncation_reason"]
        elif (len(self._progress_window) * CONTROL_PERIOD_S >= STALL_WINDOW_S
              and sum(self._progress_window) < STALL_MIN_PROGRESS_M):
            reward -= incident_cost_m(STALL_REFERENCE_SPEED_M_S) / STEP_REFERENCE_M
            terminated, reason = True, "stall"
        elif self._step_count * CONTROL_PERIOD_S >= self.horizon_s - 1e-9:
            truncated, reason = True, "horizon"
        self._return += reward
        info = {"episode_id": result["episode_id"]}
        if terminated or truncated:
            info.update(self._summary(reason, result["episode_id"]))
        return self._vector, float(reward), terminated, truncated, info

    def _apex_credit(self, reach: float) -> float:
        """Share of the apex bonus for an inside reach: linear from apex_bonus_floor
        to the judge's 0.75, so the credit concentrates where the bar is."""
        return float(np.clip((reach - self.apex_bonus_floor) / (APEX_FRACTION - self.apex_bonus_floor),
                             0.0, 1.0))

    def _shaping(self, transitions) -> float:
        """This step's apex and straight-throttle bonuses, in metres, with the
        judge's measures: reach = direction * lateral_m / half_width_m within
        APEX_WINDOW_M of the apex; upright = |lean| < 0.2 rad, front brake off."""
        bonus, length = 0.0, self.track.length
        for transition in transitions:
            road, state = transition["track"], transition["state"]
            station = float(road["progress"]) * length
            for index, (apex, direction) in enumerate(self._apexes):
                offset = (station - apex + length / 2) % length - length / 2
                if abs(offset) <= APEX_WINDOW_M:
                    reach = direction * float(road["lateral_m"]) / float(road["half_width_m"])
                    best = self._apex_reach[index]
                    self._apex_reach[index] = reach if best is None else max(best, reach)
                    if self.apex_bonus_dense:  # paid as the best reach grows, not on exit
                        gained = self.apex_bonus_m * (
                            self._apex_credit(self._apex_reach[index])
                            - self._apex_credit(best if best is not None else -1.0))
                        bonus += gained
                        self._apex_bonus += gained
                elif self._apex_reach[index] is not None:
                    if offset > 0 and not self.apex_bonus_dense:  # left the window forward
                        earned = self.apex_bonus_m * self._apex_credit(self._apex_reach[index])
                        bonus += earned
                        self._apex_bonus += earned
                    self._apex_reach[index] = None
            if (self.throttle_bonus_m and abs(float(state["lean"])) < UPRIGHT_LEAN_RAD
                    and float(transition["requested_controls"]["front_brake"]) == 0.0):
                earned = self.throttle_bonus_m * float(state["throttle_applied"]) / len(transitions)
                bonus += earned
                self._throttle_bonus += earned
        return bonus

    def _summary(self, reason, episode_id):
        metrics = self._metrics
        return {"episode_summary": {
            "episode_id": episode_id,
            "worker": self.tag,
            "termination": reason,
            "steps": self._step_count,
            "return": self._return,
            "centered_progress_m": metrics.centered_distance,
            "legal_progress_m": metrics.distance,
            "start_station_m": self._start["station"],
            "start_speed_m_s": self._start["speed"],
            "laps": self._laps,
            "lap_time_s": self._lap_time,
            "apex_bonus_m": round(self._apex_bonus, 3),
            "throttle_bonus_m": round(self._throttle_bonus, 3),
        }}
