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

Action Box([-1, 1]^2): steer with the game's steering assist (the command sets a
target lean), and combined throttle (+) / brake (-); braking applies the same
fraction to both brakes, as the rolling-start speed hold does. Gears stay
automatic.

Reward per 0.1 s step, in units of STEP_REFERENCE_M (the distance covered in one
step at the v6 reference speed of 20 m/s), reusing the reward v6 accounting in
alpagym_metrics.EpisodeMetrics tick by tick:
  r = sum_ticks legal_progress_m * max(0, 1 - |lateral| / half_width) / 2 m
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
from lap_policy import RoadTelemetry  # noqa: E402

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


def observation_vector(track: Track, observation: dict, previous_action) -> np.ndarray:
    state, road = observation["state"], observation["track"]
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
            state["lean"],
            state["lean_rate"] / 2.0,
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


def controls(action) -> dict:
    steer, pedal = (float(np.clip(value, -1.0, 1.0)) for value in action)
    brake = max(0.0, -pedal)
    return dict(steer=steer, throttle=max(0.0, pedal), front_brake=brake, rear_brake=brake)


class ThunderhillSACEnv(gym.Env):
    """A worker that fails (crash, disconnect, invalid response) is restarted and
    the interrupted episode is truncated with info["worker_restart"] = True."""

    metadata = {"render_modes": []}

    def __init__(self, *, godot: str, data_dir: str, horizon_s: float = 60.0,
                 record_godot: bool = False, starts: list[dict] | None = None,
                 seed: int = 0, policy_id: str = "sac"):
        self.godot, self.data_dir = godot, Path(data_dir)
        self.horizon_s, self.record_godot = horizon_s, record_godot
        self.fixed_starts, self.policy_id = starts, policy_id
        self.track = Track(RoadTelemetry())
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
        return result

    # Episodes ---------------------------------------------------------------
    def _choose_start(self, options):
        if options and "start" in options:
            return options["start"]
        if self.fixed_starts:
            return self.fixed_starts[self._episodes % len(self.fixed_starts)]
        station = float(self.rng.uniform(0.0, self.track.length))
        index = int(np.searchsorted(self.track._s, station, side="right") - 1)
        limit = float(self.track.speed_limits[min(index, len(self.track.speed_limits) - 1)])
        speed = max(MIN_START_SPEED_M_S, limit * self.rng.uniform(*START_SPEED_FRACTION))
        return dict(station=round(station, 2), speed=round(min(speed, MAX_INITIAL_SPEED_M_S), 2))

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
        self._start = start
        self._vector = observation_vector(self.track, observation, self._previous_action)
        return self._vector, {"episode_id": observation["episode_id"], "start": start}

    def step(self, action):
        command = controls(action)
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
        before_centered, before_incidents = metrics.centered_distance, len(metrics.incidents)
        for transition in transitions:
            metrics.observe(transition)
            for event in transition["events"]:
                if event.get("type") == "lap":
                    self._laps += 1
                    self._lap_time = float(event["time"])
        reward = (metrics.centered_distance - before_centered) / STEP_REFERENCE_M
        self._observation = result
        self._step_count += 1
        self._previous_action = (command["steer"], float(np.clip(action[1], -1.0, 1.0)))
        self._vector = observation_vector(self.track, result, self._previous_action)
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
        }}
