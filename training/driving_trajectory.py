"""Execute the policy's own trajectory, without reading track geometry.

AlpaSim rig coordinates are X forward, Y left, Z up (NVlabs/alpasim,
CONTRIBUTING.md, Coordinate Systems). Godot uses Y up and heading zero
towards -Z; positive heading, lean and assisted steer turn right.
The history conversion below deliberately represents a level, yaw-only rig,
not the leaned rider camera. It preserves measured elevation changes.
"""
from __future__ import annotations

import math
import json
from collections.abc import Sequence

DT = 0.1
GRAVITY = 9.81
SPEED_KP = 0.25  # throttle fraction per m/s speed error
SPEED_KI = 0.20  # throttle fraction per metre of accumulated speed error
MAX_LEAN = 0.88  # godot/scripts/motorcycle.gd rider_max_lean_rad

CONTROL_FIELDS = ('steer', 'throttle', 'front_brake', 'rear_brake')


def _controller_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {*CONTROL_FIELDS, 'shift'}:
        raise ValueError('Expected exactly four continuous controls and shift')
    for name in CONTROL_FIELDS:
        value = arguments[name]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)
                or not (-1 if name == 'steer' else 0) <= value <= 1):
            raise ValueError(f'Invalid continuous control: {name}')
    if type(arguments['shift']) is not int or arguments['shift'] != 0:
        raise ValueError('Trajectory controller uses automatic shifting')
    return {**{name: float(arguments[name]) for name in CONTROL_FIELDS}, 'shift': 0}


def encode_controller_controls(controls):
    """Preserve continuous controller outputs; these are not sampled tool tokens."""
    arguments = _controller_arguments({**{name: controls[name] for name in CONTROL_FIELDS},
                                       'shift': controls.get('shift', 0)})
    return json.dumps({'tool': 'control_bike', 'arguments': arguments},
                      separators=(',', ':'), allow_nan=False)


def decode_controller_controls(text):
    payload = json.loads(text)
    if (not isinstance(payload, dict) or set(payload) != {'tool', 'arguments'}
            or payload['tool'] != 'control_bike'):
        raise ValueError('Expected a control_bike controller receipt')
    return _controller_arguments(payload['arguments'])


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Trajectory and state values must be finite")
    return value


def _xyz(points: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    rows = []
    for point in points:
        if len(point) != 3:
            raise ValueError("Each position must have three coordinates")
        rows.append(tuple(_finite(value) for value in point))
    if not rows:
        raise ValueError("At least one position is required")
    return rows


def godot_history_to_ego(positions, headings):
    """Return (XYZ translations, 3x3 rotations) relative to newest pose.

    Inputs are chronological Godot positions in metres and headings in radians.
    Caller owns sampling at 10Hz, actual history collection and model padding.
    Rotations map each historical level rig's axes into the newest rig frame.
    This does not synthesize demonstration trajectories or future positions.
    """
    points = _xyz(positions)
    angles = [_finite(value) for value in headings]
    if len(angles) != len(points):
        raise ValueError("Positions and headings must have equal lengths")
    origin = points[-1]
    heading = angles[-1]
    sin_h, cos_h = math.sin(heading), math.cos(heading)
    translations, rotations = [], []
    for point, angle in zip(points, angles):
        dx, dy, dz = (a - b for a, b in zip(point, origin))
        translations.append([dx * sin_h - dz * cos_h,
                             -dx * cos_h - dz * sin_h, dy])
        yaw = heading - angle  # positive model yaw is left
        c, s = math.cos(yaw), math.sin(yaw)
        rotations.append([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return translations, rotations


class TrajectoryTracker:
    """Fixed 10Hz follower for model waypoints at t=0.1,0.2,... seconds.

    Call next_controls once per actual simulator advance. Replan resets the
    origin to the bike's current ego pose. Between replans, position and yaw
    are dead reckoned from speed and lean using a flat-road bicycle estimate.
    This approximation is intentionally explicit: it is not track following
    and cannot correct banking or tire slip from these two inputs alone.
    No plan, exhausted plans and malformed data raise instead of driving on.
    """

    def __init__(self):
        self.points = None
        self.step = 0
        self.x = self.y = self.yaw = 0.0
        self.speed_integral = 0.0

    def replan(self, xyz):
        points = _xyz(xyz)
        self.points = [(0.0, 0.0, 0.0), *points]
        # Only the geometric frame resets. Speed feedback belongs to the bike.
        self.step = 0
        self.x = self.y = self.yaw = 0.0

    def next_controls(self, speed: float, lean: float):
        speed, lean = _finite(speed), _finite(lean)
        if speed < 0 or abs(lean) >= math.pi / 2:
            raise ValueError("Expected nonnegative speed and upright bike lean")
        if self.points is None or self.step >= len(self.points) - 1:
            raise ValueError("A fresh unexhausted trajectory is required")
        # Observe motion since the preceding call, before selecting this action.
        if self.step:
            yaw_delta = -GRAVITY * math.tan(lean) / max(speed, 1.0) * DT
            mid_yaw = self.yaw + yaw_delta / 2
            self.x += speed * math.cos(mid_yaw) * DT
            self.y += speed * math.sin(mid_yaw) * DT
            self.yaw += yaw_delta
        start, end = self.points[self.step:self.step + 2]
        target_speed = math.dist(start[:2], end[:2]) / DT
        # Reverse plans cannot be executed by this forward-only motorcycle.
        if end[0] < start[0]:
            target_speed = 0.0
        lookahead = max(2.0, speed * 0.7)
        target = self.points[-1]
        for candidate in self.points[self.step + 1:]:
            target = candidate
            if math.hypot(candidate[0] - self.x, candidate[1] - self.y) >= lookahead:
                break
        dx, dy = target[0] - self.x, target[1] - self.y
        lateral = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        curvature = 2.0 * lateral / max(dx * dx + dy * dy, 1.0)
        # Model left is negative Godot steer. Assisted steer requests lean.
        requested_lean = -math.atan(speed * speed * curvature / GRAVITY)
        steer = max(-1.0, min(1.0, requested_lean / MAX_LEAN))
        error = target_speed - speed
        # A constant velocity requires nonzero engine torque to balance engine
        # braking, rolling resistance and drag. Integral feedback learns that
        # balance from measured speed alone and persists across trajectory replans.
        # Conditional integration prevents saturation from winding up the state.
        if target_speed == 0.0:
            self.speed_integral = 0.0
            effort = -SPEED_KP * speed
        else:
            candidate = self.speed_integral + SPEED_KI * error * DT
            candidate_effort = SPEED_KP * error + candidate
            if (-1.0 <= candidate_effort <= 1.0
                    or (candidate_effort > 1.0 and error < 0)
                    or (candidate_effort < -1.0 and error > 0)):
                self.speed_integral = max(-1.0, min(1.0, candidate))
            effort = SPEED_KP * error + self.speed_integral
        throttle = max(0.0, min(1.0, effort))
        front_brake = max(0.0, min(1.0, -effort * 0.6))
        controls = {"steer": steer, "throttle": throttle,
                    "front_brake": front_brake, "rear_brake": front_brake * 0.18,
                    "assist_enabled": True, "auto_shift": True}
        diagnostics = {"controller": "model_trajectory_pure_pursuit_pi_v2",
                       "speed_error_m_s": error,
                       "speed_integral_throttle": self.speed_integral,
                       "privileged_track_inputs": False,
                       "plan_step": self.step, "plan_time_s": self.step * DT,
                       "target_speed_m_s": target_speed, "target_lean_rad": requested_lean,
                       "curvature_left_m_inv": curvature,
                       "target_xyz": list(target)}
        self.step += 1
        return controls, diagnostics
