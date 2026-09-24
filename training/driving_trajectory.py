"""Execute the policy's own trajectory, without reading track geometry.

AlpaSim rig coordinates are X forward, Y left, Z up (NVlabs/alpasim,
CONTRIBUTING.md, Coordinate Systems). Godot uses Y up and heading zero
towards -Z; positive heading, lean and assisted steer turn right.
The history conversion below deliberately represents a level, yaw-only rig,
not the leaned rider camera. It preserves measured elevation changes.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence

DT = 0.1
GRAVITY = 9.81
SPEED_KP = 0.25  # throttle fraction per m/s speed error
SPEED_KI = 0.20  # throttle fraction per metre of accumulated speed error
MAX_LEAN = 0.88  # godot/scripts/motorcycle.gd rider_max_lean_rad
WHEELBASE = 1.496
MAX_STEERING = 0.50
LOW_SPEED_ASSIST_THRESHOLD = 4.0

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
    """Track native world waypoints with measured pose feedback at 10 Hz.

    World axes are AlpaSim X/Y horizontal and Z up. Never interpret a banked
    rider frame as a level plane. Steering uses horizontal world geometry;
    target speed uses full spatial displacement. This is a motorcycle actuator
    adapter, not NVIDIA's car MPC and not a source of training demonstrations.
    """

    def __init__(self):
        self.points = None
        self.step = 0
        self.speed_integral = 0.0

    def replan(self, xyz, *, origin):
        points = _xyz(xyz)
        self.points = [*_xyz([origin]), *points]
        # Only the geometric frame resets. Speed feedback belongs to the bike.
        self.step = 0

    def next_controls(self, speed: float, *, position, forward):
        speed = _finite(speed)
        position, forward = _xyz([position, forward])
        if speed < 0:
            raise ValueError("Expected nonnegative speed")
        horizontal = math.hypot(forward[0], forward[1])
        if horizontal < 1e-6:
            raise ValueError("Measured heading cannot be vertical")
        fx, fy = forward[0] / horizontal, forward[1] / horizontal
        if self.points is None or self.step >= len(self.points) - 1:
            raise ValueError("A fresh unexhausted trajectory is required")
        start, end = self.points[self.step:self.step + 2]
        target_speed = math.dist(start, end) / DT
        # Reverse plans cannot be executed by this forward-only motorcycle.
        if (end[0] - start[0]) * fx + (end[1] - start[1]) * fy < 0:
            target_speed = 0.0
        lookahead = max(2.0, speed * 0.7)
        target = self.points[-1]
        for candidate in self.points[self.step + 1:]:
            target = candidate
            if math.hypot(candidate[0] - position[0], candidate[1] - position[1]) >= lookahead:
                break
        dx, dy = target[0] - position[0], target[1] - position[1]
        lateral = -fy * dx + fx * dy
        curvature = 2.0 * lateral / max(dx * dx + dy * dy, 1.0)
        # Model left is negative Godot steer. Assisted steer requests lean.
        requested_lean = -math.atan(speed * speed * curvature / GRAVITY)
        # Godot switches actuator semantics at the low speed assist threshold:
        # below it, steer requests wheel angle, not lean. Using lean there makes
        # a stationary launch unable to steer and understeers slow trajectories.
        if speed < LOW_SPEED_ASSIST_THRESHOLD:
            steering_mode = "wheel_angle"
            requested_command = -math.atan(WHEELBASE * curvature) / MAX_STEERING
        else:
            steering_mode = "lean"
            requested_command = requested_lean / MAX_LEAN
        steer = max(-1.0, min(1.0, requested_command))
        # Residual against the timestamped plan, not the lookahead target. This
        # exposes actuator tracking failures separately from bad policy plans.
        reference_dx, reference_dy = start[0] - position[0], start[1] - position[1]
        longitudinal_error = fx * reference_dx + fy * reference_dy
        lateral_error = -fy * reference_dx + fx * reference_dy
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
        diagnostics = {"controller": "model_trajectory_world_pose_pi_v4",
                       "speed_error_m_s": error,
                       "steering_mode": steering_mode,
                       "steer_saturated": abs(requested_command) > 1.0,
                       "tracking_position_error_m": math.dist(start, position),
                       "tracking_longitudinal_error_m": longitudinal_error,
                       "tracking_lateral_error_m": lateral_error,
                       "tracking_vertical_error_m": start[2] - position[2],
                       "speed_integral_throttle": self.speed_integral,
                       "privileged_track_inputs": False,
                       "feedback": "measured_world_pose",
                       "plan_step": self.step, "plan_time_s": self.step * DT,
                       "target_speed_m_s": target_speed, "target_lean_rad": requested_lean,
                       "curvature_left_m_inv": curvature,
                       "target_xyz": list(target)}
        self.step += 1
        return controls, diagnostics
