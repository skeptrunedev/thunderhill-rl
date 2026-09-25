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
MAX_LEAN = 0.88  # godot/scripts/motorcycle.gd rider_max_lean_rad
WHEELBASE = 1.496
MAX_STEERING = 0.50
LOW_SPEED_ASSIST_THRESHOLD = 4.0

# Receding horizon tracking mirrors NVIDIA AlpaSim's controller
# (alpasim_controller/mpc_controller.py and mpc_impl/linear_mpc.py): 20 steps
# of 0.1 s, reference sampled at plan timestamps, tracking cost only from
# horizon index 10, longitudinal position weight 2 and control weight 1, and a
# 0.1 s first order acceleration actuator lag.
HORIZON_STEPS = 20
TRACKING_START_STEP = 10
LONG_POSITION_WEIGHT = 2.0
ACCELERATION_WEIGHT = 1.0
ACCELERATION_TIME_CONSTANT = 0.1
LAT_POSITION_WEIGHT = 1.0
HEADING_WEIGHT = 1.0
LEAN_WEIGHT = 1.0
MIN_ACCELERATION = -8.0  # AlpaSim braking limit; also within Godot tire grip
DISTURBANCE_GAIN = 0.5  # per control interval, observer on measured speed
MAX_DISTURBANCE = 4.0
CURVATURE_SPAN_M = 1.0  # half span of the plan heading difference
INTEGRATION_SUBSTEPS = 10
# Plan segments shorter than this carry no usable direction (model noise at rest).
DIRECTION_EPSILON_M = 0.01
MIN_STEERING_PATH_M = 0.5

# Longitudinal plant, copied from godot/scripts/motorcycle.gd parameters and
# _integrate_step. Grade, tire force limits and lateral grip reservation are
# not modeled here; the measured-speed disturbance observer absorbs them.
PHYSICS_DT = 1.0 / 120.0  # godot/project.godot physics_ticks_per_second
MASS_KG = 189.0 + 80.0 + 8.0
GEAR_RATIOS = (38 / 14, 36 / 17, 33 / 19, 32 / 21, 30 / 22, 30 / 24)
PRIMARY_RATIO = 1.8
FINAL_RATIO = 42 / 15
REAR_RADIUS_M = 0.32
DRIVETRAIN_EFFICIENCY = 0.92
PEAK_TORQUE_NM = 119.7
PEAK_POWER_W = 150700.0
IDLE_RPM = 1400.0
LAUNCH_CLUTCH_RPM = 4500.0
REV_LIMIT_RPM = 13500.0
ENGINE_BRAKE_TORQUE_NM = 12.0
DRAG_FORCE_PER_SPEED_SQUARED = 0.5 * 1.225 * 0.46
ROLLING_FORCE_N = 0.015 * MASS_KG * GRAVITY
FRONT_BRAKE_N = 4400.0
REAR_BRAKE_N = 1800.0
REAR_BRAKE_SHARE = 0.18  # rear command as a fraction of front command
THROTTLE_RATE = 3.0  # command units per second
BRAKE_RATE = 5.0
# Assisted roll response, godot/scripts/motorcycle.gd _update_steering: the
# rider assist realises lean'' = gain (target - lean) - damping lean' and picks
# the steering angle whose lateral acceleration g tan(lean) - h lean'' does so.
ROLL_GAIN = 14.0
ROLL_DAMPING = 7.0
CG_HEIGHT_M = 0.62

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


def _engine_torque_nm(rpm: float) -> float:
    fraction = max(0.0, min(1.0, rpm / 11500.0))
    torque = PEAK_TORQUE_NM * (0.55 + 0.45 * math.sin(fraction * math.pi * 0.5))
    return min(torque, PEAK_POWER_W / max(rpm * math.tau / 60.0, 1.0))


def _plant_acceleration(speed: float, gear: int, throttle: float, brake: float) -> float:
    """Level-road longitudinal acceleration of the Godot motorcycle."""
    ratio = PRIMARY_RATIO * GEAR_RATIOS[gear - 1] * FINAL_RATIO
    rpm = max(IDLE_RPM, speed / REAR_RADIUS_M * 60.0 / math.tau * ratio,
              IDLE_RPM + (LAUNCH_CLUTCH_RPM - IDLE_RPM) * throttle)
    drive = (0.0 if rpm >= REV_LIMIT_RPM else
             _engine_torque_nm(rpm) * throttle * ratio * DRIVETRAIN_EFFICIENCY / REAR_RADIUS_M)
    engine_brake = (ENGINE_BRAKE_TORQUE_NM * (1.0 - throttle) * ratio / REAR_RADIUS_M
                    * max(0.0, min(1.0, speed / 5.0)))
    brakes = brake * (FRONT_BRAKE_N + REAR_BRAKE_SHARE * REAR_BRAKE_N)
    if speed <= 0.0:
        # At rest brakes and engine braking are reactions, never reverse drive.
        return max(0.0, drive - brakes) / MASS_KG
    resistance = DRAG_FORCE_PER_SPEED_SQUARED * speed * speed + ROLLING_FORCE_N * min(speed, 1.0)
    return (drive - engine_brake - brakes - resistance) / MASS_KG


def _actuation(acceleration: float, speed: float, gear: int) -> tuple[float, float]:
    """Invert the plant: (throttle, front brake) giving the acceleration."""
    coast = _plant_acceleration(speed, gear, 0.0, 0.0)
    if acceleration <= coast:
        force = (coast - acceleration) * MASS_KG
        return 0.0, min(1.0, force / (FRONT_BRAKE_N + REAR_BRAKE_SHARE * REAR_BRAKE_N))
    low, high = 0.0, 1.0
    if _plant_acceleration(speed, gear, high, 0.0) <= acceleration:
        return high, 0.0
    for _ in range(40):  # acceleration is monotone in throttle
        middle = 0.5 * (low + high)
        if _plant_acceleration(speed, gear, middle, 0.0) < acceleration:
            low = middle
        else:
            high = middle
    return high, 0.0


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination for the small symmetric positive definite system."""
    n = len(vector)
    rows = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(rows[r][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(column + 1, n):
            factor = rows[r][column] / rows[column][column]
            for c in range(column, n + 1):
                rows[r][c] -= factor * rows[column][c]
    solution = [0.0] * n
    for r in range(n - 1, -1, -1):
        solution[r] = (rows[r][n] - sum(rows[r][c] * solution[c] for c in range(r + 1, n))) / rows[r][r]
    return solution


class TrajectoryTracker:
    """Execute the policy's timestamped world waypoints with pose feedback.

    World axes are AlpaSim X/Y horizontal and Z up. Never interpret a banked
    rider frame as a level plane. This is a motorcycle actuator adapter, not a
    source of training demonstrations.

    Longitudinal control is a receding horizon tracker of the plan's position
    at each future timestamp, structured like AlpaSim's linear MPC: the
    reference is sampled over a 2 s horizon, the vehicle lagging or leading the
    plan is corrected by position cost, and the plan's own acceleration is the
    feedforward. Deviations from AlpaSim, all motorcycle specific: position is
    measured as signed forward progress along the plan (Frenet longitudinal)
    because lateral motion is controlled separately; the control cost penalizes
    departure from the plan's acceleration instead of acceleration magnitude, so
    a planned acceleration is executed rather than resisted; the commanded
    acceleration is mapped to throttle or brake through the Godot drivetrain
    model with an observer for grade and unmodeled forces.

    Lateral control above the assist threshold is the same horizon least
    squares with AlpaSim's lateral and heading weights, but the decision
    variable is target lean and the model is Godot's assisted roll response,
    because the rider assist, not the tracker, turns lean into steering. The
    plan's curvature is the lean feedforward. Below the threshold Godot steers
    the wheel directly and pure pursuit on the plan's path is used.

    Actuator slew state starts released, as after the neutral sensor warmup;
    create a tracker per episode.

    The bike is forward only. Backward plan segments add no progress, so a
    genuinely reversing plan becomes a stop, while momentary backward noise in
    an otherwise forward plan only pauses the reference briefly.
    """

    def __init__(self):
        self.points = None
        self.step = 0
        # Actuator and disturbance state belongs to the bike, not the plan.
        self.throttle_applied = 0.0
        self.brake_applied = 0.0
        self.acceleration_state = 0.0
        self.disturbance = 0.0
        self.predicted_speed = None

    def replan(self, xyz, *, origin, forward):
        points = [*_xyz([origin]), *_xyz(xyz)]
        forward = _xyz([forward])[0]
        horizontal = math.hypot(forward[0], forward[1])
        if horizontal < 1e-6:
            raise ValueError("Measured heading cannot be vertical")
        # Direction of travel starts as the measured heading and then follows
        # the plan, so curves are progress while reversing is not.
        direction = (forward[0] / horizontal, forward[1] / horizontal)
        progress, path, reverse = [0.0], [(0.0, points[0])], 0.0
        for start, end in zip(points, points[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.dist(start, end)
            if dx * direction[0] + dy * direction[1] < 0:
                reverse += length
                progress.append(progress[-1])
                continue
            progress.append(progress[-1] + length)
            horizontal = math.hypot(dx, dy)
            if horizontal >= DIRECTION_EPSILON_M:
                direction = (dx / horizontal, dy / horizontal)
                path.append((progress[-1], end))
        self.points, self.progress, self.path = points, progress, path
        self.reverse_distance = reverse
        # Only the geometric frame resets. Actuator state belongs to the bike.
        self.step = 0

    def _progress_at(self, time_s: float) -> float:
        index = max(0.0, min(len(self.progress) - 1.0, time_s / DT))
        low = min(int(index), len(self.progress) - 2)
        fraction = index - low
        return self.progress[low] + fraction * (self.progress[low + 1] - self.progress[low])

    def _plan_acceleration(self, time_s: float) -> float:
        # Second difference centred on the policy waypoint starting this interval.
        index = max(2, min(len(self.progress) - 2, int(time_s / DT + 1e-9) + 1))
        if len(self.progress) < 4:
            return 0.0
        return (self.progress[index + 1] - 2 * self.progress[index] + self.progress[index - 1]) / (DT * DT)

    def _position_at(self, time_s: float):
        index = max(0.0, min(len(self.points) - 1.0, time_s / DT))
        low = min(int(index), len(self.points) - 2)
        fraction = index - low
        return tuple(a + fraction * (b - a) for a, b in zip(self.points[low], self.points[low + 1]))

    def _path_point(self, distance: float):
        path = self.path
        if len(path) == 1 or distance <= path[0][0]:
            return path[0][1]
        for (s0, p0), (s1, p1) in zip(path, path[1:]):
            if distance <= s1:
                fraction = (distance - s0) / max(s1 - s0, 1e-9)
                return tuple(a + fraction * (b - a) for a, b in zip(p0, p1))
        return path[-1][1]

    # Lateral geometry uses the policy's waypoints only. The plan origin is the
    # measured position, so the segment from it encodes tracking error, not the
    # intended path; the first policy segment is extended backwards instead.

    def _path_heading(self, distance: float) -> float:
        path = self.path[1:]
        for (s0, p0), (s1, p1) in zip(path, path[1:]):
            if distance <= s1:
                break
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def _path_errors(self, position, heading: float) -> tuple[float, float]:
        """(left lateral offset, heading error) of the bike from the policy path."""
        path = self.path[1:]
        best = None
        for index, ((s0, p0), (s1, p1)) in enumerate(zip(path, path[1:])):
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.hypot(dx, dy)
            along = ((position[0] - p0[0]) * dx + (position[1] - p0[1]) * dy) / length
            clamped = min(length, along) if index == 0 else max(0.0, min(length, along))
            lateral = (-dy * (position[0] - p0[0]) + dx * (position[1] - p0[1])) / length
            distance = math.hypot(along - clamped, lateral)
            if best is None or distance < best[0]:
                best = (distance, lateral, math.atan2(dy, dx))
        return best[1], math.remainder(heading - best[2], math.tau)

    def _path_curvature(self, distance: float) -> float:
        """Left curvature from path heading change, spanning model waypoint noise."""
        start, end = self.path[1][0], self.path[-1][0]
        low, high = max(start, distance - CURVATURE_SPAN_M), min(end, distance + CURVATURE_SPAN_M)
        if high - low < CURVATURE_SPAN_M:
            return 0.0
        turn = math.remainder(self._path_heading(high) - self._path_heading(low), math.tau)
        return turn / (high - low)

    def _horizon_lean(self, speed, lateral_error, heading_error, lean, lean_rate, curvatures):
        """Target left lean from least squares over the horizon, like AlpaSim's QP.

        Linear Frenet model about the plan path: e' = v psi,
        psi' = (g lean - h lean'') / v - v curvature, and the assisted roll response.
        """
        dt = DT / INTEGRATION_SUBSTEPS

        def rollout(inputs, e, psi, phi, phi_rate, path_curvatures):
            errors = [(e, psi)]
            for u, kappa in zip(inputs, path_curvatures):
                for _ in range(INTEGRATION_SUBSTEPS):
                    phi_acceleration = ROLL_GAIN * (u - phi) - ROLL_DAMPING * phi_rate
                    lateral_acceleration = GRAVITY * phi - CG_HEIGHT_M * phi_acceleration
                    e += dt * speed * psi
                    psi += dt * (lateral_acceleration / speed - speed * kappa)
                    phi_rate += dt * phi_acceleration
                    phi += dt * phi_rate
                errors.append((e, psi))
            return errors

        n = HORIZON_STEPS
        zeros = [0.0] * n
        free = rollout(zeros, lateral_error, heading_error, lean, lean_rate, curvatures)
        responses = [rollout([1.0 if i == j else 0.0 for i in range(n)], 0.0, 0.0, 0.0, 0.0, zeros)
                     for j in range(n)]
        feedforward = [math.atan(speed * speed * kappa / GRAVITY) for kappa in curvatures]
        tracked = range(TRACKING_START_STEP, n + 1)
        weights = (LAT_POSITION_WEIGHT, HEADING_WEIGHT)
        hessian = [[sum(w * responses[i][k][c] * responses[j][k][c] for k in tracked for c, w in enumerate(weights))
                    + (LEAN_WEIGHT if i == j else 0.0) for j in range(n)] for i in range(n)]
        gradient = [LEAN_WEIGHT * feedforward[i]
                    - sum(w * responses[i][k][c] * free[k][c] for k in tracked for c, w in enumerate(weights))
                    for i in range(n)]
        return max(-MAX_LEAN, min(MAX_LEAN, _solve(hessian, gradient)[0]))

    def _project(self, position, time_s: float) -> float:
        """Forward progress of the measured position on the plan's path."""
        if len(self.path) == 1:
            return 0.0
        best = None
        for (s0, p0), (s1, p1) in zip(self.path, self.path[1:]):
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length_squared = dx * dx + dy * dy
            fraction = max(0.0, min(1.0, ((position[0] - p0[0]) * dx + (position[1] - p0[1]) * dy)
                                    / length_squared))
            # Horizontal distance; plan time breaks ties on nearly stationary plans.
            offset = math.hypot(p0[0] + fraction * dx - position[0], p0[1] + fraction * dy - position[1])
            progress = s0 + fraction * (s1 - s0)
            key = (round(offset, 3), abs(progress - self._progress_at(time_s)))
            if best is None or key < best[0]:
                best = (key, progress)
        # Travel past the end of the path still counts as progress along it.
        (s0, p0), (s1, p1) = self.path[-2], self.path[-1]
        span = max(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), 1e-9)
        beyond = ((position[0] - p1[0]) * (p1[0] - p0[0]) + (position[1] - p1[1]) * (p1[1] - p0[1])) / span
        return best[1] if best[1] < s1 or beyond <= 0 else s1 + beyond

    def _horizon_acceleration(self, speed: float, reference: list[float],
                              plan_acceleration: list[float]) -> float:
        """Condensed least squares over the horizon, as AlpaSim's QP without bounds."""
        tau = ACCELERATION_TIME_CONSTANT
        decay = math.exp(-DT / tau)
        # Exact discretization of s' = v, v' = a + disturbance, a' = (u - a) / tau.
        a_a, a_u = decay, 1.0 - decay
        v_a, v_u = tau * (1.0 - decay), DT - tau * (1.0 - decay)
        s_a, s_u = tau * DT - tau * tau * (1.0 - decay), DT * DT / 2 - tau * DT + tau * tau * (1.0 - decay)

        def rollout(inputs, s, v, a, disturbance):
            positions = [s]
            for u in inputs:
                s, v, a = (s + DT * v + DT * DT / 2 * disturbance + s_a * a + s_u * u,
                           v + DT * disturbance + v_a * a + v_u * u,
                           a_a * a + a_u * u)
                positions.append(s)
            return positions

        n = HORIZON_STEPS
        free = rollout([0.0] * n, 0.0, speed, self.acceleration_state, self.disturbance)
        responses = []  # response[k] = position effect of a unit input at step j
        for j in range(n):
            responses.append(rollout([1.0 if i == j else 0.0 for i in range(n)], 0.0, 0.0, 0.0, 0.0))
        tracked = range(TRACKING_START_STEP, n + 1)
        hessian = [[LONG_POSITION_WEIGHT * sum(responses[i][k] * responses[j][k] for k in tracked)
                    + (ACCELERATION_WEIGHT if i == j else 0.0) for j in range(n)] for i in range(n)]
        gradient = [LONG_POSITION_WEIGHT * sum(responses[i][k] * (reference[k] - free[k]) for k in tracked)
                    + ACCELERATION_WEIGHT * plan_acceleration[i] for i in range(n)]
        return _solve(hessian, gradient)[0]

    def next_controls(self, speed: float, *, position, forward, gear: int, lean: float, lean_rate: float):
        """Controls from measured speed, level pose, gear and Godot lean (right positive)."""
        speed, lean, lean_rate = _finite(speed), _finite(lean), _finite(lean_rate)
        position, forward = _xyz([position, forward])
        if speed < 0:
            raise ValueError("Expected nonnegative speed")
        if type(gear) is not int or not 1 <= gear <= len(GEAR_RATIOS):
            raise ValueError("Expected the measured gear")
        horizontal = math.hypot(forward[0], forward[1])
        if horizontal < 1e-6:
            raise ValueError("Measured heading cannot be vertical")
        fx, fy = forward[0] / horizontal, forward[1] / horizontal
        if self.points is None or self.step >= len(self.points) - 1:
            raise ValueError("A fresh unexhausted trajectory is required")
        now = self.step * DT

        # Disturbance observer: measured speed against the plant prediction for
        # the previous interval. A held stationary bike has reaction forces the
        # plant does not predict, so rest does not update the estimate.
        if self.predicted_speed is not None and max(speed, self.predicted_speed) > 0.05:
            self.disturbance += DISTURBANCE_GAIN * (speed - self.predicted_speed) / DT
            self.disturbance = max(-MAX_DISTURBANCE, min(MAX_DISTURBANCE, self.disturbance))

        current = self._project(position, now)
        times = [now + k * DT for k in range(HORIZON_STEPS + 2)]
        progress = [self._progress_at(t) for t in times]
        reference = [value - current for value in progress[:HORIZON_STEPS + 1]]
        speeds = [(b - a) / DT for a, b in zip(progress, progress[1:])]
        # Feedforward uses only the model's own waypoints. The origin is the
        # measured position, so its segment carries tracking error, not intent.
        plan_acceleration = [self._plan_acceleration(t) for t in times[:HORIZON_STEPS]]
        command = self._horizon_acceleration(speed, reference, plan_acceleration)
        maximum = _plant_acceleration(speed, gear, 1.0, 0.0) + self.disturbance
        command = max(MIN_ACCELERATION, min(maximum, command))
        plant_target = command - self.disturbance
        if reference[-1] <= 0.0:
            # The plan ends at or behind the bike: never drive towards it. Only
            # a plan without forward progress ahead is a stop; momentary backward
            # waypoints elsewhere in a forward plan are not.
            plant_target = min(plant_target, _plant_acceleration(speed, gear, 0.0, 0.0))
        throttle, front_brake = _actuation(plant_target, speed, gear)

        # Predict the next measured speed under Godot's actuator slew limits.
        predicted = speed
        for _ in range(round(DT / PHYSICS_DT)):
            self.throttle_applied += max(-THROTTLE_RATE * PHYSICS_DT,
                                         min(THROTTLE_RATE * PHYSICS_DT, throttle - self.throttle_applied))
            self.brake_applied += max(-BRAKE_RATE * PHYSICS_DT,
                                      min(BRAKE_RATE * PHYSICS_DT, front_brake - self.brake_applied))
            predicted = max(0.0, predicted + PHYSICS_DT * (
                _plant_acceleration(predicted, gear, self.throttle_applied, self.brake_applied)
                + self.disturbance))
        self.predicted_speed = predicted
        decay = math.exp(-DT / ACCELERATION_TIME_CONSTANT)
        self.acceleration_state = decay * self.acceleration_state + (1.0 - decay) * command

        # Lateral. Above the assist threshold the rider assist owns roll, so a
        # horizon tracker chooses target lean against lateral and heading error
        # on the plan's path (AlpaSim weights), with plan curvature feedforward.
        # Below it Godot steers the wheel directly; pure pursuit suffices there.
        lookahead = max(2.0, speed * 0.7)
        remaining = self.path[-1][0] - current
        target = self._path_point(current + lookahead)
        if len(self.path) < 3 or remaining < MIN_STEERING_PATH_M:
            path_lateral_error = path_heading_error = 0.0
        else:
            path_lateral_error, path_heading_error = self._path_errors(position, math.atan2(fy, fx))
        if len(self.path) < 3 or remaining < MIN_STEERING_PATH_M:
            # No forward path: stay upright and straight while stopping.
            steering_mode = "wheel_angle" if speed < LOW_SPEED_ASSIST_THRESHOLD else "lean"
            curvature, requested_lean, requested_command = 0.0, 0.0, 0.0
        elif speed < LOW_SPEED_ASSIST_THRESHOLD:
            steering_mode = "wheel_angle"
            dx, dy = target[0] - position[0], target[1] - position[1]
            lateral = -fy * dx + fx * dy
            curvature = 2.0 * lateral / max(dx * dx + dy * dy, 1.0)
            requested_lean = -math.atan(speed * speed * curvature / GRAVITY)
            # Model left is negative Godot steer.
            requested_command = -math.atan(WHEELBASE * curvature) / MAX_STEERING
        else:
            steering_mode = "lean"
            curvatures = [self._path_curvature(current + speed * (k + 0.5) * DT)
                          for k in range(HORIZON_STEPS)]
            curvature = curvatures[0]
            # Model left lean is negative Godot lean and negative Godot steer.
            left_lean = self._horizon_lean(speed, path_lateral_error, path_heading_error,
                                           -lean, -lean_rate, curvatures)
            requested_lean = -left_lean
            requested_command = requested_lean / MAX_LEAN
        steer = max(-1.0, min(1.0, requested_command))

        # Residual against the timestamped plan at this instant. This exposes
        # actuator tracking failures separately from bad policy plans.
        start = self._position_at(now)
        reference_dx, reference_dy = start[0] - position[0], start[1] - position[1]
        controls = {"steer": steer, "throttle": throttle,
                    "front_brake": front_brake, "rear_brake": front_brake * REAR_BRAKE_SHARE,
                    "assist_enabled": True, "auto_shift": True}
        diagnostics = {"controller": "model_trajectory_horizon_mpc_v5",
                       "speed_error_m_s": speeds[0] - speed,
                       "steering_mode": steering_mode,
                       "steer_saturated": abs(requested_command) > 1.0,
                       "tracking_position_error_m": math.dist(start, position),
                       "tracking_longitudinal_error_m": fx * reference_dx + fy * reference_dy,
                       "tracking_lateral_error_m": -fy * reference_dx + fx * reference_dy,
                       "tracking_vertical_error_m": start[2] - position[2],
                       "tracking_progress_lag_m": reference[0],
                       "path_lateral_error_m": path_lateral_error,
                       "path_heading_error_rad": path_heading_error,
                       "horizon_s": HORIZON_STEPS * DT,
                       "horizon_reference_progress_m": reference[-1],
                       "horizon_reference_speed_m_s": speeds[HORIZON_STEPS],
                       "plan_acceleration_m_s2": plan_acceleration[0],
                       "commanded_acceleration_m_s2": command,
                       "max_acceleration_m_s2": maximum,
                       "disturbance_acceleration_m_s2": self.disturbance,
                       "predicted_next_speed_m_s": predicted,
                       "plan_forward_progress_m": self.path[-1][0],
                       "plan_reverse_distance_m": self.reverse_distance,
                       "gear": gear,
                       "privileged_track_inputs": False,
                       "feedback": "measured_world_pose",
                       "plan_step": self.step, "plan_time_s": now,
                       "target_speed_m_s": speeds[0], "target_lean_rad": requested_lean,
                       "curvature_left_m_inv": curvature,
                       "target_xyz": list(target)}
        self.step += 1
        return controls, diagnostics
