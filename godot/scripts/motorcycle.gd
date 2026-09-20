class_name MotorcycleSim
extends RefCounted
## Explicitly stepped, reduced order motorcycle prototype. SI units throughout.
## This is original code, not validated Ducati or tire manufacturer dynamics.

const MODEL_VERSION := "reduced-order-rider-v1"
const GRAVITY := 9.81
const GEAR_RATIOS := [38.0 / 14.0, 36.0 / 17.0, 33.0 / 19.0, 32.0 / 21.0, 30.0 / 22.0, 30.0 / 24.0]

# Brochure dimensions and ratios are sourced in docs/motorcycle-reference.md.
# Other parameters are explicit engineering estimates, not measured calibration.
var parameters: Dictionary = {
	"bike_mass_kg": 189.0,
	"rider_mass_kg": 80.0,
	"fuel_mass_kg": 8.0,
	"wheelbase_m": 1.496,
	"primary_ratio": 1.8,
	"final_ratio": 42.0 / 15.0,
	"rear_rolling_radius_m": 0.32,
	"cg_height_m": 0.62,
	"static_front_fraction": 0.50,
	"drivetrain_efficiency": 0.92,
	"peak_torque_nm": 119.7,
	"peak_power_w": 150700.0,
	"idle_rpm": 1400.0,
	"rev_limit_rpm": 13500.0,
	"shift_up_rpm": 12000.0,
	"shift_down_rpm": 4800.0,
	"shift_duration_s": 0.09,
	"launch_clutch_rpm": 4500.0,
	"drag_area_m2": 0.46,
	"air_density_kg_m3": 1.225,
	"rolling_coefficient": 0.015,
	"asphalt_friction": 1.10,
	"offtrack_friction": 0.48,
	"offtrack_rolling_coefficient": 0.07,
	"front_brake_capacity_n": 4400.0,
	"rear_brake_capacity_n": 1800.0,
	"engine_brake_torque_nm": 12.0,
	"steering_limit_rad": 0.50,
	"steering_rate_rad_s": 1.8,
	"throttle_rate_s": 3.0,
	"brake_rate_s": 5.0,
	"rider_max_lean_rad": 0.88,
	"rider_roll_gain": 14.0,
	"rider_roll_damping": 7.0,
	"low_speed_assist_threshold_m_s": 4.0,
	"fall_angle_rad": 1.12,
	"fall_slide_deceleration_m_s2": 5.0,
}

var position := Vector3.ZERO
var heading := 0.0
var speed := 0.0
var lean := 0.0
var lean_rate := 0.0
var steering := 0.0
var gear := 1
var rpm := 1400.0
var crashed := false
var crash_reason := ""
var elapsed := 0.0
var tick := 0
var longitudinal_acceleration := 0.0
var lateral_acceleration := 0.0
var front_load_n := 0.0
var rear_load_n := 0.0
var front_force_n := 0.0
var rear_force_n := 0.0
var lateral_force_n := 0.0
var requested_lateral_force_n := 0.0
var grip_utilization := 0.0
var target_lean := 0.0
var on_track := true
var assist_enabled := true
var auto_shift := true
var throttle_applied := 0.0
var front_brake_applied := 0.0
var rear_brake_applied := 0.0
var shift_remaining := 0.0
var last_shift_command := 0
var last_controls: Dictionary = {}


func reset(start_position: Vector3, start_heading: float, initial_speed: float = 0.0) -> void:
	position = start_position
	heading = start_heading
	speed = maxf(initial_speed, 0.0)
	lean = 0.0
	lean_rate = 0.0
	steering = 0.0
	gear = 1
	rpm = float(parameters.idle_rpm)
	crashed = false
	crash_reason = ""
	elapsed = 0.0
	tick = 0
	longitudinal_acceleration = 0.0
	lateral_acceleration = 0.0
	front_load_n = mass_kg() * GRAVITY * float(parameters.static_front_fraction)
	rear_load_n = mass_kg() * GRAVITY - front_load_n
	front_force_n = 0.0
	rear_force_n = 0.0
	lateral_force_n = 0.0
	requested_lateral_force_n = 0.0
	grip_utilization = 0.0
	target_lean = 0.0
	on_track = true
	assist_enabled = true
	auto_shift = true
	throttle_applied = 0.0
	front_brake_applied = 0.0
	rear_brake_applied = 0.0
	shift_remaining = 0.0
	last_shift_command = 0
	last_controls = {}


func mass_kg() -> float:
	return (
		float(parameters.bike_mass_kg)
		+ float(parameters.rider_mass_kg)
		+ float(parameters.fuel_mass_kg)
	)


func forward() -> Vector3:
	return Vector3(sin(heading), 0.0, -cos(heading))


func validate_controls(controls: Dictionary) -> String:
	for name: Variant in controls:
		if (
			name
			not in [
				"throttle",
				"front_brake",
				"rear_brake",
				"steer",
				"shift",
				"assist_enabled",
				"auto_shift"
			]
		):
			return "Unknown control %s" % str(name)
	for name: String in ["throttle", "front_brake", "rear_brake", "steer", "shift"]:
		var value: Variant = controls.get(name, 0.0)
		if not (value is float or value is int) or not is_finite(float(value)):
			return "Control %s must be a finite number" % name
		var lower := -1.0 if name in ["steer", "shift"] else 0.0
		if float(value) < lower or float(value) > 1.0:
			return "Control %s is outside its allowed range" % name
		if name == "shift" and float(value) != float(int(value)):
			return "Shift must be -1, 0, or 1"
	for name: String in ["assist_enabled", "auto_shift"]:
		if controls.has(name) and not controls[name] is bool:
			return "Control %s must be a boolean" % name
	return ""


func step(dt: float, controls: Dictionary, road_sample: Dictionary) -> Dictionary:
	var error := validate_controls(controls)
	if not is_finite(dt) or dt <= 0.0 or dt > 0.02:
		error = "Physics timestep must be positive and at most 0.02 seconds"
	if (
		not road_sample.has("height")
		or not road_sample.has("normal")
		or not road_sample.has("on_track")
	):
		error = "Road sample must include height, normal, and on_track"
	else:
		var height_value: Variant = road_sample.height
		var normal_value: Variant = road_sample.normal
		if not (height_value is float or height_value is int) or not is_finite(float(height_value)):
			error = "Road height must be finite"
		elif not normal_value is Vector3:
			error = "Road normal must be Vector3"
		elif (
			not normal_value.is_finite()
			or normal_value.length_squared() < 0.001
			or normal_value.y <= 0.0
		):
			error = "Road normal must be finite, nonzero and upward"
		elif not road_sample.on_track is bool:
			error = "Road on_track must be boolean"
	if not error.is_empty():
		return {"error": error, "tick": tick}
	last_controls = controls.duplicate(true)
	assist_enabled = bool(controls.get("assist_enabled", true))
	auto_shift = bool(controls.get("auto_shift", true))
	on_track = bool(road_sample.on_track)
	var normal: Vector3 = road_sample.normal
	normal = normal.normalized()
	var tangent := forward().slide(normal).normalized()
	position.y = float(road_sample.height)
	if crashed:
		speed = move_toward(speed, 0.0, float(parameters.fall_slide_deceleration_m_s2) * dt)
		position += tangent * speed * dt
		elapsed += dt
		tick += 1
		return telemetry()

	throttle_applied = move_toward(
		throttle_applied,
		float(controls.get("throttle", 0.0)),
		float(parameters.throttle_rate_s) * dt
	)
	front_brake_applied = move_toward(
		front_brake_applied,
		float(controls.get("front_brake", 0.0)),
		float(parameters.brake_rate_s) * dt
	)
	rear_brake_applied = move_toward(
		rear_brake_applied,
		float(controls.get("rear_brake", 0.0)),
		float(parameters.brake_rate_s) * dt
	)
	_update_drivetrain(dt, int(controls.get("shift", 0)))
	var total_mass := mass_kg()
	var normal_gravity := GRAVITY * maxf(normal.y, 0.0)
	var normal_force := total_mass * normal_gravity
	var friction := float(parameters.asphalt_friction if on_track else parameters.offtrack_friction)
	var roll_resistance := float(
		parameters.rolling_coefficient if on_track else parameters.offtrack_rolling_coefficient
	)
	# Prior tick tire acceleration estimates pitch load transfer. Forward drive unloads front.
	var transfer := (
		total_mass
		* longitudinal_acceleration
		* float(parameters.cg_height_m)
		/ float(parameters.wheelbase_m)
	)
	front_load_n = clampf(
		normal_force * float(parameters.static_front_fraction) - transfer, 0.0, normal_force
	)
	rear_load_n = normal_force - front_load_n
	var ratio := (
		float(parameters.primary_ratio)
		* float(GEAR_RATIOS[gear - 1])
		* float(parameters.final_ratio)
	)
	var drive_force := (
		engine_torque_nm(rpm)
		* throttle_applied
		* ratio
		* float(parameters.drivetrain_efficiency)
		/ float(parameters.rear_rolling_radius_m)
	)
	if shift_remaining > 0.0 or rpm >= float(parameters.rev_limit_rpm):
		drive_force = 0.0
	var engine_brake := (
		float(parameters.engine_brake_torque_nm)
		* (1.0 - throttle_applied)
		* ratio
		/ float(parameters.rear_rolling_radius_m)
	)
	engine_brake *= clampf(speed / 5.0, 0.0, 1.0)
	# Explicit ideal tire force limiter, not a claim to emulate proprietary ABS/TC.
	front_force_n = -minf(
		front_brake_applied * float(parameters.front_brake_capacity_n), friction * front_load_n
	)
	rear_force_n = clampf(
		drive_force - rear_brake_applied * float(parameters.rear_brake_capacity_n) - engine_brake,
		-friction * rear_load_n,
		friction * rear_load_n
	)
	var front_lateral_capacity := sqrt(
		maxf(0.0, pow(friction * front_load_n, 2.0) - front_force_n * front_force_n)
	)
	var rear_lateral_capacity := sqrt(
		maxf(0.0, pow(friction * rear_load_n, 2.0) - rear_force_n * rear_force_n)
	)
	var lateral_capacity := front_lateral_capacity + rear_lateral_capacity
	_update_steering(dt, float(controls.get("steer", 0.0)), normal_gravity)
	requested_lateral_force_n = (
		total_mass * speed * speed * tan(steering) / float(parameters.wheelbase_m)
	)
	lateral_force_n = clampf(requested_lateral_force_n, -lateral_capacity, lateral_capacity)
	lateral_acceleration = lateral_force_n / total_mass
	grip_utilization = absf(requested_lateral_force_n) / maxf(lateral_capacity, 1.0)
	var drag := (
		0.5 * float(parameters.air_density_kg_m3) * float(parameters.drag_area_m2) * speed * speed
	)
	var resistance := roll_resistance * normal_force * clampf(speed, 0.0, 1.0)
	var grade_acceleration := Vector3.DOWN.dot(tangent) * GRAVITY
	longitudinal_acceleration = (
		(front_force_n + rear_force_n - drag - resistance) / total_mass + grade_acceleration
	)
	var next_speed := maxf(0.0, speed + longitudinal_acceleration * dt)
	# Roll equation includes the destabilizing gravity term and countersteering response.
	var roll_acceleration := (
		(normal_gravity * sin(lean) - lateral_acceleration * cos(lean))
		/ float(parameters.cg_height_m)
	)
	if assist_enabled and speed < float(parameters.low_speed_assist_threshold_m_s):
		roll_acceleration = (
			-float(parameters.rider_roll_gain) * lean
			- float(parameters.rider_roll_damping) * lean_rate
		)
	lean_rate += roll_acceleration * dt
	lean += lean_rate * dt
	if speed > 0.1:
		heading = wrapf(heading + lateral_acceleration / speed * dt, -PI, PI)
	tangent = forward().slide(normal).normalized()
	position += tangent * (speed + next_speed) * 0.5 * dt
	speed = next_speed
	if absf(lean) >= float(parameters.fall_angle_rad):
		crashed = true
		crash_reason = "lean_contact"
		lean = signf(lean) * float(parameters.fall_angle_rad)
		lean_rate = 0.0
	elapsed += dt
	tick += 1
	return telemetry()


func _update_steering(dt: float, command: float, normal_gravity: float) -> void:
	var limit := float(parameters.steering_limit_rad)
	var requested_steering := command * limit
	target_lean = command * float(parameters.rider_max_lean_rad)
	if assist_enabled and speed >= float(parameters.low_speed_assist_threshold_m_s):
		var roll_acceleration := (
			float(parameters.rider_roll_gain) * (target_lean - lean)
			- float(parameters.rider_roll_damping) * lean_rate
		)
		var requested_acceleration := (
			(normal_gravity * sin(lean) - float(parameters.cg_height_m) * roll_acceleration)
			/ maxf(cos(lean), 0.25)
		)
		requested_steering = atan(
			requested_acceleration * float(parameters.wheelbase_m) / (speed * speed)
		)
	steering = move_toward(
		steering,
		clampf(requested_steering, -limit, limit),
		float(parameters.steering_rate_rad_s) * dt
	)


func _update_drivetrain(dt: float, shift_command: int) -> void:
	shift_remaining = maxf(0.0, shift_remaining - dt)
	var ratio := (
		float(parameters.primary_ratio)
		* float(GEAR_RATIOS[gear - 1])
		* float(parameters.final_ratio)
	)
	var wheel_rpm := speed / float(parameters.rear_rolling_radius_m) * 60.0 / TAU
	rpm = maxf(float(parameters.idle_rpm), wheel_rpm * ratio)
	# An explicit automatic slipping clutch permits standing starts.
	rpm = maxf(
		rpm,
		lerpf(float(parameters.idle_rpm), float(parameters.launch_clutch_rpm), throttle_applied)
	)
	var shift := 0
	if shift_command != 0 and shift_command != last_shift_command:
		shift = shift_command
	elif auto_shift:
		if rpm > float(parameters.shift_up_rpm):
			shift = 1
		elif rpm < float(parameters.shift_down_rpm) and gear > 1:
			shift = -1
	if shift_remaining <= 0.0 and shift != 0:
		var new_gear := clampi(gear + shift, 1, 6)
		if new_gear != gear:
			gear = new_gear
			shift_remaining = float(parameters.shift_duration_s)
	last_shift_command = shift_command


func engine_torque_nm(engine_rpm: float) -> float:
	# Estimated broad torque curve constrained by published US peak torque and power.
	# This is not a dyno map, throttle map, or manufacturer's engine controller.
	var fraction := clampf(engine_rpm / 11500.0, 0.0, 1.0)
	var torque := float(parameters.peak_torque_nm) * (0.55 + 0.45 * sin(fraction * PI * 0.5))
	var power_limited_torque := float(parameters.peak_power_w) / maxf(engine_rpm * TAU / 60.0, 1.0)
	return minf(torque, power_limited_torque)


func telemetry() -> Dictionary:
	return {
		"dynamics_version": MODEL_VERSION,
		"tick": tick,
		"elapsed": elapsed,
		"position": position,
		"heading": heading,
		"speed": speed,
		"lean": lean,
		"lean_rate": lean_rate,
		"target_lean": target_lean,
		"steering": steering,
		"gear": gear,
		"rpm": rpm,
		"crashed": crashed,
		"crash_reason": crash_reason,
		"on_track": on_track,
		"longitudinal_acceleration": longitudinal_acceleration,
		"lateral_acceleration": lateral_acceleration,
		"front_load_n": front_load_n,
		"rear_load_n": rear_load_n,
		"front_force_n": front_force_n,
		"rear_force_n": rear_force_n,
		"lateral_force_n": lateral_force_n,
		"grip_utilization": grip_utilization,
		"assist_enabled": assist_enabled,
		"auto_shift": auto_shift,
		"ideal_tire_force_limiter": true,
		"automatic_launch_clutch": true,
		"throttle_applied": throttle_applied,
		"front_brake_applied": front_brake_applied,
		"rear_brake_applied": rear_brake_applied,
		"shift_remaining": shift_remaining,
		"last_shift_command": last_shift_command,
		"requested_controls": last_controls.duplicate(true),
		"unsupported":
		[
			"suspension",
			"wheel_slip",
			"tire_temperature",
			"pitch_dynamics",
			"wheelies",
			"airborne_motion",
			"crash_contacts",
			"banked_roll_dynamics"
		],
	}
