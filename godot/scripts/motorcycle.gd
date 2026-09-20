class_name MotorcycleSim
extends RefCounted
## Explicitly stepped, reduced order motorcycle prototype. SI units throughout.
## This is original code, not validated Ducati or tire manufacturer dynamics.

const MODEL_VERSION := "reduced-order-combined-assist-v5"
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
var longitudinal_velocity := 0.0
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
var front_lateral_force_n := 0.0
var rear_lateral_force_n := 0.0
var front_lateral_capacity_n := 0.0
var rear_lateral_capacity_n := 0.0
var lateral_capacity_n := 0.0
var longitudinal_force_scale := 1.0
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
var road_bank_rad := 0.0
var lean_relative_road_rad := 0.0
var gravity_forward_m_s2 := 0.0
var gravity_right_m_s2 := 0.0
var gravity_normal_m_s2 := GRAVITY


func reset(start_position: Vector3, start_heading: float, initial_speed: float = 0.0) -> void:
	position = start_position
	heading = start_heading
	speed = maxf(initial_speed, 0.0)
	longitudinal_velocity = speed
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
	front_lateral_force_n = 0.0
	rear_lateral_force_n = 0.0
	front_lateral_capacity_n = 0.0
	rear_lateral_capacity_n = 0.0
	lateral_capacity_n = 0.0
	longitudinal_force_scale = 1.0
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
	road_bank_rad = 0.0
	lean_relative_road_rad = 0.0
	gravity_forward_m_s2 = 0.0
	gravity_right_m_s2 = 0.0
	gravity_normal_m_s2 = GRAVITY


func mass_kg() -> float:
	return (
		float(parameters.bike_mass_kg)
		+ float(parameters.rider_mass_kg)
		+ float(parameters.fuel_mass_kg)
	)


func forward() -> Vector3:
	return Vector3(sin(heading), 0.0, -cos(heading))


func surface_forward(normal: Vector3) -> Vector3:
	# Lift the horizontal heading onto the plane without changing its azimuth.
	# Orthogonal projection would also change azimuth on combined bank and grade.
	var direction := forward()
	direction.y = -(normal.x * direction.x + normal.z * direction.z) / normal.y
	return direction.normalized()


func _update_surface_frame(normal: Vector3, tangent: Vector3) -> void:
	var right := tangent.cross(normal).normalized()
	var upright := Vector3.UP.slide(tangent).normalized()
	var upright_right := tangent.cross(upright).normalized()
	road_bank_rad = atan2(normal.dot(upright_right), normal.dot(upright))
	lean_relative_road_rad = lean - road_bank_rad
	gravity_forward_m_s2 = GRAVITY * Vector3.DOWN.dot(tangent)
	gravity_right_m_s2 = GRAVITY * Vector3.DOWN.dot(right)
	gravity_normal_m_s2 = GRAVITY * normal.y


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
	# Unsupported contact regimes invalidate this transition atomically. Keep the
	# actual pre-step state, including actuator and drivetrain state, for reset/review.
	var snapshot: Dictionary = {}
	for property: Dictionary in get_property_list():
		if int(property.usage) & PROPERTY_USAGE_SCRIPT_VARIABLE and property.name != "parameters":
			var value: Variant = get(property.name)
			snapshot[property.name] = (
				value.duplicate(true) if value is Dictionary or value is Array else value
			)
	var result := _integrate_step(dt, controls, road_sample)
	if result.get("failure_type", "") == "unsupported_dynamics":
		for key: String in snapshot:
			set(key, snapshot[key])
		result["state"] = telemetry()
	return result


func _integrate_step(dt: float, controls: Dictionary, road_sample: Dictionary) -> Dictionary:
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
	var tangent := surface_forward(normal)
	_update_surface_frame(normal, tangent)
	position.y = float(road_sample.height)
	if crashed:
		longitudinal_velocity = move_toward(
			longitudinal_velocity, 0.0, float(parameters.fall_slide_deceleration_m_s2) * dt
		)
		speed = absf(longitudinal_velocity)
		position += tangent * longitudinal_velocity * dt
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
	var normal_gravity := gravity_normal_m_s2
	var normal_force := total_mass * normal_gravity
	var friction := float(parameters.asphalt_friction if on_track else parameters.offtrack_friction)
	var roll_resistance := float(
		parameters.rolling_coefficient if on_track else parameters.offtrack_rolling_coefficient
	)
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
	var front_brake_request := front_brake_applied * float(parameters.front_brake_capacity_n)
	var rear_brake_request := rear_brake_applied * float(parameters.rear_brake_capacity_n)
	# Aerodynamic drag and the effective rolling-resistance force are explicitly
	# modeled through COM, with no pitching moment. Tire forces act at road level.
	var drag := (
		0.5 * float(parameters.air_density_kg_m3) * float(parameters.drag_area_m2) * speed * speed
	)
	var resistance := roll_resistance * normal_force * clampf(speed, 0.0, 1.0)
	_update_steering(dt, float(controls.get("steer", 0.0)), normal_gravity)
	# Steering specifies total acceleration in the local road plane. Gravity's
	# cross slope component is already a force and must not consume tire grip.
	requested_lateral_force_n = (
		total_mass
		* (speed * speed * tan(steering) / float(parameters.wheelbase_m) - gravity_right_m_s2)
	)
	longitudinal_force_scale = 1.0
	var contact: Dictionary
	var travel_sign := signf(longitudinal_velocity)
	if speed <= 0.0:
		contact = _stationary_contact(
			normal_force, friction, drive_force, front_brake_request, rear_brake_request
		)
	else:
		contact = _contact_balance(
			normal_force,
			friction,
			-travel_sign * front_brake_request,
			drive_force - travel_sign * (rear_brake_request + engine_brake)
		)
	if contact.has("error"):
		return contact
	contact = _reserve_lateral(contact, normal_force, friction)
	var acceleration := (
		(
			(
				float(contact.front_force)
				+ float(contact.rear_force)
				- travel_sign * (drag + resistance)
			)
			/ total_mass
		)
		+ gravity_forward_m_s2
	)
	if contact.get("held", false):
		acceleration = 0.0
	var travel := longitudinal_velocity * dt + 0.5 * acceleration * dt * dt
	var next_velocity := longitudinal_velocity + acceleration * dt
	if (
		(longitudinal_velocity > 0.0 and next_velocity <= 0.0)
		or (longitudinal_velocity < 0.0 and next_velocity >= 0.0)
	):
		# Integrate to the actual stop event, not a trapezoid over a clipped speed.
		var stop_time := -longitudinal_velocity / acceleration
		travel = longitudinal_velocity * stop_time + 0.5 * acceleration * stop_time * stop_time
		contact = _stationary_contact(
			normal_force, friction, drive_force, front_brake_request, rear_brake_request
		)
		if contact.has("error"):
			return contact
		contact = _reserve_lateral(contact, normal_force, friction)
		acceleration = (
			(float(contact.front_force) + float(contact.rear_force)) / total_mass
			+ gravity_forward_m_s2
		)
		if contact.get("held", false):
			acceleration = 0.0
		var remaining := dt - stop_time
		next_velocity = acceleration * remaining
		travel += 0.5 * acceleration * remaining * remaining
	if absf(next_velocity) < 1e-10:
		next_velocity = 0.0
	front_load_n = contact.front_load
	rear_load_n = normal_force - front_load_n
	front_force_n = contact.front_force
	rear_force_n = contact.rear_force
	longitudinal_acceleration = acceleration
	front_lateral_capacity_n = sqrt(
		maxf(0.0, pow(friction * front_load_n, 2.0) - front_force_n * front_force_n)
	)
	rear_lateral_capacity_n = sqrt(
		maxf(0.0, pow(friction * rear_load_n, 2.0) - rear_force_n * rear_force_n)
	)
	# Steady yaw balance about the CG: a*Fyf - b*Fyr = 0.
	# b/L is the static front fraction, independent of current load transfer.
	# This uses road-frame axle forces (small steering angle approximation).
	var front_share := float(parameters.static_front_fraction)
	var rear_share := 1.0 - front_share
	lateral_capacity_n = minf(
		front_lateral_capacity_n / front_share if front_share > 0.0 else INF,
		rear_lateral_capacity_n / rear_share if rear_share > 0.0 else INF
	)
	lateral_force_n = clampf(requested_lateral_force_n, -lateral_capacity_n, lateral_capacity_n)
	front_lateral_force_n = front_share * lateral_force_n
	rear_lateral_force_n = rear_share * lateral_force_n
	lateral_acceleration = lateral_force_n / total_mass + gravity_right_m_s2
	grip_utilization = absf(requested_lateral_force_n) / maxf(lateral_capacity_n, 1.0)
	# In the road frame, gravity and inertial acceleration both contribute.
	# lean is world upright roll; relative lean determines contact clearance.
	var roll_acceleration := (
		(
			normal_gravity * sin(lean_relative_road_rad)
			+ (gravity_right_m_s2 - lateral_acceleration) * cos(lean_relative_road_rad)
		)
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
		var horizontal_length_squared := tangent.x * tangent.x + tangent.z * tangent.z
		var azimuth_rate := (
			lateral_acceleration / longitudinal_velocity * normal.y / horizontal_length_squared
		)
		heading = wrapf(heading + azimuth_rate * dt, -PI, PI)
	tangent = surface_forward(normal)
	_update_surface_frame(normal, tangent)
	position += tangent * travel
	longitudinal_velocity = next_velocity
	speed = absf(longitudinal_velocity)
	if absf(lean_relative_road_rad) >= float(parameters.fall_angle_rad):
		crashed = true
		crash_reason = "lean_contact"
		lean_relative_road_rad = signf(lean_relative_road_rad) * float(parameters.fall_angle_rad)
		lean = road_bank_rad + lean_relative_road_rad
		lean_rate = 0.0
	elapsed += dt
	tick += 1
	return telemetry()


func _unsupported(reason: String) -> Dictionary:
	return {"error": reason, "failure_type": "unsupported_dynamics", "rollout_valid": false}


func _reserve_lateral(contact: Dictionary, total_load: float, friction: float) -> Dictionary:
	# Explicit ideal combined TC/ABS: preserve the rider's lateral requirement
	# before spending remaining grip on longitudinal acceleration. No added force.
	# A held stationary reaction cannot be scaled without violating grade balance.
	longitudinal_force_scale = 1.0
	if not assist_enabled or contact.get("held", false):
		return contact
	var share := float(parameters.static_front_fraction)
	var lateral := minf(absf(requested_lateral_force_n), friction * total_load)
	var xf := float(contact.front_force)
	var xr := float(contact.rear_force)
	var ratio := float(parameters.cg_height_m) / float(parameters.wheelbase_m)
	var low := 0.0
	var high := 1.0
	# At scale zero, static loads support the reserved pure lateral force. Each
	# friction-circle constraint is convex in scale, so feasible scales form an
	# interval containing zero. Check one first to avoid work when unrestricted.
	for iteration in 49:
		var scale := 1.0 if iteration == 0 else (low + high) * 0.5
		var nf := share * total_load - ratio * scale * (xf + xr)
		var nr := total_load - nf
		var feasible := (
			nf >= 0.0
			and nr >= 0.0
			and (pow(scale * xf, 2) + pow(share * lateral, 2)) <= pow(friction * nf, 2)
			and ((pow(scale * xr, 2) + pow((1.0 - share) * lateral, 2)) <= pow(friction * nr, 2))
		)
		if feasible:
			low = scale
			if iteration == 0:
				break
		else:
			high = scale
	longitudinal_force_scale = low
	return {
		"front_load": share * total_load - ratio * low * (xf + xr),
		"front_force": low * xf,
		"rear_force": low * xr,
	}


func _forces_at_load(
	front_load: float, total_load: float, friction: float, front_request: float, rear_request: float
) -> Dictionary:
	return {
		"front_load": front_load,
		"front_force": clampf(front_request, -friction * front_load, friction * front_load),
		"rear_force":
		clampf(
			rear_request,
			-friction * (total_load - front_load),
			friction * (total_load - front_load)
		),
	}


func _contact_balance(
	total_load: float, friction: float, front_request: float, rear_request: float
) -> Dictionary:
	# Fixed requests yield piecewise affine load balance. Enumerating every tire
	# saturation breakpoint solves it without lag, iterative history, or load floors.
	var ratio := float(parameters.cg_height_m) / float(parameters.wheelbase_m)
	var static_front := float(parameters.static_front_fraction) * total_load
	var tolerance := maxf(total_load, 1.0) * 1e-9
	var points: Array[float] = [0.0, total_load]
	if friction > 0.0:
		for candidate: float in [
			absf(front_request) / friction, total_load - absf(rear_request) / friction
		]:
			if candidate > 0.0 and candidate < total_load:
				points.append(candidate)
	points.sort()
	var roots: Array[float] = []
	for i in range(points.size() - 1):
		var left := points[i]
		var right := points[i + 1]
		var a := _forces_at_load(left, total_load, friction, front_request, rear_request)
		var b := _forces_at_load(right, total_load, friction, front_request, rear_request)
		var ra := left - static_front + ratio * (float(a.front_force) + float(a.rear_force))
		var rb := right - static_front + ratio * (float(b.front_force) + float(b.rear_force))
		if right - left > tolerance and absf(ra) <= tolerance and absf(rb) <= tolerance:
			return _unsupported(
				"Nonunique quasistatic axle load equilibrium requires pitch dynamics"
			)
		var candidates: Array[float] = []
		if absf(ra) <= tolerance:
			candidates.append(left)
		if absf(rb) <= tolerance:
			candidates.append(right)
		if ra * rb < 0.0:
			candidates.append(left - ra * (right - left) / (rb - ra))
		for candidate: float in candidates:
			var duplicate := false
			for existing: float in roots:
				if absf(existing - candidate) <= tolerance:
					duplicate = true
			if not duplicate:
				roots.append(candidate)
	if roots.size() != 1:
		return _unsupported(
			"No unique two-contact axle equilibrium; wheel lift or pitch dynamics required"
		)
	return _forces_at_load(roots[0], total_load, friction, front_request, rear_request)


func _stationary_contact(
	total_load: float, friction: float, drive: float, front_brake: float, rear_brake: float
) -> Dictionary:
	# At rest brake force is a reaction within the commanded torque capacity,
	# not a force that accelerates a parked motorcycle backwards. A signed fraction
	# distributes that reaction proportionally between the commanded brakes.
	var required_force := -mass_kg() * gravity_forward_m_s2
	var ratio := float(parameters.cg_height_m) / float(parameters.wheelbase_m)
	var front_load := float(parameters.static_front_fraction) * total_load - ratio * required_force
	var tolerance := maxf(total_load, 1.0) * 1e-9
	if front_load >= 0.0 and front_load <= total_load:
		var low := -1.0
		var high := 1.0
		var minimum := _forces_at_load(
			front_load, total_load, friction, -front_brake, drive - rear_brake
		)
		var maximum := _forces_at_load(
			front_load, total_load, friction, front_brake, drive + rear_brake
		)
		var min_force := float(minimum.front_force) + float(minimum.rear_force)
		var max_force := float(maximum.front_force) + float(maximum.rear_force)
		if required_force >= min_force - tolerance and required_force <= max_force + tolerance:
			var held: Dictionary
			for iteration in 60:
				var fraction := (low + high) * 0.5
				held = _forces_at_load(
					front_load,
					total_load,
					friction,
					-fraction * front_brake,
					drive - fraction * rear_brake
				)
				var force := float(held.front_force) + float(held.rear_force)
				if force > required_force:
					low = fraction
				else:
					high = fraction
			held["held"] = true
			return held
	# If no stationary reaction is available, let gravity/drive choose breakaway
	# direction, then apply brake torque opposite that impending motion.
	var free := _contact_balance(total_load, friction, 0.0, drive)
	if free.has("error"):
		return free
	var free_force := float(free.front_force) + float(free.rear_force) - required_force
	var direction := signf(free_force)
	var moving := _contact_balance(
		total_load, friction, -direction * front_brake, drive - direction * rear_brake
	)
	if moving.has("error"):
		return moving
	return moving


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
			gravity_right_m_s2
			+ (
				(
					normal_gravity * sin(lean_relative_road_rad)
					- float(parameters.cg_height_m) * roll_acceleration
				)
				/ maxf(cos(lean_relative_road_rad), 0.25)
			)
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
		"longitudinal_velocity": longitudinal_velocity,
		"lean": lean,
		"lean_rate": lean_rate,
		"road_bank_rad": road_bank_rad,
		"lean_relative_road_rad": lean_relative_road_rad,
		"gravity_forward_m_s2": gravity_forward_m_s2,
		"gravity_right_m_s2": gravity_right_m_s2,
		"gravity_normal_m_s2": gravity_normal_m_s2,
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
		"front_lateral_force_n": front_lateral_force_n,
		"rear_lateral_force_n": rear_lateral_force_n,
		"front_lateral_capacity_n": front_lateral_capacity_n,
		"rear_lateral_capacity_n": rear_lateral_capacity_n,
		"lateral_capacity_n": lateral_capacity_n,
		"requested_lateral_force_n": requested_lateral_force_n,
		"grip_utilization": grip_utilization,
		"assist_enabled": assist_enabled,
		"auto_shift": auto_shift,
		"ideal_tire_force_limiter": true,
		"combined_force_policy": "lateral_priority" if assist_enabled else "longitudinal_priority",
		"longitudinal_force_scale": longitudinal_force_scale,
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
			"yaw_inertia",
			"steered_tire_force_rotation",
			"tire_temperature",
			"pitch_dynamics",
			"wheelies",
			"airborne_motion",
			"crash_contacts",
			"surface_curvature_normal_load",
			"surface_frame_rotation_dynamics"
		],
	}
