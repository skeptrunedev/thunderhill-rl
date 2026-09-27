class_name CarSim
extends RefCounted
## Explicitly stepped planar dynamic car: a single track (bicycle) yaw/sideslip
## model with per wheel vertical loads. SI units throughout. Original code, not
## validated Mazda or BFGoodrich dynamics; see docs/car-reference.md.
##
## Tires: normalized Pacejka Magic Formula lateral force per wheel on its own
## load and load-sensitive friction, combined with the wheel's longitudinal force
## through a friction ellipse. Longitudinal force is the commanded drive or
## brake force clipped at the wheel's friction limit (no wheel spin state: ideal
## traction and ABS limiting). Loads carry quasi-static longitudinal and lateral
## transfer from the previous substep's accelerations.
##
## Interface mirrors MotorcycleSim so main.gd swaps vehicles at one point. The
## motorcycle's lean state is reported as the car's body roll: lean is world
## roll (body roll plus road bank, positive = right side down, as the
## motorcycle's lean), lean_rate its rate, lean_relative_road_rad the body roll
## on its suspension. A car cannot fall; a spin (sideslip beyond
## spin_sideslip_rad, or rolling backwards) sets crashed with crash_reason
## "spin", which main.gd and the trainers treat like the motorcycle's fall.

const MODEL_VERSION := "mx5-cup-single-track-mf-v1"
const VEHICLE := "car"
const GRAVITY := 9.81
const SUBSTEPS := 8
# Mazda ND six speed manual ratios (docs/car-reference.md).
const GEAR_RATIOS := [5.087, 2.991, 2.035, 1.594, 1.286, 1.000]
# Estimated wide open throttle torque (rpm, N m) through the published 205 N m
# at 4,000 rpm and 135 kW at 7,000 rpm. Not a dyno map.
const TORQUE_CURVE := [
	Vector2(1000, 130), Vector2(2000, 165), Vector2(3000, 190), Vector2(4000, 205),
	Vector2(5000, 203), Vector2(6000, 196), Vector2(7000, 184), Vector2(7500, 168),
]

const FRICTION_KEYS := {
	"curb": "curb_friction", "asphalt": "asphalt_friction", "offroad": "offtrack_friction"
}
const ROLLING_KEYS := {
	"curb": "curb_rolling_coefficient",
	"asphalt": "rolling_coefficient",
	"offroad": "offtrack_rolling_coefficient"
}

# Published values are sourced in docs/car-reference.md; the rest are labelled
# engineering estimates there, not measured calibration.
var parameters: Dictionary = {
	"car_mass_kg": 1050.0,
	"driver_mass_kg": 80.0,
	"fuel_mass_kg": 25.0,
	"wheelbase_m": 2.310,
	"front_track_m": 1.495,
	"rear_track_m": 1.505,
	"cg_height_m": 0.44,
	"static_front_fraction": 0.50,
	"yaw_inertia_kg_m2": 1350.0,
	"roll_stiffness_front_fraction": 0.55,
	"roll_gradient_rad_per_m_s2": 0.0025,
	"tire_rolling_radius_m": 0.305,
	"final_ratio": 2.866,
	"drivetrain_efficiency": 0.88,
	"rotating_mass_base": 0.04,
	"rotating_mass_per_ratio2": 0.0015,
	"idle_rpm": 900.0,
	"rev_limit_rpm": 7500.0,
	"shift_up_rpm": 7300.0,
	"shift_down_rpm": 4000.0,
	"shift_duration_s": 0.08,
	"launch_clutch_rpm": 3500.0,
	"engine_brake_torque_nm": 30.0,
	"drag_area_m2": 0.78,
	"air_density_kg_m3": 1.225,
	"rolling_coefficient": 0.015,
	"asphalt_friction": 1.30,
	# Estimated painted curb grip, below the asphalt racing surface.
	"curb_friction": 1.15,
	"curb_rolling_coefficient": 0.015,
	"offtrack_friction": 0.48,
	"offtrack_rolling_coefficient": 0.07,
	"load_sensitivity": 0.12,
	"tire_nominal_load_n": 2800.0,
	# Magic Formula stiffness per axle. The stiffer rear gives the stable
	# understeer balance of a road car (Cr b > Cf a); identical axles on a 50/50
	# car oversteer, and above a critical speed the yaw mode diverges.
	"tire_shape_b_front": 14.0,
	"tire_shape_b_rear": 20.0,
	"tire_shape_c": 1.45,
	"tire_shape_e": 0.1,
	"brake_capacity_n": 18000.0,
	"brake_balance_front": 0.70,
	# Ideal ABS: braking uses at most this fraction of a wheel's friction limit,
	# keeping lateral grip as a slip controlled tire does. Drive has no traction
	# control, so wheel spin can still use the whole ellipse.
	"abs_utilization": 0.90,
	"steering_limit_rad": 0.42,
	"steering_rate_rad_s": 1.4,
	"throttle_rate_s": 8.0,
	"brake_rate_s": 8.0,
	"spin_sideslip_rad": 0.8,
	"spin_min_speed_m_s": 4.0,
	"spin_slide_deceleration_m_s2": 6.0,
}

var position := Vector3.ZERO
var heading := 0.0
var speed := 0.0
var longitudinal_velocity := 0.0
var lateral_velocity := 0.0
var yaw_rate := 0.0
var sideslip := 0.0
var lean := 0.0
var lean_rate := 0.0
var body_roll := 0.0
var steering := 0.0
var gear := 1
var rpm := 900.0
var crashed := false
var crash_reason := ""
var collision_contact: Dictionary = {}
var elapsed := 0.0
var tick := 0
var longitudinal_acceleration := 0.0
var lateral_acceleration := 0.0
var wheel_loads_n := [0.0, 0.0, 0.0, 0.0]
var wheel_longitudinal_n := [0.0, 0.0, 0.0, 0.0]
var wheel_lateral_n := [0.0, 0.0, 0.0, 0.0]
var front_slip_angle := 0.0
var rear_slip_angle := 0.0
var front_load_n := 0.0
var rear_load_n := 0.0
var front_force_n := 0.0
var rear_force_n := 0.0
var front_lateral_force_n := 0.0
var rear_lateral_force_n := 0.0
var lateral_force_n := 0.0
var grip_utilization := 0.0
var on_track := true
var surface_material := "unknown"
var surface_friction := 0.0
var surface_rolling_coefficient := 0.0
var auto_shift := true
var throttle_applied := 0.0
var brake_applied := 0.0
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
var _yaw_travel := 0.0


func reset(start_position: Vector3, start_heading: float, initial_speed: float = 0.0) -> void:
	position = start_position
	heading = start_heading
	speed = maxf(initial_speed, 0.0)
	longitudinal_velocity = speed
	lateral_velocity = 0.0
	yaw_rate = 0.0
	sideslip = 0.0
	lean = 0.0
	lean_rate = 0.0
	body_roll = 0.0
	steering = 0.0
	# Rolling starts begin in the gear automatic shifting settles into.
	gear = 1
	while gear < GEAR_RATIOS.size() and _engine_rpm_in_gear(gear) > float(parameters.shift_up_rpm):
		gear += 1
	rpm = maxf(float(parameters.idle_rpm), _engine_rpm_in_gear(gear))
	crashed = false
	crash_reason = ""
	collision_contact.clear()
	elapsed = 0.0
	tick = 0
	longitudinal_acceleration = 0.0
	lateral_acceleration = 0.0
	var weight := mass_kg() * GRAVITY
	var front := float(parameters.static_front_fraction)
	wheel_loads_n = [weight * front * 0.5, weight * front * 0.5,
		weight * (1.0 - front) * 0.5, weight * (1.0 - front) * 0.5]
	wheel_longitudinal_n = [0.0, 0.0, 0.0, 0.0]
	wheel_lateral_n = [0.0, 0.0, 0.0, 0.0]
	front_slip_angle = 0.0
	rear_slip_angle = 0.0
	front_load_n = weight * front
	rear_load_n = weight - front_load_n
	front_force_n = 0.0
	rear_force_n = 0.0
	front_lateral_force_n = 0.0
	rear_lateral_force_n = 0.0
	lateral_force_n = 0.0
	grip_utilization = 0.0
	on_track = true
	surface_material = "unknown"
	surface_friction = 0.0
	surface_rolling_coefficient = 0.0
	auto_shift = true
	throttle_applied = 0.0
	brake_applied = 0.0
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
		float(parameters.car_mass_kg)
		+ float(parameters.driver_mass_kg)
		+ float(parameters.fuel_mass_kg)
	)


func wheel_radius_m() -> float:
	return float(parameters.tire_rolling_radius_m)


func forward() -> Vector3:
	return Vector3(sin(heading), 0.0, -cos(heading))


func surface_forward(normal: Vector3) -> Vector3:
	# Same azimuth preserving lift onto the road plane as MotorcycleSim.
	var direction := forward()
	direction.y = -(normal.x * direction.x + normal.z * direction.z) / normal.y
	return direction.normalized()


func _update_surface_frame(normal: Vector3, tangent: Vector3) -> void:
	var right := tangent.cross(normal).normalized()
	var upright := Vector3.UP.slide(tangent).normalized()
	var upright_right := tangent.cross(upright).normalized()
	road_bank_rad = atan2(normal.dot(upright_right), normal.dot(upright))
	lean_relative_road_rad = body_roll
	lean = road_bank_rad + body_roll
	gravity_forward_m_s2 = GRAVITY * Vector3.DOWN.dot(tangent)
	gravity_right_m_s2 = GRAVITY * Vector3.DOWN.dot(right)
	gravity_normal_m_s2 = GRAVITY * normal.y


func validate_controls(controls: Dictionary) -> String:
	# The motorcycle's control vocabulary: the brake pedal is the larger of
	# front_brake and rear_brake, split by brake_balance_front. assist_enabled is
	# accepted and ignored (no rider assists on a car).
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


func step(
	dt: float, controls: Dictionary, road_sample: Dictionary, after_step: Callable = Callable()
) -> Dictionary:
	# Unsupported or failed transitions restore the complete pre-step state.
	var snapshot: Dictionary = {}
	for property: Dictionary in get_property_list():
		if int(property.usage) & PROPERTY_USAGE_SCRIPT_VARIABLE and property.name != "parameters":
			var value: Variant = get(property.name)
			snapshot[property.name] = (
				value.duplicate(true) if value is Dictionary or value is Array else value
			)
	var result := _integrate_step(dt, controls, road_sample)
	if not result.has("error") and after_step.is_valid():
		var ground: Dictionary = after_step.call()
		if ground.has("error"):
			result = ground
		else:
			result["ground_after"] = ground
	if result.has("error"):
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
	if road_sample.has("on_curb") and not road_sample.on_curb is bool:
		error = "Road on_curb must be boolean"
	if road_sample.has("on_pavement") and not road_sample.on_pavement is bool:
		error = "Road on_pavement must be boolean"
	if not error.is_empty():
		return {"error": error, "tick": tick}
	var material_name := (
		"curb"
		if road_sample.get("on_curb", false)
		else ("asphalt" if road_sample.get("on_pavement", road_sample.on_track) else "offroad")
	)
	for key: String in [FRICTION_KEYS[material_name], ROLLING_KEYS[material_name]]:
		var value: Variant = parameters.get(key)
		if (
			not (value is float or value is int)
			or not is_finite(float(value))
			or float(value) < 0.0
		):
			return {
				"error": "Surface coefficient %s must be finite and nonnegative" % key, "tick": tick
			}
	surface_material = material_name
	surface_friction = float(parameters[FRICTION_KEYS[material_name]])
	surface_rolling_coefficient = float(parameters[ROLLING_KEYS[material_name]])
	last_controls = controls.duplicate(true)
	auto_shift = bool(controls.get("auto_shift", true))
	on_track = bool(road_sample.on_track)
	var normal: Vector3 = road_sample.normal
	normal = normal.normalized()
	var tangent := surface_forward(normal)
	_update_surface_frame(normal, tangent)
	position.y = float(road_sample.height)
	if crashed:
		# A spun car slides to rest along its velocity; no further control.
		var planar := Vector2(longitudinal_velocity, lateral_velocity)
		var slowed := planar.move_toward(
			Vector2.ZERO, float(parameters.spin_slide_deceleration_m_s2) * dt
		)
		longitudinal_velocity = slowed.x
		lateral_velocity = slowed.y
		speed = slowed.length()
		yaw_rate = move_toward(yaw_rate, 0.0, 2.0 * dt)
		var right := tangent.cross(normal).normalized()
		position += (tangent * longitudinal_velocity + right * lateral_velocity) * dt
		heading = wrapf(heading + yaw_rate * dt, -PI, PI)
		elapsed += dt
		tick += 1
		return telemetry()

	throttle_applied = move_toward(
		throttle_applied,
		float(controls.get("throttle", 0.0)),
		float(parameters.throttle_rate_s) * dt
	)
	var pedal := maxf(float(controls.get("front_brake", 0.0)), float(controls.get("rear_brake", 0.0)))
	brake_applied = move_toward(brake_applied, pedal, float(parameters.brake_rate_s) * dt)
	front_brake_applied = brake_applied
	rear_brake_applied = brake_applied
	var limit := float(parameters.steering_limit_rad)
	steering = move_toward(
		steering,
		clampf(float(controls.get("steer", 0.0)), -1.0, 1.0) * limit,
		float(parameters.steering_rate_rad_s) * dt
	)
	_update_drivetrain(dt, int(controls.get("shift", 0)))
	var ratio := float(GEAR_RATIOS[gear - 1]) * float(parameters.final_ratio)
	var radius := wheel_radius_m()
	var drive_force := (
		engine_torque_nm(rpm)
		* throttle_applied
		* ratio
		* float(parameters.drivetrain_efficiency)
		/ radius
	)
	if shift_remaining > 0.0 or rpm >= float(parameters.rev_limit_rpm):
		drive_force = 0.0
	var engine_brake := (
		float(parameters.engine_brake_torque_nm) * (1.0 - throttle_applied) * ratio / radius
	)
	engine_brake *= clampf(longitudinal_velocity / 5.0, 0.0, 1.0)
	var brake_force := brake_applied * float(parameters.brake_capacity_n)
	var balance := float(parameters.brake_balance_front)
	var inertia_factor := (
		1.0
		+ float(parameters.rotating_mass_base)
		+ float(parameters.rotating_mass_per_ratio2) * ratio * ratio
	)

	var h := dt / SUBSTEPS
	var travel := Vector2.ZERO
	_yaw_travel = 0.0
	for substep in SUBSTEPS:
		# Body frame displacement, rotated by the yaw accumulated this step
		# into the (forward, right) frame of the step's starting heading.
		var turned := _yaw_travel
		var moved := _substep(
			h, drive_force, engine_brake, brake_force, balance, inertia_factor
		)
		travel += Vector2(
			moved.x * cos(turned) - moved.y * sin(turned),
			moved.x * sin(turned) + moved.y * cos(turned)
		)
	# Heading follows the road plane yaw, as the motorcycle's azimuth rate does.
	var horizontal_length_squared := tangent.x * tangent.x + tangent.z * tangent.z
	var start_tangent := tangent
	var right_vector := tangent.cross(normal).normalized()
	heading = wrapf(
		heading + _yaw_travel * normal.y / horizontal_length_squared, -PI, PI
	)
	position += start_tangent * travel.x + right_vector * travel.y
	tangent = surface_forward(normal)
	speed = Vector2(longitudinal_velocity, lateral_velocity).length()
	sideslip = atan2(lateral_velocity, maxf(absf(longitudinal_velocity), 0.5))
	# Quasi-static body roll on the suspension: outward in a turn, so a right
	# hand turn (positive lateral acceleration) rolls the body left (negative).
	var previous_roll := body_roll
	body_roll = -float(parameters.roll_gradient_rad_per_m_s2) * lateral_acceleration
	lean_rate = (body_roll - previous_roll) / dt
	_update_surface_frame(normal, tangent)
	if (
		speed >= float(parameters.spin_min_speed_m_s)
		and (
			absf(sideslip) >= float(parameters.spin_sideslip_rad)
			or longitudinal_velocity < 0.0
		)
	):
		crashed = true
		crash_reason = "spin"
	elapsed += dt
	tick += 1
	return telemetry()



## One explicit substep of the planar single track model. Returns the body
## frame displacement (forward, right) and accumulates yaw into _yaw_travel.
func _substep(
	h: float,
	drive_force: float,
	engine_brake: float,
	brake_force: float,
	balance: float,
	inertia_factor: float
) -> Vector2:
	if h == 0.0:
		return Vector2.ZERO
	var m := mass_kg()
	var wheelbase := float(parameters.wheelbase_m)
	var a := wheelbase * (1.0 - float(parameters.static_front_fraction))
	var b := wheelbase - a
	var cg := float(parameters.cg_height_m)
	var weight := m * gravity_normal_m_s2
	# Quasi-static load transfer from the previous substep's accelerations.
	# Transfer follows the tire and aero forces, not gravity's road components.
	var inertial_x := longitudinal_acceleration - gravity_forward_m_s2
	var front_axle := weight * b / wheelbase - m * inertial_x * cg / wheelbase
	var rear_axle := weight - front_axle
	var roll_front := float(parameters.roll_stiffness_front_fraction)
	var lateral_transfer := m * (lateral_acceleration - gravity_right_m_s2) * cg
	var front_shift := lateral_transfer * roll_front / float(parameters.front_track_m)
	var rear_shift := lateral_transfer * (1.0 - roll_front) / float(parameters.rear_track_m)
	# Order: front left, front right, rear left, rear right. A right hand turn
	# (positive lateral acceleration) loads the left (outside) wheels.
	var loads := [
		maxf(0.0, 0.5 * front_axle + front_shift),
		maxf(0.0, 0.5 * front_axle - front_shift),
		maxf(0.0, 0.5 * rear_axle + rear_shift),
		maxf(0.0, 0.5 * rear_axle - rear_shift),
	]
	var vx := longitudinal_velocity
	var vy := lateral_velocity
	var r := yaw_rate
	# Slip angle denominators keep a small floor so the model stays defined
	# from a standing start; at walking pace the tires are nearly linear.
	var denominator := maxf(absf(vx), 1.0)
	front_slip_angle = steering - atan2(vy + a * r, denominator)
	rear_slip_angle = -atan2(vy - b * r, denominator)
	var moving := vx > 0.05
	var demands := [
		-balance * 0.5 * brake_force if moving else 0.0,
		-balance * 0.5 * brake_force if moving else 0.0,
		0.5 * (drive_force - ((1.0 - balance) * brake_force + engine_brake if moving else 0.0)),
		0.5 * (drive_force - ((1.0 - balance) * brake_force + engine_brake if moving else 0.0)),
	]
	var fx := [0.0, 0.0, 0.0, 0.0]
	var fy := [0.0, 0.0, 0.0, 0.0]
	var utilization := 0.0
	for i in 4:
		var load: float = loads[i]
		var mu := _load_friction(load)
		var capacity := mu * load
		var demand := float(demands[i])
		var longitudinal := clampf(
			demand,
			-capacity * (float(parameters.abs_utilization) if demand < 0.0 else 1.0),
			capacity
		)
		var slip: float = front_slip_angle if i < 2 else rear_slip_angle
		var pure := capacity * _magic_formula(slip, i < 2)
		var ellipse := 1.0
		if capacity > 0.0:
			ellipse = sqrt(maxf(0.0, 1.0 - pow(longitudinal / capacity, 2.0)))
			utilization = maxf(
				utilization, Vector2(longitudinal, pure * ellipse).length() / capacity
			)
		fx[i] = longitudinal
		fy[i] = pure * ellipse
	var front_x: float = fx[0] + fx[1]
	var rear_x: float = fx[2] + fx[3]
	var front_y: float = fy[0] + fy[1]
	var rear_y: float = fy[2] + fy[3]
	var cos_steer := cos(steering)
	var sin_steer := sin(steering)
	var body_x := front_x * cos_steer - front_y * sin_steer + rear_x
	var body_y := front_x * sin_steer + front_y * cos_steer + rear_y
	var yaw_moment := a * (front_x * sin_steer + front_y * cos_steer) - b * rear_y
	var drag := (
		0.5 * float(parameters.air_density_kg_m3) * float(parameters.drag_area_m2) * vx * absf(vx)
	)
	var resistance := surface_rolling_coefficient * weight * clampf(vx, 0.0, 1.0)
	var ax := (body_x - drag - resistance) / (m * inertia_factor) + gravity_forward_m_s2
	var ay := body_y / m + gravity_right_m_s2
	var next_vx := vx + (ax + vy * r) * h
	var next_vy := vy + (ay - vx * r) * h
	var next_r := r + yaw_moment / float(parameters.yaw_inertia_kg_m2) * h
	if not moving and next_vx <= maxf(vx, 0.0):
		# At rest the brakes and drivetrain hold the car: it neither creeps nor
		# rolls backwards (there is no reverse gear).
		next_vx = 0.0
		next_vy = 0.0
		next_r = 0.0
	if vx > 0.0 and next_vx < 0.0 and brake_force + engine_brake > 0.0 and drive_force <= 0.0:
		# Brakes stop the car; they never drive it backwards.
		next_vx = 0.0
	var displacement := Vector2(
		0.5 * (vx + next_vx) * h, 0.5 * (vy + next_vy) * h
	)
	_yaw_travel += 0.5 * (r + next_r) * h
	longitudinal_velocity = next_vx
	lateral_velocity = next_vy
	yaw_rate = next_r
	longitudinal_acceleration = ax
	lateral_acceleration = ay
	wheel_loads_n = loads
	wheel_longitudinal_n = fx
	wheel_lateral_n = fy
	front_load_n = loads[0] + loads[1]
	rear_load_n = loads[2] + loads[3]
	front_force_n = front_x
	rear_force_n = rear_x
	front_lateral_force_n = front_y
	rear_lateral_force_n = rear_y
	lateral_force_n = body_y
	grip_utilization = utilization
	return displacement


## Load sensitive friction: heavier loaded tires produce less grip per newton.
func _load_friction(load: float) -> float:
	var nominal := float(parameters.tire_nominal_load_n)
	var factor := 1.0 - float(parameters.load_sensitivity) * (load / nominal - 1.0)
	return surface_friction * clampf(factor, 0.5, 1.3)


## Normalized Magic Formula, sin(C atan(B a - E (B a - atan(B a)))), peak one.
func _magic_formula(slip: float, front: bool) -> float:
	var b := float(parameters.tire_shape_b_front if front else parameters.tire_shape_b_rear)
	var c := float(parameters.tire_shape_c)
	var e := float(parameters.tire_shape_e)
	var x := b * slip
	return sin(c * atan(x - e * (x - atan(x))))


func _engine_rpm_in_gear(selected_gear: int) -> float:
	var ratio := float(GEAR_RATIOS[selected_gear - 1]) * float(parameters.final_ratio)
	return maxf(longitudinal_velocity, 0.0) / wheel_radius_m() * 60.0 / TAU * ratio


func _update_drivetrain(dt: float, shift_command: int) -> void:
	shift_remaining = maxf(0.0, shift_remaining - dt)
	rpm = maxf(float(parameters.idle_rpm), _engine_rpm_in_gear(gear))
	# An automatic slipping launch clutch permits standing starts.
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
			# Downshift only when the lower gear stays below the upshift point.
			if _engine_rpm_in_gear(gear - 1) < float(parameters.shift_up_rpm) * 0.97:
				shift = -1
	if shift_remaining <= 0.0 and shift != 0:
		var new_gear := clampi(gear + shift, 1, GEAR_RATIOS.size())
		if new_gear != gear:
			gear = new_gear
			shift_remaining = float(parameters.shift_duration_s)
	last_shift_command = shift_command


func engine_torque_nm(engine_rpm: float) -> float:
	var first: Vector2 = TORQUE_CURVE[0]
	if engine_rpm <= first.x:
		return first.y
	for i in range(1, TORQUE_CURVE.size()):
		var high: Vector2 = TORQUE_CURVE[i]
		if engine_rpm <= high.x:
			var low: Vector2 = TORQUE_CURVE[i - 1]
			return lerpf(low.y, high.y, (engine_rpm - low.x) / (high.x - low.x))
	return TORQUE_CURVE[-1].y


func telemetry() -> Dictionary:
	return {
		"vehicle": VEHICLE,
		"dynamics_version": MODEL_VERSION,
		"tick": tick,
		"elapsed": elapsed,
		"position": position,
		"heading": heading,
		"speed": speed,
		"longitudinal_velocity": longitudinal_velocity,
		"lateral_velocity": lateral_velocity,
		"yaw_rate": yaw_rate,
		"sideslip_rad": sideslip,
		"body_roll_rad": body_roll,
		"lean": lean,
		"lean_rate": lean_rate,
		"road_bank_rad": road_bank_rad,
		"lean_relative_road_rad": lean_relative_road_rad,
		"gravity_forward_m_s2": gravity_forward_m_s2,
		"gravity_right_m_s2": gravity_right_m_s2,
		"gravity_normal_m_s2": gravity_normal_m_s2,
		"steering": steering,
		"gear": gear,
		"rpm": rpm,
		"crashed": crashed,
		"crash_reason": crash_reason,
		"collision_contact": collision_contact.duplicate(true),
		"on_track": on_track,
		"surface_material": surface_material,
		"surface_friction": surface_friction,
		"surface_rolling_coefficient": surface_rolling_coefficient,
		"longitudinal_acceleration": longitudinal_acceleration,
		"lateral_acceleration": lateral_acceleration,
		"wheel_loads_n": wheel_loads_n.duplicate(),
		"wheel_longitudinal_n": wheel_longitudinal_n.duplicate(),
		"wheel_lateral_n": wheel_lateral_n.duplicate(),
		"front_slip_angle_rad": front_slip_angle,
		"rear_slip_angle_rad": rear_slip_angle,
		"front_load_n": front_load_n,
		"rear_load_n": rear_load_n,
		"front_force_n": front_force_n,
		"rear_force_n": rear_force_n,
		"lateral_force_n": lateral_force_n,
		"front_lateral_force_n": front_lateral_force_n,
		"rear_lateral_force_n": rear_lateral_force_n,
		"grip_utilization": grip_utilization,
		"auto_shift": auto_shift,
		"automatic_launch_clutch": true,
		"throttle_applied": throttle_applied,
		"brake_applied": brake_applied,
		"front_brake_applied": front_brake_applied,
		"rear_brake_applied": rear_brake_applied,
		"shift_remaining": shift_remaining,
		"last_shift_command": last_shift_command,
		"requested_controls": last_controls.duplicate(true),
		"unsupported":
		[
			"suspension_dynamics",
			"wheel_spin_state",
			"tire_relaxation_length",
			"tire_temperature",
			"camber_thrust",
			"aero_downforce",
			"differential_yaw_moment",
			"airborne_motion",
			"crash_contacts",
			"surface_curvature_normal_load",
		],
	}
