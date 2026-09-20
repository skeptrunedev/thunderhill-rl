extends SceneTree

const Motorcycle = preload("res://scripts/motorcycle.gd")
const DT := 1.0 / 120.0
const G := 9.81
const FLAT := {"height": 0.0, "normal": Vector3.UP, "on_track": true}
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func bike_at(velocity: float = 0.0) -> RefCounted:
	var bike := Motorcycle.new()
	bike.parameters.drag_area_m2 = 0.0
	bike.parameters.rolling_coefficient = 0.0
	bike.parameters.engine_brake_torque_nm = 0.0
	bike.reset(Vector3.ZERO, 0.0, absf(velocity))
	bike.longitudinal_velocity = velocity
	return bike


func _initialize() -> void:
	var model := bike_at()
	var weight: float = model.mass_kg() * G
	var mu: float = model.parameters.asphalt_friction
	var q: float = model.parameters.cg_height_m / model.parameters.wheelbase_m
	var front_fraction: float = model.parameters.static_front_fraction
	var front_expected := mu * front_fraction * weight / (1.0 - mu * q)
	var rear_expected := mu * (1.0 - front_fraction) * weight / (1.0 + mu * q)
	var front: Dictionary = model._contact_balance(weight, mu, -4400.0, 0.0)
	var rear: Dictionary = model._contact_balance(weight, mu, 0.0, -1800.0)
	check(not front.has("error") and not rear.has("error"), "Analytic brake cases rejected")
	check(
		absf(float(front.front_force) + front_expected) < 1e-6,
		"Front brake differs from closed form"
	)
	check(
		absf(float(rear.rear_force) + rear_expected) < 1e-6, "Rear brake differs from closed form"
	)
	for result: Dictionary in [front, rear]:
		check(
			(
				absf(
					(
						float(result.front_load)
						- front_fraction * weight
						+ q * (float(result.front_force) + float(result.rear_force))
					)
				)
				< 1e-6
			),
			"Pitch moment residual nonzero"
		)
	var a := bike_at(20.0)
	var b := bike_at(20.0)
	a.longitudinal_acceleration = -100.0
	b.longitudinal_acceleration = 100.0
	a.step(DT, {"front_brake": 1.0}, FLAT)
	b.step(DT, {"front_brake": 1.0}, FLAT)
	check(a.telemetry() == b.telemetry(), "Previous acceleration still changes force solution")
	for slope: float in [-0.15, 0.0, 0.15]:
		var parked := bike_at()
		parked.front_brake_applied = 1.0
		parked.rear_brake_applied = 1.0
		var road := {"height": 0.0, "normal": Vector3(0, cos(slope), sin(slope)), "on_track": true}
		var held: Dictionary = parked.step(DT, {"front_brake": 1.0, "rear_brake": 1.0}, road)
		check(not held.has("error"), "Supported parked incline failed")
		check(
			parked.speed == 0.0 and parked.position == Vector3.ZERO,
			"Brakes accelerated parked motorcycle"
		)
		check(parked.longitudinal_acceleration == 0.0, "Parked acceleration must be zero")
		check(
			(
				absf(parked.front_force_n + parked.rear_force_n - parked.mass_kg() * G * sin(slope))
				< 0.001
			),
			"Static brake reaction does not balance grade"
		)
		var free := bike_at()
		free.step(DT, {}, road)
		check(
			absf(free.longitudinal_velocity + G * sin(slope) * DT) < 1e-6,
			"Unbraked incline must roll under gravity"
		)
		check(
			absf(free.speed - absf(free.longitudinal_velocity)) < 1e-10,
			"Speed magnitude inconsistent with velocity"
		)
	# Constant unsaturated braking from 0.01 m/s stops inside this tick. Compare
	# displacement to v²/(2a), independently of the integrator's time partition.
	var stopping := bike_at(0.01)
	stopping.front_brake_applied = 0.2
	var deceleration: float = 0.2 * stopping.parameters.front_brake_capacity_n / stopping.mass_kg()
	stopping.step(DT, {"front_brake": 0.2}, FLAT)
	check(stopping.speed == 0.0, "Stop event left nonzero speed")
	check(
		absf(-stopping.position.z - 0.01 * 0.01 / (2.0 * deceleration)) < 1e-9,
		"Stop distance differs from kinematics"
	)
	check(
		stopping.longitudinal_acceleration == 0.0 and absf(stopping.front_force_n) < 1e-6,
		"Stopped telemetry retained braking force"
	)
	var backward := bike_at(-0.01)
	backward.front_brake_applied = 0.2
	backward.step(DT, {"front_brake": 0.2}, FLAT)
	check(backward.speed == 0.0 and backward.position.z > 0.0, "Brake did not oppose rollback")
	# At very high grip and braking demand, both tire forces demand a pitching
	# moment that two nonnegative normal reactions cannot balance.
	var unsupported := bike_at(10.0)
	unsupported.parameters.asphalt_friction = 3.0
	unsupported.parameters.front_brake_capacity_n = 100000.0
	unsupported.parameters.rear_brake_capacity_n = 100000.0
	unsupported.front_brake_applied = 1.0
	unsupported.rear_brake_applied = 1.0
	var before: Dictionary = unsupported.telemetry()
	var failure: Dictionary = unsupported.step(DT, {"front_brake": 1.0, "rear_brake": 1.0}, FLAT)
	check(
		failure.get("failure_type", "") == "unsupported_dynamics",
		"Infeasible contact was silently clamped"
	)
	check(unsupported.telemetry() == before, "Unsupported contact partially mutated state")
	print(
		"LONGITUDINAL_CHECK ",
		JSON.stringify(
			{
				"ok": failures == 0,
				"failures": failures,
				"front_brake_n": front_expected,
				"rear_brake_n": rear_expected
			}
		)
	)
	quit(failures)
