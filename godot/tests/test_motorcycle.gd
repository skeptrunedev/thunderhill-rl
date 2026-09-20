extends SceneTree

const Motorcycle = preload("res://scripts/motorcycle.gd")
const ROAD := {"height": 0.0, "normal": Vector3.UP, "on_track": true}
const DT := 1.0 / 120.0
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func advance(bike: RefCounted, controls: Dictionary, count: int) -> void:
	for i in count:
		var result: Dictionary = bike.step(DT, controls, ROAD)
		check(not result.has("error"), "Valid step rejected")
		check(is_finite(bike.speed) and is_finite(bike.lean), "Nonfinite state")
		check(bike.speed >= 0.0, "Unexpected reversing")
		check(
			absf(bike.front_load_n + bike.rear_load_n - bike.mass_kg() * 9.81) < 0.001,
			"Normal loads do not sum to weight"
		)
		var mu: float = bike.parameters.asphalt_friction
		var remaining: float = (
			sqrt(maxf(0.0, pow(mu * bike.front_load_n, 2) - pow(bike.front_force_n, 2)))
			+ sqrt(maxf(0.0, pow(mu * bike.rear_load_n, 2) - pow(bike.rear_force_n, 2)))
		)
		check(absf(bike.lateral_force_n) <= remaining + 0.001, "Combined tire budget exceeded")


func _initialize() -> void:
	var bike := Motorcycle.new()
	bike.reset(Vector3.ZERO, 0.0, 10.0)
	advance(bike, {"throttle": 0.5}, 240)
	check(bike.speed > 10.0, "Throttle did not accelerate")
	check(bike.front_load_n < bike.rear_load_n, "Drive should unload front axle")
	var recorded: Dictionary = bike.telemetry()
	bike.reset(Vector3.ZERO, 0.0, 10.0)
	advance(bike, {"throttle": 0.5}, 240)
	check(bike.telemetry() == recorded, "Reset sequence was not exactly repeatable")
	var start_speed: float = bike.speed
	advance(bike, {"front_brake": 0.7, "rear_brake": 0.2}, 120)
	check(bike.speed < start_speed, "Brakes did not reduce speed")
	check(bike.front_load_n > bike.rear_load_n, "Braking should load front axle")
	advance(bike, {"front_brake": 1.0, "rear_brake": 1.0}, 600)
	check(bike.speed == 0.0, "Full braking should stop without reversing")
	var before: Dictionary = bike.telemetry()
	for controls: Dictionary in [
		{"throttle": -0.1},
		{"steer": 1.01},
		{"shift": 0.5},
		{"throttle": NAN},
		{"front_brake": "full"},
		{"unknown": 1}
	]:
		check(bike.step(DT, controls, ROAD).has("error"), "Invalid controls accepted")
		check(before == bike.telemetry(), "Rejected controls modified state")
	check(bike.step(-DT, {}, ROAD).has("error"), "Negative dt accepted")
	for invalid_road: Dictionary in [
		{},
		{"height": NAN, "normal": Vector3.UP, "on_track": true},
		{"height": 0.0, "normal": Vector3.ZERO, "on_track": true}
	]:
		check(bike.step(DT, {}, invalid_road).has("error"), "Invalid road sample accepted")
		check(before == bike.telemetry(), "Rejected road sample modified state")
	bike.reset(Vector3.ZERO, 0.0, 30.0)
	bike.step(DT, {"steer": 0.35}, ROAD)
	check(
		bike.steering < 0.0 and bike.lean_rate > 0.0,
		"Right lean must initiate with left countersteer"
	)
	advance(bike, {"steer": 0.35, "throttle": 0.1}, 360)
	check(not bike.crashed and bike.lean > 0.1 and bike.heading > 0.0, "Assisted right turn failed")
	print("turn_result ", bike.telemetry())
	bike.reset(Vector3.ZERO, 0.0, 30.0)
	advance(bike, {"steer": 1.0, "assist_enabled": false}, 240)
	check(bike.crashed, "Unbalanced direct steering should fall")
	bike.reset(Vector3.ZERO, 0.0, 20.0)
	for i in 120:
		bike.step(
			DT,
			{"front_brake": 1.0, "rear_brake": 1.0},
			{"height": 0.0, "normal": Vector3.UP, "on_track": false}
		)
		check(
			(
				absf(bike.front_force_n)
				<= float(bike.parameters.offtrack_friction) * bike.front_load_n + 0.001
			),
			"Offtrack front grip exceeded"
		)
		check(
			(
				absf(bike.rear_force_n)
				<= float(bike.parameters.offtrack_friction) * bike.rear_load_n + 0.001
			),
			"Offtrack rear grip exceeded"
		)
	check(not bike.on_track and bike.speed > 0.0, "Offtrack should have lower braking grip")
	var uphill := Motorcycle.new()
	var downhill := Motorcycle.new()
	uphill.reset(Vector3.ZERO, 0.0, 20.0)
	downhill.reset(Vector3.ZERO, 0.0, 20.0)
	for i in 120:
		uphill.step(
			DT,
			{"throttle": 0.2},
			{
				"height": -uphill.position.z * 0.1,
				"normal": Vector3(0, 1, 0.1).normalized(),
				"on_track": true
			}
		)
		downhill.step(
			DT,
			{"throttle": 0.2},
			{
				"height": downhill.position.z * 0.1,
				"normal": Vector3(0, 1, -0.1).normalized(),
				"on_track": true
			}
		)
	check(uphill.speed < downhill.speed, "Gravity must slow uphill versus downhill")
	check(uphill.position.y > 0.0 and downhill.position.y < 0.0, "Road tangent slope sign is wrong")
	print("Motorcycle dynamics checks, failures: ", failures)
	quit(failures)
