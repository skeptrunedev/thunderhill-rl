extends SceneTree

const Motorcycle = preload("res://scripts/motorcycle.gd")
const DT := 1.0 / 120.0
const G := 9.81
const COEFFICIENTS := {
	"asphalt": [0.85, 0.01],
	"curb": [0.55, 0.03],
	"offroad": [0.25, 0.09],
}
var failures := 0
var checks := 0


func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)


func bike_at() -> RefCounted:
	var bike := Motorcycle.new()
	bike.parameters.drag_area_m2 = 0.0
	bike.parameters.engine_brake_torque_nm = 0.0
	bike.parameters.front_brake_capacity_n = 10000.0
	bike.parameters.asphalt_friction = COEFFICIENTS.asphalt[0]
	bike.parameters.rolling_coefficient = COEFFICIENTS.asphalt[1]
	bike.parameters.curb_friction = COEFFICIENTS.curb[0]
	bike.parameters.curb_rolling_coefficient = COEFFICIENTS.curb[1]
	bike.parameters.offtrack_friction = COEFFICIENTS.offroad[0]
	bike.parameters.offtrack_rolling_coefficient = COEFFICIENTS.offroad[1]
	bike.reset(Vector3.ZERO, 0.0, 20.0)
	bike.front_brake_applied = 1.0
	return bike


func road(legal: bool, curb: bool = false) -> Dictionary:
	return {"height": 0.0, "normal": Vector3.UP, "on_track": legal, "on_curb": curb}


func state(bike: RefCounted) -> Dictionary:
	# Include actuators and internal state, not just the public telemetry fields.
	var snapshot: Dictionary = {}
	for property: Dictionary in bike.get_property_list():
		if int(property.usage) & PROPERTY_USAGE_SCRIPT_VARIABLE:
			var value: Variant = bike.get(property.name)
			snapshot[property.name] = (
				value.duplicate(true) if value is Dictionary or value is Array else value
			)
	return snapshot


func expect_material(bike: RefCounted, material: String, mu: float, rolling: float) -> void:
	var data: Dictionary = bike.telemetry()
	check(data.get("surface_material") == material, "Wrong material: expected %s" % material)
	check(data.get("surface_friction") == mu, "Wrong friction for %s" % material)
	check(data.get("surface_rolling_coefficient") == rolling, "Wrong rolling for %s" % material)


func braking_step(bike: RefCounted, sample: Dictionary, material: String) -> void:
	var mu: float = COEFFICIENTS[material][0]
	var rolling: float = COEFFICIENTS[material][1]
	var mass: float = bike.mass_kg()
	var q: float = bike.parameters.cg_height_m / bike.parameters.wheelbase_m
	var front_fraction: float = bike.parameters.static_front_fraction
	# Saturated front braking alone: F = mu*(f*m*g + q*F).
	# Rolling resistance acts through COM and therefore adds no pitch moment.
	var expected_force := mu * front_fraction * mass * G / (1.0 - mu * q)
	var expected_acceleration := -expected_force / mass - rolling * G
	var old_velocity: float = bike.longitudinal_velocity
	var result: Dictionary = bike.step(DT, {"front_brake": 1.0}, sample)
	check(not result.has("error"), "Valid %s contact rejected" % material)
	check(absf(bike.front_force_n + expected_force) < 1e-6, "Wrong front force for %s" % material)
	check(absf(bike.rear_force_n) < 1e-6, "Unexpected rear force for %s" % material)
	check(
		absf(bike.longitudinal_acceleration - expected_acceleration) < 1e-8,
		"Wrong deceleration for %s" % material
	)
	check(
		absf(bike.longitudinal_velocity - old_velocity - expected_acceleration * DT) < 1e-8,
		"Wrong integrated velocity for %s" % material
	)
	check(bike.on_track == sample.on_track, "Material selection changed track legality")
	expect_material(bike, material, mu, rolling)


func expect_rejected(bike: RefCounted, sample: Dictionary, label: String) -> void:
	var before := state(bike)
	var result: Dictionary = bike.step(DT, {"throttle": 0.6, "steer": 0.2, "shift": 1}, sample)
	check(result.has("error"), "Invalid %s accepted" % label)
	# NaN compares unequal to itself. Exclude parameters from comparison because
	# they deliberately contain the invalid input and are not integration state.
	var after := state(bike)
	before.erase("parameters")
	after.erase("parameters")
	check(after == before, "Rejected %s partially mutated state" % label)


func _initialize() -> void:
	var fresh := Motorcycle.new()
	expect_material(fresh, "unknown", 0.0, 0.0)
	check(fresh.parameters.get("curb_friction") == 1.10, "Default curb friction changed")
	check(fresh.parameters.get("curb_rolling_coefficient") == 0.015, "Default curb rolling changed")
	for legal: bool in [true, false]:
		var bike := bike_at()
		braking_step(bike, road(legal, true), "curb")
		bike.reset(Vector3(1, 2, 3), 0.2, 8.0)
		expect_material(bike, "unknown", 0.0, 0.0)
		for explicit_flag: bool in [false, true]:
			var legacy := road(legal)
			if not explicit_flag:
				legacy.erase("on_curb")
			braking_step(bike_at(), legacy, "asphalt" if legal else "offroad")
	# Rendered pavement identity is independent of centerline based legality.
	for legal: bool in [true, false]:
		for paved: bool in [true, false]:
			var sample := road(legal)
			sample.on_pavement = paved
			braking_step(bike_at(), sample, "asphalt" if paved else "offroad")
			sample.on_curb = true
			braking_step(bike_at(), sample, "curb")
	for invalid: Variant in [null, 0, "true", NAN]:
		var invalid_pavement := road(true)
		invalid_pavement.on_pavement = invalid
		expect_rejected(bike_at(), invalid_pavement, "on_pavement")
	# Every force step must select its current contact, not retain the previous
	# material. Include a curb that is legal and one outside legal track bounds.
	var crossing := bike_at()
	for item: Array in [
		[road(true), "asphalt"],
		[road(false, true), "curb"],
		[road(false), "offroad"],
		[road(true, true), "curb"],
		[road(true), "asphalt"],
	]:
		braking_step(crossing, item[0], item[1])
	# Material telemetry describes the selected contact even after a crash.
	# Sliding deliberately uses fixed deceleration, not the selected tire grip.
	var sliding := bike_at()
	braking_step(sliding, road(true), "asphalt")
	sliding.crashed = true
	var before_slide: float = sliding.speed
	var slide_result: Dictionary = sliding.step(DT, {}, road(false, true))
	check(not slide_result.has("error"), "Valid crash slide rejected")
	expect_material(sliding, "curb", COEFFICIENTS.curb[0], COEFFICIENTS.curb[1])
	check(
		(
			absf(
				sliding.speed - before_slide + sliding.parameters.fall_slide_deceleration_m_s2 * DT
			)
			< 1e-9
		),
		"Crash slide did not use its fixed deceleration"
	)
	for invalid: Variant in [null, 0, 1, 0.0, "true", NAN, INF, [], {}]:
		var bad_flag := road(true)
		bad_flag.on_curb = invalid
		expect_rejected(crossing, bad_flag, "on_curb=%s" % str(invalid))
	for selection: Array in [
		[road(true), "asphalt_friction", "rolling_coefficient"],
		[road(true, true), "curb_friction", "curb_rolling_coefficient"],
		[road(false, true), "curb_friction", "curb_rolling_coefficient"],
		[road(false), "offtrack_friction", "offtrack_rolling_coefficient"],
	]:
		for key: String in [selection[1], selection[2]]:
			for invalid: Variant in [-0.01, NAN, INF, -INF, "0.5", true, null, [], {}]:
				var bike := bike_at()
				braking_step(bike, road(true), "asphalt")
				bike.parameters[key] = invalid
				expect_rejected(bike, selection[0], "%s=%s" % [key, str(invalid)])
	# Zero grip and rolling are valid endpoints, with no braking force.
	var zero := bike_at()
	zero.parameters.curb_friction = 0
	zero.parameters.curb_rolling_coefficient = 0
	var zero_result: Dictionary = zero.step(DT, {"front_brake": 1.0}, road(false, true))
	check(not zero_result.has("error"), "Zero material coefficients rejected")
	check(zero.speed == 20.0 and zero.front_force_n == 0.0, "Zero grip still applied braking")
	expect_material(zero, "curb", 0.0, 0.0)
	print(
		"SURFACE_GRIP_CHECK ",
		JSON.stringify({"ok": failures == 0, "checks": checks, "failures": failures})
	)
	quit(failures)
