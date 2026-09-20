extends SceneTree
const Motorcycle = preload("res://scripts/motorcycle.gd")
const DT := 1.0 / 120.0
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func balance(bike: RefCounted) -> void:
	var mu: float = bike.parameters.asphalt_friction
	var weight: float = bike.mass_kg() * bike.gravity_normal_m_s2
	var fraction: float = bike.parameters.static_front_fraction
	var length: float = bike.parameters.wheelbase_m
	check(
		(
			absf(
				(
					bike.front_load_n
					- fraction * weight
					+ (
						bike.parameters.cg_height_m
						/ length
						* (bike.front_force_n + bike.rear_force_n)
					)
				)
			)
			< 0.00001
		),
		"Pitch balance violated by assistance"
	)
	check(
		(
			absf(
				(
					(1.0 - fraction) * length * bike.front_lateral_force_n
					- fraction * length * bike.rear_lateral_force_n
				)
			)
			< 0.00001
		),
		"Yaw balance violated by assistance"
	)
	check(
		(
			pow(bike.front_force_n, 2) + pow(bike.front_lateral_force_n, 2)
			<= pow(mu * bike.front_load_n, 2) + 0.0001
		),
		"Assisted front circle exceeded"
	)
	check(
		(
			pow(bike.rear_force_n, 2) + pow(bike.rear_lateral_force_n, 2)
			<= pow(mu * bike.rear_load_n, 2) + 0.0001
		),
		"Assisted rear circle exceeded"
	)


func _initialize() -> void:
	# Reproduce the recorded start bank magnitude (about 0.018 radians), mirrored
	# and magnified. Full throttle with zero steering must preserve upright riding.
	for bank: float in [-0.08, -0.018, 0.018, 0.08]:
		var bike := Motorcycle.new()
		bike.reset(Vector3.ZERO, 0.0)
		var road := {"height": 0.0, "normal": Vector3(sin(bank), cos(bank), 0), "on_track": true}
		var interventions := 0
		for i in 600:
			var result: Dictionary = bike.step(DT, {"throttle": 1.0, "steer": 0.0}, road)
			check(not result.has("error") and not bike.crashed, "Full throttle bank launch failed")
			if bike.longitudinal_force_scale < 0.999999:
				interventions += 1
			balance(bike)
		check(interventions > 0, "Launch never reserved bank grip")
		check(
			bike.speed > 20.0 and absf(bike.lean) < 0.001 and absf(bike.heading) < 0.001,
			"Bank launch failed to accelerate upright and straight"
		)
		for i in 180:
			var result: Dictionary = bike.step(DT, {"front_brake": 1.0, "rear_brake": 1.0}, road)
			check(not result.has("error") and not bike.crashed, "Assisted bank braking failed")
			balance(bike)
	# Independent closed form at zero CG height: a rear-only longitudinal force
	# must reduce to sqrt((mu*N/2)^2 - (Fy/2)^2) under the reserved force.
	var bike := Motorcycle.new()
	bike.parameters.cg_height_m = 0.0
	bike.requested_lateral_force_n = 800.0
	var result: Dictionary = bike._reserve_lateral(
		{"front_load": 1000.0, "front_force": 0.0, "rear_force": 1000.0}, 2000.0, 1.0
	)
	check(
		absf(result.rear_force - sqrt(1000000.0 - 160000.0)) < 0.00001,
		"Combined allocation differs from independent circle solution"
	)
	bike.requested_lateral_force_n = 0.0
	result = bike._reserve_lateral(
		{"front_load": 1000.0, "front_force": -1000.0, "rear_force": -500.0}, 2000.0, 1.0
	)
	check(
		result.front_force == -1000.0 and result.rear_force == -500.0,
		"No lateral request must preserve braking"
	)
	bike.longitudinal_force_scale = 0.2
	var held := {"front_load": 1000.0, "front_force": 0.0, "rear_force": 0.0, "held": true}
	check(
		bike._reserve_lateral(held, 2000.0, 1.0) == held and bike.longitudinal_force_scale == 1.0,
		"Held reaction and intervention telemetry must remain coherent"
	)
	print("COMBINED_ASSIST_CHECK ", JSON.stringify({"ok": failures == 0, "failures": failures}))
	quit(failures)
