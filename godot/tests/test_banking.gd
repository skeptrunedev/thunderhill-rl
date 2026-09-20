extends SceneTree

const Motorcycle = preload("res://scripts/motorcycle.gd")
const DT := 1.0 / 120.0
const G := 9.81
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func clean_bike() -> RefCounted:
	var bike := Motorcycle.new()
	bike.parameters.drag_area_m2 = 0.0
	bike.parameters.rolling_coefficient = 0.0
	bike.parameters.engine_brake_torque_nm = 0.0
	bike.reset(Vector3.ZERO, 0.0, 20.0)
	return bike


func _initialize() -> void:
	for sign_value: float in [-1.0, 1.0]:
		var bank := sign_value * 0.2
		var normal := Vector3(sin(bank), cos(bank), 0)
		var road := {"height": 0.0, "normal": normal, "on_track": true}
		var bike := clean_bike()
		bike.step(DT, {"assist_enabled": false}, road)
		# Independent force balance for a straight contour line on an inclined plane:
		# normal support mg cos(bank), uphill static friction -mg sin(bank).
		check(
			absf(bike.front_load_n + bike.rear_load_n - bike.mass_kg() * G * cos(bank)) < 0.001,
			"Incline normal support must be mg cos(bank)"
		)
		check(
			absf(bike.lateral_force_n + bike.mass_kg() * G * sin(bank)) < 0.001,
			"Contour riding needs uphill friction"
		)
		check(
			absf(bike.lateral_acceleration) < 0.00001,
			"Static friction must cancel downslope gravity"
		)
		check(
			absf(bike.lean_rate) < 0.00001,
			"World upright bike should balance straight contour riding"
		)
		check(
			absf(bike.lean_relative_road_rad + bank) < 0.00001,
			"World lean and road relative lean confused"
		)

		# With no friction, Newton's second law gives downslope g sin(bank),
		# independent of steer command. Contact normal alone must not add grip.
		var slippery := clean_bike()
		slippery.parameters.asphalt_friction = 0.0
		slippery.step(DT, {"assist_enabled": false}, road)
		check(absf(slippery.lateral_force_n) < 0.00001, "Frictionless surface generated tire force")
		check(
			absf(slippery.lateral_acceleration - G * sin(bank)) < 0.00001,
			"Frictionless slide differs from inclined plane solution"
		)
		var expected_heading := G * sin(bank) / 20.0 * cos(bank) * DT
		check(
			absf(slippery.heading - expected_heading) < 0.00001,
			"Surface turning rate incorrectly converted to world azimuth"
		)

		# Instantaneous equilibrium derives from a zero moment about road contact.
		# tan(relative lean) = (a_right - g sin(bank)) / (g cos(bank)).
		var equilibrium := clean_bike()
		var acceleration := sign_value * 3.0
		var relative := atan2(acceleration - G * sin(bank), G * cos(bank))
		equilibrium.lean = bank + relative
		equilibrium.steering = atan(
			acceleration * float(equilibrium.parameters.wheelbase_m) / 400.0
		)
		equilibrium.step(
			DT,
			{
				"assist_enabled": false,
				"steer": equilibrium.steering / float(equilibrium.parameters.steering_limit_rad)
			},
			road
		)
		check(
			absf(equilibrium.lateral_acceleration - acceleration) < 0.00001,
			"Equilibrium acceleration differs"
		)
		check(
			absf(equilibrium.lean_rate) < 0.00001,
			"Banked equilibrium has spurious roll acceleration"
		)
		check(
			(
				absf(
					(
						equilibrium.lateral_force_n
						- equilibrium.mass_kg() * (acceleration - G * sin(bank))
					)
				)
				< 0.001
			),
			"Tire demand incorrectly includes gravity"
		)

		var assisted := clean_bike()
		for tick in 120:
			assisted.step(DT, {"steer": 0.0}, road)
		check(
			absf(assisted.lean) < 0.0001 and absf(assisted.heading) < 0.0001,
			"Rider must preserve world upright straight line on mirrored banks"
		)

	# Combined grade and cross slope must preserve requested horizontal heading.
	var combined := clean_bike()
	combined.heading = 0.4
	var n := Vector3(0.2, 1.0, 0.15).normalized()
	var tangent: Vector3 = combined.surface_forward(n)
	check(absf(tangent.dot(n)) < 0.00001, "Forward vector is not tangent to road")
	check(
		absf(atan2(tangent.x, -tangent.z) - combined.heading) < 0.00001,
		"Surface projection changed heading"
	)
	print("BANKING_CHECK ", JSON.stringify({"ok": failures == 0, "failures": failures}))
	quit(failures)
