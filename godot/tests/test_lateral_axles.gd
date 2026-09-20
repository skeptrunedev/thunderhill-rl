extends SceneTree

const Motorcycle = preload("res://scripts/motorcycle.gd")
const DT := 1.0 / 120.0
const ROAD := {"height": 0.0, "normal": Vector3.UP, "on_track": true}
var failures := 0
var cases := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func _initialize() -> void:
	# Different CG locations, axle braking and mirrored turns. Force requirements
	# follow Newton force/moment balance, not the implementation helper functions.
	for fraction: float in [0.35, 0.5, 0.65]:
		for brakes: Vector2 in [
			Vector2.ZERO, Vector2(0.3, 0), Vector2(0, 0.3), Vector2(1, 0), Vector2(0, 1)
		]:
			for turn: float in [-0.2, -0.003, 0.003, 0.2]:
				var bike := Motorcycle.new()
				bike.parameters.static_front_fraction = fraction
				bike.parameters.cg_height_m = 0.4  # Keep both contacts supported under full front braking.
				bike.parameters.drag_area_m2 = 0.0
				bike.parameters.rolling_coefficient = 0.0
				bike.parameters.engine_brake_torque_nm = 0.0
				bike.reset(Vector3.ZERO, 0.0, 20.0)
				bike.steering = turn
				bike.front_brake_applied = brakes.x
				bike.rear_brake_applied = brakes.y
				var result: Dictionary = bike.step(
					DT,
					{
						"assist_enabled": false,
						"steer": turn / float(bike.parameters.steering_limit_rad),
						"front_brake": brakes.x,
						"rear_brake": brakes.y
					},
					ROAD
				)
				check(not result.has("error"), "Valid contact rejected")
				var a: float = (1.0 - fraction) * bike.parameters.wheelbase_m
				var b: float = fraction * bike.parameters.wheelbase_m
				var fyf: float = bike.front_lateral_force_n
				var fyr: float = bike.rear_lateral_force_n
				check(absf(a * fyf - b * fyr) < 0.00001, "Steady yaw moment is nonzero")
				check(
					absf(fyf + fyr - bike.lateral_force_n) < 0.00001,
					"Axle lateral forces do not sum"
				)
				check(
					absf((fyf + fyr) / bike.mass_kg() - bike.lateral_acceleration) < 0.00001,
					"Newton lateral force balance failed"
				)
				var mu: float = bike.parameters.asphalt_friction
				var cf := sqrt(
					maxf(0.0, pow(mu * bike.front_load_n, 2) - pow(bike.front_force_n, 2))
				)
				var cr := sqrt(maxf(0.0, pow(mu * bike.rear_load_n, 2) - pow(bike.rear_force_n, 2)))
				check(
					(
						pow(fyf, 2) + pow(bike.front_force_n, 2)
						<= pow(mu * bike.front_load_n, 2) + 0.0001
					),
					"Front friction circle exceeded"
				)
				check(
					(
						pow(fyr, 2) + pow(bike.rear_force_n, 2)
						<= pow(mu * bike.rear_load_n, 2) + 0.0001
					),
					"Rear friction circle exceeded"
				)
				var demand: float = bike.mass_kg() * 400.0 * tan(turn) / bike.parameters.wheelbase_m
				if absf(demand) * fraction <= cf and absf(demand) * (1.0 - fraction) <= cr:
					check(
						absf(bike.lateral_force_n - demand) < 0.0001,
						"Feasible steering demand was reduced"
					)
				else:
					check(
						absf(absf(fyf) - cf) < 0.0001 or absf(absf(fyr) - cr) < 0.0001,
						"Neither axle reaches limiting lateral budget"
					)
				if brakes.x == 1.0 or brakes.y == 1.0:
					check(
						absf(bike.lateral_force_n) < 0.001,
						"Saturated brake still generates steady cornering"
					)
				if brakes == Vector2.ZERO and absf(turn) > 0.1:
					check(
						absf(absf(bike.lateral_force_n) - mu * bike.mass_kg() * 9.81) < 0.0001,
						"Unbraked flat grip should be mu*m*g"
					)
				var telemetry: Dictionary = bike.telemetry()
				check(
					(
						telemetry.front_lateral_force_n == fyf
						and telemetry.rear_lateral_force_n == fyr
					),
					"Telemetry omits axle force"
				)
				bike.reset(Vector3.ZERO, 0.0)
				check(
					(
						bike.front_lateral_force_n == 0.0
						and bike.rear_lateral_force_n == 0.0
						and bike.lateral_capacity_n == 0.0
					),
					"Reset retains old axle force"
				)
				cases += 1
	print(
		"LATERAL_AXLES_CHECK ",
		JSON.stringify({"ok": failures == 0, "failures": failures, "cases": cases})
	)
	quit(failures)
