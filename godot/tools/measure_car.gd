extends SceneTree
## Performance envelope of the car dynamics (car.gd) on a flat, level asphalt
## sample, stepped exactly as main.gd steps it (1/120 s ticks). Measures the
## published MX-5 comparison points: top speed, 0 to 100 km/h, braking from
## 100 km/h, and steady state lateral acceleration on constant radius circles.
## The track and its grades are not involved; this is not a lap simulation.
##   godot --headless --path godot --script res://tools/measure_car.gd
const CarScript = preload("res://scripts/car.gd")
const DT := 1.0 / 120.0
const FLAT := {"height": 0.0, "normal": Vector3.UP, "on_track": true, "on_pavement": true}


func _initialize() -> void:
	var report := {"dynamics_version": CarScript.MODEL_VERSION}
	report.merge(_acceleration())
	report.merge(_braking())
	report["skidpad"] = [_skidpad(40.0), _skidpad(80.0), _skidpad(150.0)]
	print("CAR_MEASUREMENT ", JSON.stringify(report))
	quit(0)


func _acceleration() -> Dictionary:
	var car = CarScript.new()
	car.reset(Vector3.ZERO, 0.0, 0.0)
	var to_100 := -1.0
	var gears := {}
	for tick in 120 * 120:
		var result: Dictionary = car.step(DT, {"throttle": 1.0}, FLAT)
		if result.has("error"):
			return {"acceleration_error": result.error}
		gears[car.gear] = maxf(float(gears.get(car.gear, 0.0)), car.speed)
		if to_100 < 0.0 and car.speed >= 100.0 / 3.6:
			to_100 = car.elapsed
	return {
		"zero_to_100_kmh_s": to_100,
		"top_speed_m_s": car.speed,
		"top_speed_kmh": car.speed * 3.6,
		"top_speed_gear": car.gear,
		"top_speed_rpm": car.rpm,
		"max_speed_by_gear_m_s": gears,
	}


func _braking() -> Dictionary:
	var car = CarScript.new()
	car.reset(Vector3.ZERO, 0.0, 100.0 / 3.6)
	var start: Vector3 = car.position
	var peak := 0.0
	for tick in 120 * 10:
		car.step(DT, {"front_brake": 1.0, "rear_brake": 1.0}, FLAT)
		peak = maxf(peak, -car.longitudinal_acceleration / CarScript.GRAVITY)
		if car.speed <= 0.0:
			break
	return {
		"braking_100_to_0_m": car.position.distance_to(start),
		"braking_100_to_0_s": car.elapsed,
		"peak_braking_g": peak,
		"braking_stopped": car.speed <= 0.0,
	}


## Hold speed on a circle of the given radius with a steering and throttle
## controller, raising the target speed until the car can no longer hold the
## radius. Reports the highest steady lateral acceleration it held.
func _skidpad(radius: float) -> Dictionary:
	var best := {"radius_m": radius, "lateral_g": 0.0}
	var target := 8.0
	while target < 60.0:
		var held := _hold_circle(radius, target)
		if not held.held:
			break
		best = {
			"radius_m": radius,
			"speed_m_s": held.speed,
			"lateral_g": held.lateral_g,
			"steering_rad": held.steering,
			"sideslip_rad": held.sideslip,
			"front_slip_rad": held.front_slip,
			"rear_slip_rad": held.rear_slip,
		}
		target += 0.25
	return best


func _hold_circle(radius: float, target: float) -> Dictionary:
	var car = CarScript.new()
	car.reset(Vector3.ZERO, 0.0, target)
	var wheelbase := float(car.parameters.wheelbase_m)
	var limit := float(car.parameters.steering_limit_rad)
	var samples: Array[float] = []
	var integral := 0.0
	var throttle_integral := 0.0
	for tick in 120 * 12:
		# Feed forward kinematic steer plus PI yaw rate feedback onto v / R;
		# the integral absorbs the car's understeer.
		var yaw_target: float = car.speed / radius
		var error: float = yaw_target - car.yaw_rate
		integral = clampf(integral + 1.5 * error * DT, -0.3, 0.3)
		var steer: float = atan(wheelbase / radius) + 0.8 * error + integral
		throttle_integral = clampf(throttle_integral + 0.5 * (target - car.speed) * DT, -0.5, 1.0)
		var throttle: float = clampf(0.1 + 0.5 * (target - car.speed) + throttle_integral, 0.0, 1.0)
		car.step(DT, {"steer": clampf(steer / limit, -1.0, 1.0), "throttle": throttle}, FLAT)
		if car.crashed:
			return {"held": false}
		if tick >= 120 * 8:
			samples.append(car.lateral_acceleration)
	var mean := 0.0
	for value in samples:
		mean += value / samples.size()
	var achieved: float = car.speed * car.speed / radius
	var held: bool = (
		absf(car.speed - target) < 0.1
		and absf(car.yaw_rate - car.speed / radius) < 0.01 * car.speed / radius + 0.005
		and absf(mean - achieved) < 0.05 * achieved
	)
	return {
		"held": held,
		"speed": car.speed,
		"lateral_g": mean / CarScript.GRAVITY,
		"steering": car.steering,
		"sideslip": car.sideslip,
		"front_slip": car.front_slip_angle,
		"rear_slip": car.rear_slip_angle,
	}
