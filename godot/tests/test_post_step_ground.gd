extends SceneTree
const Motorcycle = preload("res://scripts/motorcycle.gd")
var failures := 0


func _initialize() -> void:
	var sim := Motorcycle.new()
	sim.reset(Vector3.ZERO, 0.0)
	var before: Dictionary = sim.telemetry().duplicate(true)
	var road := {"height": 0.0, "normal": Vector3.UP, "on_track": true}
	var result: Dictionary = sim.step(
		1.0 / 120.0,
		{"throttle": 1.0},
		road,
		func() -> Dictionary:
			return {
				"error": "Injected missing destination triangle", "failure_type": "infrastructure"
			}
	)
	if not result.has("error") or sim.telemetry() != before:
		failures += 1
		push_error("Invalid destination ground was accepted or mutated simulator state")
	result = sim.step(1.0 / 120.0, {"throttle": 1.0}, road, func() -> Dictionary: return road)
	if result.has("error") or sim.tick != 1 or result.get("ground_after") != road:
		failures += 1
		push_error("Valid destination ground failed to advance")
	print("POST_STEP_GROUND_CHECK failures=", failures)
	quit(failures)
