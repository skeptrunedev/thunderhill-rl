extends SceneTree
const Benchmark = preload("res://scripts/control_benchmark.gd")
const Motorcycle = preload("res://scripts/motorcycle.gd")
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func write_trace(manifest: Dictionary, row: Dictionary = {}) -> String:
	var path := "user://benchmark-contract-fixture.jsonl"
	var file := FileAccess.open(path, FileAccess.WRITE)
	file.store_line(JSON.stringify(manifest))
	if not row.is_empty():
		file.store_line(JSON.stringify(row))
	file.close()
	return path


func _initialize() -> void:
	var sim := Motorcycle.new()
	var manifest := {
		"type": "episode",
		"track_sha256": "test-track",
		"physics_version": sim.MODEL_VERSION,
		"parameters": sim.parameters.duplicate(true),
		"physics_dt": 1.0 / 120.0,
		"initial_state": {"tick": 0}
	}
	for field in ["track_sha256", "physics_version", "parameters", "physics_dt", "initial_state"]:
		var bad := manifest.duplicate(true)
		match field:
			"track_sha256", "physics_version":
				bad[field] = "wrong"
			"parameters":
				bad[field]["bike_mass_kg"] = 1.0
			"physics_dt":
				bad[field] = 0.01
			"initial_state":
				bad[field]["tick"] = 1
		var loader := Benchmark.new()
		check(
			not loader.open_trace(write_trace(bad), sim, "test-track", 1.0 / 120.0).is_empty(),
			"Accepted incompatible " + field
		)
	for tick in [1, 3]:
		var row := {
			"type": "transition",
			"tick": tick,
			"previous_tick": tick - 1,
			"requested_controls": {"throttle": 0.5},
			"state": {"position": [0, 0, 0]}
		}
		var loader := Benchmark.new()
		check(
			(
				loader
				. open_trace(write_trace(manifest, row), sim, "test-track", 1.0 / 120.0)
				. is_empty()
			),
			"Rejected valid manifest"
		)
		var action: Dictionary = loader.next_action()
		check(action.has("error") == (tick != 1), "Tick continuity validation failed")
		if tick == 1:
			check(action.controls.throttle == 0.5, "Recorded controls changed")
			check(loader.next_action().is_empty(), "End of trace was not detected")
	var timing := Benchmark.new()
	timing.start_usec = 1000000
	timing.record_capture("gpu_readback", 1010000, 1025000)
	var event: Dictionary = timing.capture_events[0]
	check(event.start_ms == 10.0 and event.end_ms == 25.0, "Capture clock origin changed")
	check(event.duration_ms == 15.0, "Capture duration units changed")
	print("BENCHMARK_CONTRACT_CHECK failures=", failures)
	quit(failures)
