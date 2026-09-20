extends SceneTree
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func _initialize() -> void:
	run.call_deferred()


func run() -> void:
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	await process_frame
	var initial: Dictionary = game.serializable(game.sim.telemetry())
	initial.speed = 7.0
	initial.longitudinal_velocity = 7.0
	initial.gear = 2
	var previous_episode: int = game.episode_number
	var observation: Dictionary = game.reset_episode(0.0, "diagnostic-recorded-inputs", initial)
	check(game.episode_number == previous_episode + 1, "Reset created duplicate episodes")
	game.recorder.flush()
	var lines := FileAccess.get_file_as_string(game.recorder.get_path()).split("\n", false)
	check(lines.size() == 1, "Reset did not create exactly one manifest")
	var manifest: Dictionary = JSON.parse_string(lines[0])
	check(
		manifest.initial_state == JSON.parse_string(JSON.stringify(initial)),
		"Manifest differs from supplied initial state"
	)
	check(
		manifest.initial_state == JSON.parse_string(JSON.stringify(observation.state)),
		"Manifest differs from live simulation"
	)
	var transition: Dictionary = game._step({"throttle": 0.0})
	check(not transition.has("error"), "Initial state could not advance")
	check(
		transition.previous_tick == manifest.initial_state.tick,
		"First transition lost initial tick"
	)
	game.reset_episode(0.0)
	game.recorder.flush()
	var ordinary: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string(game.recorder.get_path()).split("\n", false)[0]
	)
	check(ordinary.initial_state.speed == 0.0, "Human reset retained benchmark initial state")
	check(
		(
			ordinary.initial_state
			== JSON.parse_string(JSON.stringify(game.serializable(game.sim.telemetry())))
		),
		"Human manifest differs from live state"
	)
	print("EPISODE_MANIFEST_CHECK failures=", failures)
	game.queue_free()
	await process_frame
	quit(failures)
