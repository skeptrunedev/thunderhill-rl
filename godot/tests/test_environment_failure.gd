extends SceneTree


class FailedMotorcycle:
	extends MotorcycleSim

	func step(_dt: float, _controls: Dictionary, _road: Dictionary) -> Dictionary:
		return {
			"error": "Injected unsupported contact state", "failure_type": "unsupported_dynamics"
		}


var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	run.call_deferred()


func run() -> void:
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	await process_frame
	game.sim = FailedMotorcycle.new()
	game.reset_episode(0.0)
	game.paused = false
	var result: Dictionary = game._step({"throttle": 0.5})
	check(result.get("failure_type") == "unsupported_dynamics", "Failure classification lost")
	check(game.paused, "Human game did not stop at invalid dynamics")
	check(game.sim.tick == 0, "Failed step advanced time")
	check(not game.observation().rollout_valid, "Failed rollout is marked valid")
	check(
		game.observation().truncated and not game.observation().terminated,
		"Model limit treated as task termination"
	)
	check(not result.has("reward_components"), "Model failure assigned reward")
	game.menu_action("ride")
	check(game.paused, "Ride resumed invalid dynamics")
	game.reset_episode(0.0)
	check(game.environment_failure.is_empty(), "Reset retained previous failure")
	check(game.observation().rollout_valid, "Reset did not restore valid episode")
	game.agent_mode = true
	var command := {
		"op": "advance",
		"episode_id": game.episode_id,
		"expected_tick": 0,
		"action_id": "failing-action",
		"controls": {"throttle": 0.5}
	}
	var failed: Dictionary = game._request(command)
	check(
		failed.get("failure_type") == "unsupported_dynamics",
		"Agent lost model failure classification"
	)
	check(failed == game._request(command), "Failed action retry was not idempotent")
	check(failed.transitions.is_empty(), "Failed step became a normal transition")
	check(game.sim.tick == 0, "Failed retry advanced physics")
	game.recorder.flush()
	var records := FileAccess.get_file_as_string(game.recorder.get_path()).split("\n", false)
	var failures_recorded := 0
	for line in records:
		var row: Dictionary = JSON.parse_string(line)
		if row.get("type") == "environment_failure":
			failures_recorded += 1
	check(failures_recorded == 1, "Failure was absent or duplicated in recording")
	print("ENVIRONMENT_FAILURE_CHECK failures=", failures)
	game.queue_free()
	await process_frame
	quit(failures)
