extends SceneTree


class FailedMotorcycle:
	extends MotorcycleSim

	func step(
		_dt: float, _controls: Dictionary, _road: Dictionary, _after_step: Callable = Callable()
	) -> Dictionary:
		return {
			"error": "Injected unsupported contact state", "failure_type": "unsupported_dynamics"
		}


class InvalidDestinationTrack:
	extends Node3D
	var source: Node3D
	var simulator: RefCounted
	var initial_tick: int
	var invalid_samples := 0
	var integrated_state: Dictionary = {}

	func sample_world(point: Vector3) -> Dictionary:
		var sample: Dictionary = source.sample_world(point)
		# The starting sample remains real and valid. Inject the missing ground
		# only after the actual simulator has advanced, not via a fake step().
		if simulator.tick > initial_tick:
			invalid_samples += 1
			integrated_state = simulator.telemetry().duplicate(true)
			sample.height = NAN
		return sample


var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	run.call_deferred()


func simulator_state(simulator: RefCounted) -> Dictionary:
	var state: Dictionary = {}
	for property: Dictionary in simulator.get_property_list():
		if int(property.usage) & PROPERTY_USAGE_SCRIPT_VARIABLE:
			var value: Variant = simulator.get(property.name)
			state[property.name] = (
				value.duplicate(true) if value is Dictionary or value is Array else value
			)
	return state


func check_invalid_destination(game: Node3D) -> void:
	game.sim = MotorcycleSim.new()
	game.reset_episode(0.0)
	game.agent_mode = true
	var original_track: Node3D = game.track
	var injected := InvalidDestinationTrack.new()
	injected.source = original_track
	injected.simulator = game.sim
	injected.initial_tick = game.sim.tick
	game.track = injected
	var before := simulator_state(game.sim)
	var before_progress := [game.lap_time, game.last_progress, game.legal_distance, game.next_gate]
	var command := {
		"op": "advance",
		"episode_id": game.episode_id,
		"expected_tick": game.sim.tick,
		"action_id": "invalid-destination-action",
		"controls": {"throttle": 0.7, "steer": 0.1}
	}
	var response: Dictionary = game._request(command)
	check(injected.invalid_samples == 1, "Destination sample was not tested exactly once")
	check(
		injected.integrated_state.get("tick", -1) == injected.initial_tick + 1,
		"Ground fault was injected before actual integration completed"
	)
	check(
		simulator_state(game.sim) == before,
		"Destination failure did not restore full simulator state"
	)
	check(
		[game.lap_time, game.last_progress, game.legal_distance, game.next_gate] == before_progress,
		"Destination failure advanced lap or reward progress"
	)
	check(
		response.get("failure_type") == "infrastructure",
		"Invalid ground failure classification lost"
	)
	check(response.transitions.is_empty(), "Invalid destination became a normal transition")
	check(not response.has("reward_components"), "Invalid destination received a reward")
	check(not game.observation().rollout_valid, "Invalid ground rollout remained valid")
	check(response == game._request(command), "Invalid destination retry was not idempotent")
	check(injected.invalid_samples == 1, "Invalid action retry integrated again")
	game.recorder.flush()
	var records := FileAccess.get_file_as_string(game.recorder.get_path()).split("\n", false)
	var recorded_failures := 0
	var recorded_transitions := 0
	for line in records:
		var row: Dictionary = JSON.parse_string(line)
		if row.get("type") == "environment_failure":
			recorded_failures += 1
			check(
				(
					row.state
					== JSON.parse_string(JSON.stringify(game.serializable(game.sim.telemetry())))
				),
				"Recorded failure state differs from restored state"
			)
		if row.get("type") == "transition":
			recorded_transitions += 1
	check(recorded_failures == 1, "Invalid destination failure absent or duplicated in recording")
	check(recorded_transitions == 0, "Recording contains transition for invalid destination")
	game.track = original_track
	injected.free()


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
	check_invalid_destination(game)
	print("ENVIRONMENT_FAILURE_CHECK failures=", failures)
	game.queue_free()
	await process_frame
	quit(failures)
