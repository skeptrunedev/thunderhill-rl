extends SceneTree
# Actual simulator boundary qualification. The near-finish accounting fixture
# below is not a model rollout or training sample.
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.agent_mode = true
	game.reset_episode(0.0, "simulator-handoff-qualification", {}, {}, null, 8.0)
	var neutral := {"throttle": 0.0, "steer": 0.0, "front_brake": 0.0, "rear_brake": 0.0}
	for i in range(180):
		game._step(neutral)
	var before: Dictionary = game.observation()
	var handoff: Dictionary = game._request({
		"op": "begin_model_control", "episode_id": game.episode_id, "expected_tick": 180,
	})
	check(not handoff.has("error"), "Handoff rejected measured warmup")
	check(handoff.state == before.state and handoff.tick == before.tick, "Handoff changed physics")
	check(game.sim.speed >= 4.0, "Neutral coasting lost the required moving history")
	check(game.last_progress > 0.0 and game.last_progress < 1.0 / 32, "Fixture is not inside a gate sector")
	check(game.legal_distance == 0.0 and game.lap_time == 0.0, "Warmup counted toward lap")
	# Emulate the accounting immediately after the 32nd fixed gate, while still
	# short of returning to the arbitrary handoff origin. Execute real physics
	# on either side of the full-distance boundary to test termination.
	game.passed_gates = 32
	game.legal_distance = game.track.length_m - 10.0
	var short_step: Dictionary = game._step(neutral)
	check(not short_step.terminated and game.completed_laps == 0, "Fixed gates awarded a short lap")
	game.legal_distance = game.track.length_m - 0.001
	var complete_step: Dictionary = game._step(neutral)
	check(complete_step.termination_reason == "lap_completed", "Full circuit did not complete")
	check(game.legal_distance >= game.track.length_m, "Completed lap is short of a full circuit")
	check(game.completed_laps == 1, "Lap count changed incorrectly")
	game.recorder.flush()
	print("MODEL_CONTROL_HANDOFF_CHECK training_eligible=false failures=", failures,
		" handoff_speed_m_s=", handoff.state.speed,
		" warmup_distance_m=", handoff.model_control_start.initial_legal_distance_m)
	game.queue_free()
	await process_frame
	quit(failures)
