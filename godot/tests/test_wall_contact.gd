extends SceneTree
## Real scene integration: human and agent commands share obstacle termination,
## recording and replay. Wall mesh solidity is tested separately.

var failures := 0
var checks := 0
var last_wall_elapsed_ms := 0.0


class FailedSweep:
	extends RefCounted
	var calls := 0

	func sweep(_space, _start, _finish, _mask) -> Dictionary:
		calls += 1
		return {"error": "Injected obstacle query failure"}


func check(value: bool, message: String) -> void:
	checks += 1
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	run.call_deferred()


func snapshot(simulator: RefCounted) -> Dictionary:
	var state: Dictionary = {}
	for property: Dictionary in simulator.get_property_list():
		if int(property.usage) & PROPERTY_USAGE_SCRIPT_VARIABLE:
			var value: Variant = simulator.get(property.name)
			state[property.name] = (
				value.duplicate(true) if value is Dictionary or value is Array else value
			)
	return state


func prepare(game: Node3D, agent: bool) -> void:
	game.reset_episode(0.0)
	game.agent_mode = agent
	game.paused = false
	game.sim.speed = 80.0
	game.sim.longitudinal_velocity = 80.0
	game.sim.gear = 6


func contact_events(transition: Dictionary) -> Array:
	var found: Array = []
	for event: Dictionary in transition.get("events", []):
		if event.get("type") == "obstacle_contact":
			found.append(event)
	return found


func assert_contact(game: Node3D, transition: Dictionary) -> void:
	check(not transition.has("error"), "Wall contact was an infrastructure error")
	check(game.sim.crashed and game.sim.crash_reason == "pit_wall", "Wall crash was not set")
	check(game.terminated and not game.lap_valid, "Wall contact did not invalidate and end lap")
	check(game.sim.tick == 1, "Wall contact did not stop on first tick")
	check(game.sim.speed == 0 and game.sim.longitudinal_velocity == 0, "Contact retained velocity")
	check(game.sim.lean_rate == 0, "Contact retained roll velocity")
	check(not game.sim.collision_contact.is_empty(), "Contact provenance absent")
	check(contact_events(transition).size() == 1, "Contact event absent or duplicated")
	check(transition.get("terminated", false), "Transition did not report termination")
	check(
		transition.get("reward_components", {}).get("crash", false), "Crash reward component absent"
	)
	check(game.observation().rollout_valid, "Valid obstacle crash invalidated rollout")
	var contact: Dictionary = game.sim.collision_contact
	check(
		float(contact.get("safe_fraction", -1)) > 0 and float(contact.get("safe_fraction", 1)) < 1,
		"Impact did not retain a clear prefix within the tick"
	)
	check(
		(
			(
				float(contact.get("interval_end_fraction", -1))
				>= float(contact.get("safe_fraction", 0))
			)
			and float(contact.get("interval_end_fraction", 2)) <= 1
		),
		"Contact interval is invalid"
	)
	check(
		(
			float(contact.get("spatial_uncertainty_m", INF))
			<= game.collision_sweep.SPATIAL_TOLERANCE_M
		),
		"Impact exceeds declared spatial tolerance"
	)
	check(contact.get("obstacle_kind") == "pit_wall", "Obstacle identity lost")
	check(not str(contact.get("component_id", "")).is_empty(), "Bike component identity lost")
	check(contact.has("candidate_preimpact_state"), "Preimpact state missing")
	if contact.has("candidate_preimpact_state"):
		check(
			contact.candidate_preimpact_state.longitudinal_velocity > 70,
			"Preimpact speed was overwritten"
		)
	game._update_visual(0.0)
	var overlaps: Array = game.collision_sweep.envelope.overlaps(
		game.get_world_3d().direct_space_state,
		game.bike_root.global_transform,
		-game.sim.lean,
		-game.sim.steering,
		game.wheel_rotation,
		preload("res://scripts/pit_wall.gd").COLLISION_LAYER
	)
	check(overlaps.is_empty(), "Certified stopped pose intersects obstacle")
	check(
		game.bike_root.position == game.sim.position, "Rendered root differs from collision state"
	)
	if contact.has("root_rotation"):
		var rotation: Array = contact.root_rotation
		check(
			game.bike_root.rotation.is_equal_approx(Vector3(rotation[0], rotation[1], rotation[2])),
			"Rendered crash orientation lost"
		)
	else:
		check(false, "Certified root orientation absent")
	var stopped := snapshot(game.sim)
	var stopped_progress := [game.lap_time, game.last_progress, game.legal_distance, game.next_gate]
	var rejected: Dictionary = game._step({"throttle": 1.0})
	check(rejected.has("error"), "Terminal direct step was accepted")
	game.agent_mode = false
	game._physics_process(game.DT)
	check(snapshot(game.sim) == stopped, "Human physics advanced terminal motorcycle")
	check(
		(
			[game.lap_time, game.last_progress, game.legal_distance, game.next_gate]
			== stopped_progress
		),
		"Terminal progress advanced"
	)


func check_recording(game: Node3D) -> void:
	game.recorder.flush()
	var rows := FileAccess.get_file_as_string(game.recorder.get_path()).split("\n", false)
	var manifest: Dictionary = JSON.parse_string(rows[0])
	if manifest.get("build", {}).get("kind") == "unbundled_development":
		for source in [
			"chamfered_box",
			"rider_pose",
			"rider_glove",
			"rider_arm_visual",
			"curb_surface",
			"triangle_ribbon"
		]:
			var path: String = "res://scripts/" + source + ".gd"
			check(
				(
					manifest.build.get("geometry_helper_sha256", {}).get(path)
					== FileAccess.get_sha256(path)
				),
				"Recording omitted geometry helper hash: " + source
			)
	check(
		(
			manifest.get("obstacle_collision", {}).get("pit_wall_sha256")
			== FileAccess.get_sha256("res://data/pit-wall.json")
		),
		"Manifest omitted loaded wall hash"
	)
	var changed := manifest.duplicate(true)
	changed.obstacle_collision.pit_wall_sha256 = "deliberately-different-wall"
	var fixture_path := "user://wall-contact-replay-manifest-test.jsonl"
	var fixture := FileAccess.open(fixture_path, FileAccess.WRITE)
	fixture.store_line(JSON.stringify(changed))
	fixture.close()
	var verifier := preload("res://scripts/replay.gd").new()
	check(
		(
			verifier
			. open_recording(fixture_path, FileAccess.get_sha256("res://data/track.json"))
			. contains("pit wall hash")
		),
		"Replay accepted mismatched wall geometry"
	)
	verifier.file.close()
	changed.erase("obstacle_collision")
	fixture = FileAccess.open(fixture_path, FileAccess.WRITE)
	fixture.store_line(JSON.stringify(changed))
	fixture.close()
	check(
		(
			verifier
			. open_recording(fixture_path, FileAccess.get_sha256("res://data/track.json"))
			. is_empty()
		),
		"Legacy replay without wall hash was rejected"
	)
	verifier.file.close()
	var transitions := 0
	var contacts := 0
	var last_state: Dictionary = {}
	for line in rows:
		var row: Dictionary = JSON.parse_string(line)
		if row.get("type") == "transition":
			transitions += 1
			contacts += contact_events(row).size()
			last_state = row.state
	check(transitions == 1 and contacts == 1, "Recording duplicated or omitted impact transition")
	if last_state.is_empty():
		return
	var replay_sim := MotorcycleSim.new()
	var replay := preload("res://scripts/replay.gd").new()
	replay.apply_state(replay_sim, last_state)
	check(
		replay_sim.collision_contact == last_state.collision_contact, "Replay omitted contact state"
	)
	check(
		replay_sim.crashed and replay_sim.crash_reason == "pit_wall", "Replay omitted crash reason"
	)
	check(replay_sim.position == game.sim.position, "Replay moved impact position")
	var saved_sim: RefCounted = game.sim
	game.sim = replay_sim
	check(
		(
			replay
			. open_recording(
				game.recorder.get_path(), FileAccess.get_sha256("res://data/track.json")
			)
			. is_empty()
		),
		"Contact recording failed to open for playback"
	)
	game.replay = replay
	game.agent_mode = false
	game.wheel_rotation = -999.0
	game._physics_process(game.DT)
	game._update_visual(0.0)
	check(
		is_equal_approx(game.wheel_rotation, float(last_state.collision_contact.wheel_rotation)),
		"Replay visual lost certified wheel orientation"
	)
	game.sim = saved_sim
	game.replay = null


func check_historical_wall(game: Node3D) -> void:
	var path := "res://data/pit-wall.json"
	check(FileAccess.file_exists(path), "Historical wall source is missing")
	var profile: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
	var row: Dictionary = profile.rows[profile.rows.size() / 2]
	var inward := (
		Vector3(float(row.road_inward_xz[0]), 0, float(row.road_inward_xz[1])).normalized()
	)
	var face := Vector3(float(row.face_xz_m[0]), 0, float(row.face_xz_m[1]))
	prepare(game, true)
	game.sim.position = face + inward * 1.3
	game.sim.position.y = game.track.sample_world(game.sim.position).height
	var forward := -inward
	game.sim.heading = atan2(forward.x, -forward.z)
	game._update_visual(0.0)
	var envelope: RefCounted = game.collision_sweep.envelope
	var space: PhysicsDirectSpaceState3D = game.get_world_3d().direct_space_state
	var mask: int = preload("res://scripts/pit_wall.gd").COLLISION_LAYER
	check(
		envelope.overlaps(space, game.bike_root.global_transform, 0, 0, 0, mask).is_empty(),
		"Historical wall fixture starts overlapping"
	)
	game.collision_sweep.profiling_enabled = "--profile-sweep" in OS.get_cmdline_user_args()
	var query_started := Time.get_ticks_usec()
	var transition: Dictionary = game._step({"throttle": 0.0})
	last_wall_elapsed_ms = (Time.get_ticks_usec() - query_started) / 1000.0
	print("HISTORICAL_WALL_STEP_MS ", last_wall_elapsed_ms)
	if game.collision_sweep.profiling_enabled:
		print("HISTORICAL_WALL_PROFILE ", JSON.stringify(game.collision_sweep.profile))
	check(not transition.has("error"), "Historical wall contact produced infrastructure failure")
	check(
		game.sim.tick == 1 and game.sim.crashed and game.sim.crash_reason == "pit_wall",
		"Historical wall did not stop first tick"
	)
	check(
		game.sim.collision_contact.get("status") != "initial_overlap",
		"Historical wall test used overlap instead of sweep"
	)
	check(
		game.sim.collision_contact.get("obstacle_kind") == "pit_wall",
		"Historical wall identity missing"
	)
	check(contact_events(transition).size() == 1, "Historical wall contact event missing")
	game._update_visual(0.0)
	check(
		(
			envelope
			. overlaps(
				space,
				game.bike_root.global_transform,
				-game.sim.lean,
				-game.sim.steering,
				game.wheel_rotation,
				mask
			)
			. is_empty()
		),
		"Historical wall stop is not clear"
	)
	check(
		game.sim.speed == 0 and game.terminated and not game.lap_valid,
		"Historical wall contact did not terminate lap"
	)
	print("HISTORICAL_WALL_CONTACT ", JSON.stringify(game.sim.collision_contact))
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--screenshot=") and DisplayServer.get_name() != "headless":
			game.camera_mode = 0
			game._update_visual(1.0)
			game.hud.visible = false
			await RenderingServer.frame_post_draw
			var screenshot := get_root().get_texture().get_image()
			var output := arg.trim_prefix("--screenshot=")
			check(screenshot.save_png(output) == OK, "Historical wall screenshot failed")
			print("WALL_CONTACT_SCREENSHOT ", output)


func run() -> void:
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.set_physics_process(false)
	game.set_process(false)
	prepare(game, false)
	var obstacle := StaticBody3D.new()
	obstacle.collision_layer = preload("res://scripts/pit_wall.gd").COLLISION_LAYER
	obstacle.collision_mask = 0
	obstacle.set_meta("obstacle_kind", "pit_wall")
	var shape := BoxShape3D.new()
	shape.size = Vector3(4.0, 3.0, 0.08)
	var collider := CollisionShape3D.new()
	collider.shape = shape
	obstacle.add_child(collider)
	root.add_child(obstacle)
	obstacle.position = game.sim.position + game.sim.forward() * 1.4 + Vector3.UP * 1.5
	obstacle.rotation.y = -game.sim.heading
	await physics_frame
	await physics_frame
	await process_frame
	var human: Dictionary = game._step({"throttle": 0.0})
	assert_contact(game, human)
	check_recording(game)
	var human_state: Dictionary = game.serializable(game.sim.telemetry())
	prepare(game, true)
	check(
		game.sim.collision_contact.is_empty() and not game.sim.crashed,
		"Reset retained crash contact"
	)
	var request := {
		"op": "advance",
		"episode_id": game.episode_id,
		"expected_tick": 0,
		"action_id": "wall-hit",
		"controls": {"throttle": 0.0}
	}
	var response: Dictionary = game._request(request)
	check(response.get("transitions", []).size() == 1, "Agent batch continued past contact")
	if response.get("transitions", []).size() == 1:
		assert_contact(game, response.transitions[0])
	check(
		game.serializable(game.sim.telemetry()) == human_state,
		"Human and agent impact states diverged"
	)
	game.agent_mode = true
	check(response == game._request(request), "Agent impact retry was not idempotent")
	check(game.sim.tick == 1, "Agent retry advanced terminal physics")
	check_recording(game)
	# A reset inside an obstacle must detect initial overlap even at zero speed.
	prepare(game, true)
	game.sim.speed = 0
	game.sim.longitudinal_velocity = 0
	obstacle.position = game.sim.position + Vector3.UP * 1.5
	await physics_frame
	await physics_frame
	await process_frame
	var overlap: Dictionary = game._step({})
	check(not overlap.has("error") and game.sim.crashed, "Initial overlap was missed")
	check(contact_events(overlap).size() == 1, "Initial overlap did not emit one contact")
	game.reset_episode(0.0)
	check(
		game.sim.collision_contact.is_empty() and not game.terminated,
		"Reset failed after initial overlap"
	)
	var original_solver: RefCounted = game.collision_sweep
	var failing_solver := FailedSweep.new()
	game.collision_sweep = failing_solver
	var before := snapshot(game.sim)
	var before_progress := [game.lap_time, game.last_progress, game.legal_distance, game.next_gate]
	var fault_request := {
		"op": "advance",
		"episode_id": game.episode_id,
		"expected_tick": 0,
		"action_id": "query-failure",
		"controls": {"throttle": 0.5}
	}
	var failure: Dictionary = game._request(fault_request)
	check(failure.has("error"), "Query failure was swallowed")
	check(failure.get("transitions", []).is_empty(), "Failed query became transition")
	check(not failure.has("reward_components"), "Failed query received reward")
	check(snapshot(game.sim) == before, "Query failure did not restore full simulator state")
	check(
		[game.lap_time, game.last_progress, game.legal_distance, game.next_gate] == before_progress,
		"Query failure advanced progress"
	)
	check(not game.observation().rollout_valid, "Failed query left rollout valid")
	check(
		game._request(fault_request) == failure and failing_solver.calls == 1,
		"Query failure retry was not idempotent"
	)
	game.recorder.flush()
	var failure_rows := FileAccess.get_file_as_string(game.recorder.get_path()).split("\n", false)
	var recorded_failures := 0
	var recorded_transitions := 0
	for line in failure_rows:
		var row: Dictionary = JSON.parse_string(line)
		recorded_failures += int(row.get("type") == "environment_failure")
		recorded_transitions += int(row.get("type") == "transition")
	check(
		recorded_failures == 1 and recorded_transitions == 0,
		"Query failure recording lost atomicity"
	)
	game.collision_sweep = original_solver
	obstacle.queue_free()
	await physics_frame
	await physics_frame
	var comparison_source := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--compare-sweep-source="):
			comparison_source = arg.trim_prefix("--compare-sweep-source=")
	if comparison_source.is_empty():
		await check_historical_wall(game)
	else:
		await compare_sweeps(game, comparison_source)
	game.queue_free()
	await process_frame
	print("WALL_CONTACT_CHECK checks=", checks, " failures=", failures)
	quit(failures)


func compare_sweeps(game: Node3D, source_path: String) -> void:
	var baseline_script := GDScript.new()
	baseline_script.source_code = FileAccess.get_file_as_string(source_path)
	var valid := not baseline_script.source_code.is_empty() and baseline_script.reload() == OK
	check(valid, "Cannot load requested baseline sweep source")
	if not valid:
		return
	var current: RefCounted = game.collision_sweep
	var baseline: RefCounted = baseline_script.new()
	check(baseline.build(current.envelope).is_empty(), "Cannot build baseline sweep")
	var timings := {"baseline": [], "current": []}
	var expected := {}
	for trial in range(3):
		for label in ["baseline", "current"]:
			game.collision_sweep = baseline if label == "baseline" else current
			await check_historical_wall(game)
			timings[label].append(last_wall_elapsed_ms)
			var contact: Dictionary = game.sim.collision_contact.duplicate(true)
			contact.erase("queries")
			contact.erase("intervals")
			if expected.is_empty():
				expected = contact
			check(contact == expected, "Compared sweeps changed the recorded contact")
	game.collision_sweep = current
	print("SWEEP_COMPARISON ", JSON.stringify(timings))
