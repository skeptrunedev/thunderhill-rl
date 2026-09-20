extends Node3D

const TrackScript = preload("res://scripts/track.gd")
const BikeScript = preload("res://scripts/bike_visual.gd")
const SimScript = preload("res://scripts/motorcycle.gd")
const HudScript = preload("res://scripts/hud.gd")
const AgentCameraScript = preload("res://scripts/agent_camera.gd")
const DT: float = 1.0 / 120.0
var track: Node3D
var sim: RefCounted
var collision_sweep: RefCounted
var _collision_start_pose: Dictionary = {}
var engine_audio: AudioStreamPlayer
var bike_root: Node3D
var bike: Node3D
var camera: Camera3D
var hud: Control
var paused := true
var agent_mode := false
var camera_mode := 0
var wheel_rotation := 0.0
var lap_time := 0.0
var completed_laps := 0
var lap_valid := true
var last_progress := 0.0
var legal_distance := 0.0
var next_gate := 1
var passed_gates := 0
var episode_id: String
var episode_number := 0
var recorder: FileAccess
var controls: Dictionary = {}
var steering_input := 0.0
var server := TCPServer.new()
var clients: Array = []
var action_cache: Dictionary = {}
var action_requests: Dictionary = {}
var policy_id := "human"
var terminated := false
var environment_failure: Dictionary = {}
var server_port := 0
var screenshot_path := ""
var screenshot_thread: Thread
var qa_ticks := 0
var qa_target := 0
var preview_mode := false
var preview_station := 1650.0
var run_id: String
var replay_path: String = ""
var replay: RefCounted
var benchmark: RefCounted
var benchmark_path := ""
var frame_times: Array = []
var agent_camera: Node
var agent_request_pending := false
var startup_profile: FileAccess
var startup_started_usec: int
var startup_previous_usec: int


func _ready() -> void:
	startup_started_usec = Time.get_ticks_usec()
	startup_previous_usec = startup_started_usec
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--startup-profile="):
			startup_profile = FileAccess.open(
				arg.trim_prefix("--startup-profile="), FileAccess.WRITE
			)
			if startup_profile == null:
				push_error(
					"Cannot open requested startup profile: %s" % FileAccess.get_open_error()
				)
				get_tree().quit(2)
				return
	_startup_mark("scene_ready_begin")
	run_id = "%d_%d" % [Time.get_unix_time_from_system(), OS.get_process_id()]
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--agent-port="):
			server_port = int(arg.split("=")[1])
			agent_mode = true
			paused = false
		if arg.begins_with("--screenshot="):
			screenshot_path = arg.trim_prefix("--screenshot=")
		if arg.begins_with("--qa-ticks="):
			qa_target = int(arg.split("=")[1])
			paused = false
		if arg.begins_with("--preview-station="):
			preview_station = float(arg.trim_prefix("--preview-station="))
		if arg.begins_with("--preview-camera="):
			camera_mode = int(arg.trim_prefix("--preview-camera="))
		if arg == "--preview":
			preview_mode = true
		if arg.begins_with("--benchmark-recording="):
			benchmark_path = arg.trim_prefix("--benchmark-recording=")
		if arg.begins_with("--replay="):
			replay_path = arg.trim_prefix("--replay=")
	if (
		"--qa-controls" in OS.get_cmdline_user_args()
		and (agent_mode or preview_mode or qa_target > 0 or not replay_path.is_empty())
	):
		push_error("Control checks require an ordinary human game session")
		get_tree().quit(2)
		return
	if (
		not benchmark_path.is_empty()
		and (
			agent_mode
			or preview_mode
			or qa_target > 0
			or not replay_path.is_empty()
			or "--qa-controls" in OS.get_cmdline_user_args()
		)
	):
		push_error("Benchmark requires an exclusive diagnostic session")
		get_tree().quit(2)
		return
	track = TrackScript.new()
	track.startup_observer = _startup_mark
	add_child(track)
	if not track.initialization_error.is_empty():
		get_tree().quit(2)
		return
	_startup_mark("track_complete")
	var horizon = preload("res://scripts/horizon.gd").new()
	add_child(horizon)
	horizon.build(track)
	_startup_mark("horizon_complete")
	var landmarks = preload("res://scripts/landmarks.gd").new()
	add_child(landmarks)
	landmarks.build(track)
	if not landmarks.initialization_error.is_empty():
		push_error(landmarks.initialization_error)
		get_tree().quit(2)
		return
	_startup_mark("landmarks_complete")
	var scenery = preload("res://scripts/scenery.gd").new()
	add_child(scenery)
	scenery.build(track)
	_startup_mark("scenery_complete")
	sim = SimScript.new()
	if DisplayServer.get_name() != "headless":
		engine_audio = preload("res://scripts/engine_audio.gd").new()
		engine_audio.sim = sim
		add_child(engine_audio)
	_environment()
	bike_root = Node3D.new()
	add_child(bike_root)
	bike = BikeScript.new()
	bike_root.add_child(bike)
	var envelope = preload("res://scripts/bike_collision_envelope.gd").new()
	var collision_error: String = envelope.build(bike)
	collision_sweep = preload("res://scripts/bike_sweep.gd").new()
	if collision_error.is_empty():
		collision_error = collision_sweep.build(envelope)
	if not collision_error.is_empty():
		push_error(collision_error)
		get_tree().quit(2)
		return
	_startup_mark("environment_and_bike_complete")
	AgentCameraScript.assign_rider_layer(bike.rider)
	camera = Camera3D.new()
	camera.far = 3000
	camera.near = 0.06
	camera.fov = 64
	add_child(camera)
	if agent_mode and DisplayServer.get_name() != "headless":
		agent_camera = AgentCameraScript.new()
		add_child(agent_camera)
		agent_camera.configure(get_world_3d())
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = HudScript.new()
	hud.game = self
	layer.add_child(hud)
	if benchmark_path.is_empty():
		reset_episode(0.0)
	if preview_mode:
		reset_episode(preview_station)
		paused = true
		hud.visible = false
	if not replay_path.is_empty():
		replay = preload("res://scripts/replay.gd").new()
		var replay_error: String = replay.open_recording(
			replay_path, FileAccess.get_sha256("res://data/track.json")
		)
		if not replay_error.is_empty():
			push_error(replay_error)
			get_tree().quit(2)
			return
		replay.apply_state(sim, replay.manifest.initial_state)
		paused = false
	if not benchmark_path.is_empty():
		benchmark = preload("res://scripts/control_benchmark.gd").new()
		var error: String = benchmark.open_trace(
			benchmark_path, sim, FileAccess.get_sha256("res://data/track.json"), DT
		)
		if not error.is_empty():
			push_error(error)
			get_tree().quit(2)
			return
		reset_episode(
			float(benchmark.reader.manifest.start_station),
			"diagnostic-recorded-inputs",
			benchmark.reader.manifest.initial_state
		)
		paused = false
		benchmark.start()
	if server_port > 0:
		var err := server.listen(server_port, "127.0.0.1")
		if err != OK:
			push_error("Cannot bind agent port: %s" % err)
			get_tree().quit(2)
	_update_visual(1.0)
	print(
		"THUNDERHILL_READY ",
		JSON.stringify(
			{"port": server_port, "episode_id": episode_id, "track_samples": track.points.size()}
		)
	)

	_startup_mark("scene_ready_complete")
	if startup_profile != null:
		if DisplayServer.get_name() == "headless":
			startup_profile.close()
			startup_profile = null
		else:
			RenderingServer.frame_post_draw.connect(_startup_first_draw, CONNECT_ONE_SHOT)
	if "--qa-controls" in OS.get_cmdline_user_args():
		_run_control_checks.call_deferred()


func _startup_mark(stage: String) -> void:
	if startup_profile == null:
		return
	var now := Time.get_ticks_usec()
	startup_profile.store_line(
		JSON.stringify(
			{
				"stage": stage,
				"elapsed_ms": (now - startup_started_usec) / 1000.0,
				"interval_ms": (now - startup_previous_usec) / 1000.0,
				"pid": OS.get_process_id(),
				"platform": OS.get_name(),
				"display_server": DisplayServer.get_name()
			}
		)
	)
	# Preserve completed stages while the next expensive operation is still running.
	startup_profile.flush()
	startup_previous_usec = now


func _startup_first_draw() -> void:
	_startup_mark("first_frame_post_draw")
	startup_profile.close()
	startup_profile = null


func _run_control_checks() -> void:
	var checks = preload("res://scripts/control_checks.gd").new()
	var failures: int = await checks.run(self)
	get_tree().quit(0 if failures == 0 else 1)


func _environment() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	var sky := Sky.new()
	var sky_mat := ShaderMaterial.new()
	sky_mat.shader = preload("res://shaders/sky.gdshader")
	sky_mat.set_shader_parameter("panorama", preload("res://assets/sky/kloofendal_2k.hdr"))
	sky.sky_material = sky_mat
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	# Sky lighting strength lives in sky.gdshader. ambient_light_energy does
	# not attenuate a sky with full ambient_light_sky_contribution.
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.fog_enabled = true
	env.fog_light_color = Color("b9bdad")
	env.fog_density = 0.00035
	env.fog_sky_affect = 0.12
	var world := WorldEnvironment.new()
	world.environment = env
	add_child(world)
	var sun := DirectionalLight3D.new()
	sun.light_color = Color("fff0d5")
	sun.light_energy = 1.0
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 160
	sun.directional_shadow_mode = DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
	add_child(sun)
	var sky_metadata: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/sky.json")
	)
	var direction: Array = sky_metadata.toward_sun
	sun.look_at(-Vector3(direction[0], direction[1], direction[2]), Vector3.UP)


func reset_episode(
	station: float, checkpoint: String = "human", initial_state: Dictionary = {}
) -> Dictionary:
	var index := 0
	for i in track.samples.size():
		if (
			absf(float(track.samples[i].s) - station)
			< absf(float(track.samples[index].s) - station)
		):
			index = i
	var tangent: Vector3 = (
		(track.points[(index + 1) % track.points.size()] - track.points[index]).normalized()
	)
	sim.reset(track.points[index], atan2(tangent.x, -tangent.z))
	episode_number += 1
	episode_id = run_id + "_" + str(episode_number)
	policy_id = checkpoint
	lap_time = 0
	completed_laps = 0
	legal_distance = 0
	passed_gates = 0
	last_progress = float(track.samples[index].s) / track.length_m
	next_gate = (int(last_progress * 32) + 1) % 32
	lap_valid = true
	terminated = false
	environment_failure.clear()
	wheel_rotation = 0
	steering_input = 0
	action_cache.clear()
	action_requests.clear()
	if not initial_state.is_empty():
		preload("res://scripts/replay.gd").new().apply_state(sim, initial_state)
	_start_recording(station)
	return observation()


func _start_recording(station: float) -> void:
	if recorder:
		recorder.close()
	var folder: String = "user://runs/" + run_id
	DirAccess.make_dir_recursive_absolute(folder)
	recorder = FileAccess.open(folder + "/" + episode_id + ".jsonl", FileAccess.WRITE)
	_record(
		{
			"type": "episode",
			"schema_version": 1,
			"build": _build_provenance(),
			"episode_id": episode_id,
			"policy_id": policy_id,
			"track_sha256": FileAccess.get_sha256("res://data/track.json"),
			"terrain_sha256": FileAccess.get_sha256("res://data/terrain.json"),
			"surface_sha256": FileAccess.get_sha256("res://data/surface.json"),
			"physics_version": sim.MODEL_VERSION,
			"physics_dt": DT,
			"obstacle_collision":
			{
				"envelope": preload("res://scripts/bike_collision_envelope.gd").MODEL_VERSION,
				"sweep": preload("res://scripts/bike_sweep.gd").MODEL_VERSION,
				"spatial_tolerance_m": preload("res://scripts/bike_sweep.gd").SPATIAL_TOLERANCE_M,
				"pit_wall_sha256": FileAccess.get_sha256("res://data/pit-wall.json"),
				"response": "terminal_freeze_no_impact_dynamics"
			},
			"initial_state": sim.telemetry(),
			"parameters": sim.parameters,
			"engine": Engine.get_version_info(),
			"start_station": station
		}
	)


func _record(record: Dictionary) -> void:
	if recorder:
		recorder.store_line(JSON.stringify(serializable(record)))


func serializable(value: Variant) -> Variant:
	if value is Vector3:
		return [value.x, value.y, value.z]
	if value is Vector2:
		return [value.x, value.y]
	if value is Dictionary:
		var result: Dictionary = {}
		for k in value:
			result[k] = serializable(value[k])
		return result
	if value is Array:
		var result: Array = []
		for v in value:
			result.append(serializable(v))
		return result
	return value


func _physics_process(_dt: float) -> void:
	if (
		agent_mode
		or paused
		or not environment_failure.is_empty()
		or (terminated and replay == null)
	):
		return
	if replay != null:
		var row: Dictionary = replay.next_state()
		if row.is_empty():
			paused = true
			print("REPLAY_COMPLETE ticks=", replay.consumed_ticks)
			return
		replay.apply_state(sim, row.state)
		lap_time = sim.elapsed
		wheel_rotation += sim.longitudinal_velocity * DT / 0.32
		if not sim.collision_contact.is_empty():
			wheel_rotation = float(sim.collision_contact.wheel_rotation)
		return
	if benchmark != null:
		var action: Dictionary = benchmark.next_action()
		if action.is_empty() or action.has("error"):
			_finish_benchmark(str(action.get("error", "")))
			return
		controls = action.controls
	else:
		controls = _human_controls()
	var result: Dictionary = _step(controls)
	if benchmark != null:
		if result.has("error"):
			_finish_benchmark(str(result.error))
			return
		benchmark.record_step(sim, bool(result.track.on_track))
		if sim.crashed:
			_finish_benchmark("Motorcycle crashed during recorded input run")
			return
	if qa_target > 0:
		qa_ticks += 1
		if qa_ticks >= qa_target:
			print("QA_COMPLETE ", JSON.stringify(observation()))
			if recorder:
				recorder.flush()
			get_tree().quit()


func _finish_benchmark(failure: String) -> void:
	paused = true
	var result: Dictionary = benchmark.report(sim, completed_laps, failure)
	result["build"] = _build_provenance()
	_record({"type": "benchmark_result", "result": result})
	if recorder:
		recorder.flush()
	print("BENCHMARK_RESULT ", JSON.stringify(result))
	get_tree().quit(0 if result.ok else 2)


func _human_controls() -> Dictionary:
	var throttle := (
		1.0
		if Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP)
		else 0.0
	)
	var brake := (
		1.0
		if Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN)
		else 0.0
	)
	var steer := (
		float(Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT))
		- float(Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT))
	)
	var rear := 1.0 if Input.is_physical_key_pressed(KEY_SPACE) else 0.0
	var pads := Input.get_connected_joypads()
	if not pads.is_empty():
		var pad: int = pads[0]
		throttle = maxf(throttle, Input.get_joy_axis(pad, JOY_AXIS_TRIGGER_RIGHT))
		brake = maxf(brake, Input.get_joy_axis(pad, JOY_AXIS_TRIGGER_LEFT))
		var stick := Input.get_joy_axis(pad, JOY_AXIS_LEFT_X)
		if absf(stick) > 0.08:
			steer = signf(stick) * (absf(stick) - 0.08) / 0.92
	steering_input = move_toward(steering_input, steer, DT * 2.1)
	return {
		"throttle": throttle,
		"front_brake": brake,
		"rear_brake": rear,
		"steer": steering_input,
		"assist_enabled": true,
		"auto_shift": true
	}


func _sample_ground_after_step() -> Dictionary:
	if not sim.position.is_finite():
		return {"error": "Integrated position is not finite", "failure_type": "infrastructure"}
	var ground: Dictionary = track.sample_world(sim.position)
	if (
		not is_finite(float(ground.height))
		or not ground.normal.is_finite()
		or ground.normal.y <= 0.0
	):
		return {
			"error": "Integrated position has invalid ground contact",
			"failure_type": "infrastructure"
		}
	sim.position.y = ground.height
	var next_wheel: float = wheel_rotation + sim.longitudinal_velocity * DT / 0.32
	var finish := _collision_pose(ground.normal, next_wheel)
	var contact: Dictionary = collision_sweep.sweep(
		get_world_3d().direct_space_state,
		_collision_start_pose,
		finish,
		preload("res://scripts/pit_wall.gd").COLLISION_LAYER
	)
	if contact.has("error"):
		return {"error": contact.error, "failure_type": "infrastructure"}
	if contact.status == "clear":
		return ground
	var stopped: Dictionary = (
		_collision_start_pose
		if contact.status == "initial_overlap"
		else collision_sweep.pose_at(float(contact.safe_fraction))
	)
	var stop_ground: Dictionary = track.sample_world(stopped.root.origin)
	if (
		not is_finite(float(stop_ground.height))
		or not stop_ground.normal.is_finite()
		or stop_ground.normal.y <= 0.0
	):
		return {
			"error": "Obstacle contact pose has invalid ground", "failure_type": "infrastructure"
		}
	contact["type"] = "obstacle_contact"
	contact["candidate_preimpact_state"] = serializable(sim.telemetry())
	var collider := instance_from_id(int(contact.collider_id))
	contact["obstacle_kind"] = str(collider.get_meta("obstacle_kind", "pit_wall"))
	# Instance IDs are process local, so recordings use the obstacle kind and
	# source hashes rather than presenting the ID as stable across rollouts.
	contact.erase("collider_id")
	var angles: Vector3 = stopped.root.basis.get_euler()
	contact["root_rotation"] = [angles.x, angles.y, angles.z]
	contact["wheel_rotation"] = stopped.wheel_rotation
	sim.position = stopped.root.origin
	sim.heading = -angles.y
	sim.lean = -float(stopped.lean)
	sim.steering = -float(stopped.steering)
	sim.speed = 0.0
	sim.longitudinal_velocity = 0.0
	sim.lean_rate = 0.0
	sim.crashed = true
	sim.crash_reason = "pit_wall"
	sim.collision_contact = contact
	return stop_ground


func _collision_pose(normal: Vector3, wheel: float) -> Dictionary:
	return {
		"root": _bike_root_transform(normal),
		"lean": -sim.lean,
		"steering": -sim.steering,
		"wheel_rotation": wheel
	}


func _bike_root_transform(normal: Vector3) -> Transform3D:
	if not sim.collision_contact.is_empty():
		var angles: Array = sim.collision_contact.root_rotation
		return Transform3D(Basis.from_euler(Vector3(angles[0], angles[1], angles[2])), sim.position)
	var tangent: Vector3 = sim.surface_forward(normal)
	return Transform3D(
		Basis.from_euler(
			Vector3(atan2(tangent.y, Vector2(tangent.x, tangent.z).length()), -sim.heading, 0)
		),
		sim.position
	)


func _step(action: Dictionary) -> Dictionary:
	if terminated:
		return {"error": "Episode terminated; reset required"}
	var old_tick: int = sim.tick
	var road: Dictionary = track.sample_world(sim.position)
	_collision_start_pose = _collision_pose(road.normal, wheel_rotation)
	var result: Dictionary = sim.step(DT, action, road, _sample_ground_after_step)
	if result.has("error"):
		environment_failure = {
			"type": "environment_failure",
			"episode_id": episode_id,
			"policy_id": policy_id,
			"tick": old_tick,
			"error": result.error,
			"failure_type": result.get("failure_type", "infrastructure"),
			"rollout_valid": false,
			"requested_controls": action,
			"state": sim.telemetry()
		}
		_record(environment_failure)
		if recorder:
			recorder.flush()
		paused = not agent_mode
		if qa_target > 0:
			get_tree().quit(2)
		return environment_failure.duplicate(true)
	var after: Dictionary = result.ground_after
	if sim.collision_contact.is_empty():
		sim.position.y = after.height
	sim.on_track = after.on_track
	var difference := float(after.progress) - last_progress
	if difference > 0.5:
		difference -= 1.0
	if difference < -0.5:
		difference += 1.0
	var moved: float = difference * track.length_m
	lap_time += DT
	var events: Array = []
	if not sim.collision_contact.is_empty():
		lap_valid = false
		events.append(sim.collision_contact.duplicate(true))
	if not after.on_track:
		lap_valid = false
	var gate := int(float(after.progress) * 32) % 32
	if gate == next_gate and moved >= 0 and moved < 5.0:
		passed_gates += 1
		next_gate = (next_gate + 1) % 32
		events.append({"type": "gate", "gate": gate})
		if passed_gates >= 32:
			completed_laps += 1
			events.append({"type": "lap", "time": lap_time, "valid": lap_valid})
			passed_gates = 0
			if agent_mode:
				terminated = true
			else:
				lap_time = 0
				lap_valid = true
	if after.on_track and absf(moved) < 5.0:
		legal_distance += moved
	last_progress = after.progress
	if sim.crashed:
		lap_valid = false
		terminated = true
	wheel_rotation += sim.longitudinal_velocity * DT / 0.32
	if not sim.collision_contact.is_empty():
		wheel_rotation = float(sim.collision_contact.wheel_rotation)
	var transition: Dictionary = {
		"type": "transition",
		"episode_id": episode_id,
		"policy_id": policy_id,
		"previous_tick": old_tick,
		"tick": sim.tick,
		"requested_controls": action,
		"state": sim.telemetry(),
		"track":
		{
			"progress": after.progress,
			"lateral_m": after.distance,
			"on_track": after.on_track,
			"lap_valid": lap_valid,
			"legal_distance": legal_distance,
			"next_gate": next_gate
		},
		"reward_components":
		{
			"legal_progress_m": moved if after.on_track and absf(moved) < 5.0 else 0.0,
			"elapsed_s": DT,
			"crash": sim.crashed
		},
		"events": events,
		"terminated": terminated,
		"truncated": false
	}
	_record(transition)
	return transition


func observation() -> Dictionary:
	var road: Dictionary = track.sample_world(sim.position)
	return serializable(
		{
			"episode_id": episode_id,
			"tick": sim.tick,
			"sim_time": sim.elapsed,
			"policy_id": policy_id,
			"observation_version": "telemetry-v1",
			"state": sim.telemetry(),
			"track":
			{
				"progress": road.progress,
				"lateral_m": road.distance,
				"on_track": road.on_track,
				"lap_valid": lap_valid,
				"legal_distance": legal_distance,
				"completed_laps": completed_laps
			},
			"terminated": terminated,
			"truncated": not environment_failure.is_empty(),
			"rollout_valid": environment_failure.is_empty(),
			"environment_failure": environment_failure.duplicate(true)
		}
	)


func _process(dt: float) -> void:
	if benchmark != null and not paused:
		benchmark.record_frame(get_viewport())
	_poll_agent()
	if engine_audio != null:
		engine_audio.muted = paused or agent_mode
	_update_visual(dt)
	if not paused and frame_times.size() < 72000:
		frame_times.append(dt)
	if not screenshot_path.is_empty() and Engine.get_process_frames() > 12:
		var path := screenshot_path
		screenshot_path = ""
		var wait_begin := Time.get_ticks_usec()
		await RenderingServer.frame_post_draw
		var read_begin := Time.get_ticks_usec()
		var screenshot := get_viewport().get_texture().get_image()
		var read_end := Time.get_ticks_usec()
		if benchmark != null:
			benchmark.record_capture("post_draw_wait", wait_begin, read_begin)
			benchmark.record_capture("gpu_readback", read_begin, read_end)
		# The worker exclusively owns the CPU image and never accesses the scene or GPU.
		screenshot_thread = Thread.new()
		var error := screenshot_thread.start(_save_screenshot.bind(screenshot, path))
		if error == OK:
			while screenshot_thread.is_alive():
				await get_tree().process_frame
			var saved: Dictionary = screenshot_thread.wait_to_finish()
			error = saved.error
			if benchmark != null:
				benchmark.record_capture("png_save_worker", saved.begin_usec, saved.end_usec)
		screenshot_thread = null
		print("SCREENSHOT ", path, " result=", error)
		if preview_mode:
			get_tree().quit()


static func _save_screenshot(screenshot: Image, path: String) -> Dictionary:
	var begin_usec := Time.get_ticks_usec()
	var error := screenshot.save_png(path)
	return {"error": error, "begin_usec": begin_usec, "end_usec": Time.get_ticks_usec()}


func _update_visual(dt: float) -> void:
	if sim == null:
		return
	var road: Dictionary = track.sample_world(sim.position)
	bike_root.transform = _bike_root_transform(road.normal)
	var riding_tangent: Vector3 = sim.surface_forward(road.normal)
	bike.update_pose(-sim.lean, -sim.steering, wheel_rotation)
	bike.update_instruments(sim.speed, sim.rpm, sim.gear)
	camera.set_cull_mask_value(20, camera_mode == 0)
	var forward := Vector3(sin(sim.heading), 0, -cos(sim.heading))
	if camera_mode == 0:
		camera.fov = 64.0
		var desired: Vector3 = sim.position - forward * 5.5 + Vector3.UP * 2.35
		var focus: Vector3 = sim.position + forward * 10 + Vector3.UP * 1.0
		camera.position = camera.position.lerp(desired, 1.0 - exp(-dt * 8.0))
		camera.look_at(focus, Vector3.UP)
	else:
		# Both views stay attached to the bike. The onboard view is a framing
		# study against the footage, separate from the helmet eye and policy camera.
		var onboard := camera_mode == 2
		var anchor: Vector3 = bike.ONBOARD_CAMERA_LOCAL if onboard else bike.RIDER_EYE_LOCAL
		var look_down := 0.38 if onboard else 0.25
		camera.fov = 74.0 if onboard else 90.0
		camera.global_position = bike.to_global(anchor)
		var upright := Vector3.UP.slide(riding_tangent).normalized()
		var gaze := riding_tangent * cos(look_down) - upright * sin(look_down)
		camera.look_at(camera.position + gaze * 30.0, upright)
		camera.rotate_object_local(Vector3.FORWARD, sim.lean * 0.22)


func _input(event: InputEvent) -> void:
	if not event is InputEventKey or not event.is_pressed() or event.is_echo():
		return
	if benchmark != null:
		return
	if event.keycode == KEY_ESCAPE and environment_failure.is_empty():
		paused = not paused
	if event.keycode == KEY_C:
		camera_mode = (camera_mode + 1) % 3
	if event.keycode == KEY_R and not agent_mode and replay == null:
		reset_episode(0)
	if event.keycode == KEY_F11:
		DisplayServer.window_set_mode(
			(
				DisplayServer.WINDOW_MODE_WINDOWED
				if DisplayServer.window_get_mode() == DisplayServer.WINDOW_MODE_FULLSCREEN
				else DisplayServer.WINDOW_MODE_FULLSCREEN
			)
		)


func menu_action(action: String) -> void:
	if agent_mode and action in ["ride", "reset", "cyclone"]:
		return
	match action:
		"ride":
			if environment_failure.is_empty():
				paused = false
		"reset":
			reset_episode(0)
			paused = false
		"cyclone":
			reset_episode(1650)
			paused = false
		"camera":
			camera_mode = (camera_mode + 1) % 3
		"quit":
			get_tree().quit()


func _poll_agent() -> void:
	if server_port == 0 or agent_request_pending:
		return
	while server.is_connection_available():
		clients.append({"peer": server.take_connection(), "buffer": ""})
	for client in clients.duplicate():
		var peer: StreamPeerTCP = client.peer
		peer.poll()
		if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			clients.erase(client)
			continue
		var available := peer.get_available_bytes()
		if available > 0:
			client.buffer += peer.get_utf8_string(available)
		if client.buffer.length() > 65536:
			peer.disconnect_from_host()
			clients.erase(client)
			continue
		while client.buffer.contains("\n"):
			var line: String = client.buffer.get_slice("\n", 0)
			client.buffer = client.buffer.substr(line.length() + 1)
			var request: Variant = JSON.parse_string(line)
			var response: Dictionary = {"error": "Request must be JSON object"}
			if request is Dictionary:
				if request.get("op", "") == "capture":
					agent_request_pending = true
					response = await _capture_request(request)
					agent_request_pending = false
				else:
					response = _request(request)
			peer.put_data((JSON.stringify(response) + "\n").to_utf8_buffer())


func _capture_request(request: Dictionary) -> Dictionary:
	if not agent_mode:
		return {"error": "Agent capture requires --agent-port"}
	if request.get("episode_id", "") != episode_id:
		return {"error": "Episode mismatch"}
	if request.get("expected_tick", -1) != sim.tick:
		return {"error": "Tick mismatch"}
	if agent_camera == null:
		return {
			"error":
			"Camera capture requires a rendered instance; headless dummy rendering is unsupported",
			"failure_type": "infrastructure"
		}
	_update_visual(0.0)
	var road: Dictionary = track.sample_world(sim.position)
	var captured: Dictionary = await agent_camera.capture(
		sim, road.normal, episode_id, "user://runs/" + run_id
	)
	if not captured.has("error"):
		var metadata: Dictionary = captured.duplicate(true)
		metadata.image.erase("base64")
		metadata["type"] = "camera_observation"
		_record(metadata)
		if recorder:
			recorder.flush()
	return captured


func _request(request: Dictionary) -> Dictionary:
	if not agent_mode:
		return {"error": "Agent stepping requires --agent-port"}
	var op: String = request.get("op", "")
	if op == "reset":
		var station: Variant = request.get("station", 0.0)
		if (
			not (station is float or station is int)
			or not is_finite(float(station))
			or station < 0
			or station >= track.length_m
		):
			return {"error": "Invalid station"}
		return reset_episode(float(station), str(request.get("policy_id", "unassigned")))
	if request.get("episode_id", "") != episode_id:
		return {"error": "Episode mismatch"}
	if op == "observe":
		return observation()
	if op == "advance":
		var id: String = str(request.get("action_id", ""))
		if id.is_empty():
			return {"error": "action_id required"}
		if action_cache.has(id):
			if action_requests[id] != request:
				return {"error": "action_id already used with different request"}
			return action_cache[id]
		if request.get("expected_tick", -1) != sim.tick:
			return {"error": "Tick mismatch"}
		if not environment_failure.is_empty():
			return serializable(environment_failure)
		if terminated:
			return {"error": "Episode terminated; reset required"}
		var action: Variant = request.get("controls", {})
		if not action is Dictionary:
			return {"error": "controls must be object"}
		var error: String = sim.validate_controls(action)
		if not error.is_empty():
			return {"error": error}
		# Server owned action duration, 12 ticks / 100ms, never chosen by the policy.
		var transitions: Array = []
		for tick in 12:
			var transition: Dictionary = _step(action)
			if transition.has("error"):
				var failure: Dictionary = serializable(transition)
				failure["action_id"] = id
				failure["transitions"] = transitions
				action_cache[id] = failure
				action_requests[id] = request.duplicate(true)
				return failure
			transitions.append(serializable(transition))
			if terminated:
				break
		var response := observation()
		response["transitions"] = transitions
		response["action_id"] = id
		action_cache[id] = response
		action_requests[id] = request.duplicate(true)
		if recorder:
			recorder.flush()
		return response
	return {"error": "Unknown operation"}


func _exit_tree() -> void:
	if screenshot_thread != null and screenshot_thread.is_started():
		screenshot_thread.wait_to_finish()
	if recorder:
		recorder.flush()
		recorder.close()
	if not frame_times.is_empty():
		frame_times.sort()
		print(
			"FRAME_TIMES ",
			JSON.stringify(
				{
					"samples": frame_times.size(),
					"median_ms": frame_times[int(frame_times.size() * 0.5)] * 1000,
					"p95_ms": frame_times[int(frame_times.size() * 0.95)] * 1000,
					"renderer": ProjectSettings.get_setting("rendering/renderer/rendering_method"),
					"platform": OS.get_name()
				}
			)
		)


func _build_provenance() -> Dictionary:
	if not OS.has_feature("editor") and FileAccess.file_exists("res://data/build-info.json"):
		var manifest: Dictionary = JSON.parse_string(
			FileAccess.get_file_as_string("res://data/build-info.json")
		)
		return {
			"build_id": manifest.build_id,
			"source_commit": manifest.source_commit,
			"source_dirty": manifest.source_dirty,
			"content_sha256": manifest.content_sha256
		}
	return {
		"kind": "unbundled_development",
		"physics_script_sha256": FileAccess.get_sha256("res://scripts/motorcycle.gd"),
		"game_script_sha256": FileAccess.get_sha256("res://scripts/main.gd"),
		"bike_visual_script_sha256": FileAccess.get_sha256("res://scripts/bike_visual.gd"),
		"envelope_script_sha256": FileAccess.get_sha256("res://scripts/bike_collision_envelope.gd"),
		"sweep_script_sha256": FileAccess.get_sha256("res://scripts/bike_sweep.gd")
	}
