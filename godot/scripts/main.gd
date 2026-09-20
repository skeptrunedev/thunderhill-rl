extends Node3D

const TrackScript = preload("res://scripts/track.gd")
const BikeScript = preload("res://scripts/bike_visual.gd")
const SimScript = preload("res://scripts/motorcycle.gd")
const HudScript = preload("res://scripts/hud.gd")
const AgentCameraScript = preload("res://scripts/agent_camera.gd")
const DT: float = 1.0 / 120.0
var track: Node3D
var sim: RefCounted
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
var server_port := 0
var screenshot_path := ""
var qa_ticks := 0
var qa_target := 0
var preview_mode := false
var preview_station := 1650.0
var run_id: String
var replay_path: String = ""
var replay: RefCounted
var frame_times: Array = []
var agent_camera: Node
var agent_request_pending := false


func _ready() -> void:
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
		if arg.begins_with("--replay="):
			replay_path = arg.trim_prefix("--replay=")
	track = TrackScript.new()
	add_child(track)
	var horizon = preload("res://scripts/horizon.gd").new()
	add_child(horizon)
	horizon.build(track)
	var landmarks = preload("res://scripts/landmarks.gd").new()
	add_child(landmarks)
	landmarks.build(track)
	var scenery = preload("res://scripts/scenery.gd").new()
	add_child(scenery)
	scenery.build(track)
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
	env.ambient_light_energy = 0.45
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


func reset_episode(station: float, checkpoint: String = "human") -> Dictionary:
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
	wheel_rotation = 0
	steering_input = 0
	action_cache.clear()
	action_requests.clear()
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
			"physics_version": sim.MODEL_VERSION,
			"physics_dt": DT,
			"initial_state": sim.telemetry(),
			"parameters": sim.parameters,
			"engine": Engine.get_version_info(),
			"start_station": station
		}
	)
	return observation()


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
	if agent_mode or paused:
		return
	if replay != null:
		var row: Dictionary = replay.next_state()
		if row.is_empty():
			paused = true
			print("REPLAY_COMPLETE ticks=", replay.consumed_ticks)
			return
		replay.apply_state(sim, row.state)
		lap_time = sim.elapsed
		wheel_rotation += sim.speed * DT / 0.32
		return
	controls = _human_controls()
	_step(controls)
	if qa_target > 0:
		qa_ticks += 1
		if qa_ticks >= qa_target:
			print("QA_COMPLETE ", JSON.stringify(observation()))
			if recorder:
				recorder.flush()
			get_tree().quit()


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


func _step(action: Dictionary) -> Dictionary:
	var old_tick: int = sim.tick
	var road: Dictionary = track.sample_world(sim.position)
	var result: Dictionary = sim.step(DT, action, road)
	if result.has("error"):
		return result
	var after: Dictionary = track.sample_world(sim.position)
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
		terminated = true
	wheel_rotation += sim.speed * DT / 0.32
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
			"truncated": false
		}
	)


func _process(dt: float) -> void:
	_poll_agent()
	if engine_audio != null:
		engine_audio.muted = paused or agent_mode
	_update_visual(dt)
	if not paused and frame_times.size() < 72000:
		frame_times.append(dt)
	if not screenshot_path.is_empty() and Engine.get_process_frames() > 12:
		var path := screenshot_path
		screenshot_path = ""
		await RenderingServer.frame_post_draw
		var error := get_viewport().get_texture().get_image().save_png(path)
		print("SCREENSHOT ", path, " result=", error)
		if preview_mode:
			get_tree().quit()


func _update_visual(dt: float) -> void:
	if sim == null:
		return
	var road: Dictionary = track.sample_world(sim.position)
	bike_root.position = sim.position
	var riding_tangent: Vector3 = sim.surface_forward(road.normal)
	bike_root.rotation = Vector3(
		atan2(riding_tangent.y, Vector2(riding_tangent.x, riding_tangent.z).length()),
		-sim.heading,
		0
	)
	bike.update_pose(-sim.lean, -sim.steering, wheel_rotation)
	bike.update_instruments(sim.speed, sim.rpm, sim.gear)
	camera.set_cull_mask_value(20, camera_mode == 0)
	var forward := Vector3(sin(sim.heading), 0, -cos(sim.heading))
	var desired: Vector3
	var focus: Vector3
	if camera_mode == 0:
		desired = sim.position - forward * 5.5 + Vector3.UP * 2.35
		focus = sim.position + forward * 10 + Vector3.UP * 1.0
	else:
		desired = sim.position + Vector3.UP * 1.23 + forward * 0.05
		focus = desired + forward * 30 + Vector3.UP * (-0.3)
	camera.position = camera.position.lerp(desired, 1.0 - exp(-dt * 8.0))
	camera.look_at(focus, Vector3.UP)
	if camera_mode == 1:
		camera.rotate_object_local(Vector3.FORWARD, sim.lean * 0.22)


func _input(event: InputEvent) -> void:
	if not event is InputEventKey or not event.is_pressed() or event.is_echo():
		return
	if event.keycode == KEY_ESCAPE:
		paused = not paused
	if event.keycode == KEY_C:
		camera_mode = (camera_mode + 1) % 2
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
			paused = false
		"reset":
			reset_episode(0)
			paused = false
		"cyclone":
			reset_episode(1650)
			paused = false
		"camera":
			camera_mode = (camera_mode + 1) % 2
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
				return {
					"error": transition.error,
					"failure_type": "infrastructure",
					"rollout_valid": false,
					"episode_id": episode_id,
					"tick": sim.tick
				}
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
		"game_script_sha256": FileAccess.get_sha256("res://scripts/main.gd")
	}
