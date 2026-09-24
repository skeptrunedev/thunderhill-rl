extends SceneTree
var failures := 0


class FaultyCamera:
	extends AgentCamera
	var frame: Image
	var reads := 0

	func _read_image() -> Image:
		reads += 1
		return frame


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("Camera failure integration requires a real renderer")
		quit(2)
		return
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.agent_mode = true
	var camera := FaultyCamera.new()
	game.add_child(camera)
	camera.configure(game.get_world_3d())
	game.agent_camera = camera
	for reason in ["all_black", "wrong_dimensions", "empty_image"]:
		game.reset_episode(0.0)
		if reason == "empty_image":
			camera.frame = null
		else:
			var size := Vector2i(AgentCamera.WIDTH, AgentCamera.HEIGHT) if reason == "all_black" else Vector2i(8, 8)
			camera.frame = Image.create(size.x, size.y, false, Image.FORMAT_RGBA8)
			camera.frame.fill(Color.BLACK if reason == "all_black" else Color.RED)
		var request := {"episode_id": game.episode_id, "expected_tick": 0}
		var response: Dictionary = await game._capture_request(request)
		check(response.get("validation") == reason, "Readback failure reason lost")
		check(response.get("failure_type") == "infrastructure", "Capture error misclassified")
		check(not response.get("rollout_valid", true), "Failed capture left rollout valid")
		check(
			not response.has("image") and not response.has("observation_id"),
			"Invalid image accepted"
		)
		check(game.sim.tick == 0, "Capture failure advanced physics")
		var reads := camera.reads
		var repeated: Dictionary = await game._capture_request(request)
		check(repeated == response and camera.reads == reads, "Failure repeated GPU readback")
		var advance: Dictionary = game._request(
			{
				"op": "advance",
				"episode_id": game.episode_id,
				"expected_tick": 0,
				"action_id": "after-failed-capture",
				"controls": {"throttle": 0.8}
			}
		)
		check(advance.has("error") and game.sim.tick == 0, "Failed rollout advanced")
		game.recorder.flush()
		var errors := 0
		var observations := 0
		for line in FileAccess.get_file_as_string(game.recorder.get_path()).split("\n", false):
			var row: Dictionary = JSON.parse_string(line)
			if row.get("episode_id") != game.episode_id:
				continue
			errors += int(row.get("type") == "environment_failure")
			observations += int(row.get("type") == "camera_observation")
		check(errors == 1 and observations == 0, "Failure recording was missing or duplicated")
	game.reset_episode(0.0)
	check(game.observation().rollout_valid, "Reset did not clear capture failure")
	game.queue_free()
	await process_frame
	print("CAMERA_FAILURE_CHECK failures=", failures)
	quit(failures)
