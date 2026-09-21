extends SceneTree
## Fixed geometry camera study against onboard footage. Does not change gameplay cameras.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 720)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var steering := 0.0
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--steering="):
			var value := arg.trim_prefix("--steering=")
			if not value.is_valid_float():
				_fail("Steering must be numeric radians")
				return
			steering = float(value)
		elif arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
	if not output.is_absolute_path() or DisplayServer.get_name() == "headless":
		_fail("Use a real renderer and an absolute --output-dir")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create output directory")
		return
	var variants := [
		{"name": "production", "anchor": Vector3(0, 1.14, -0.30), "pitch": 0.40, "fov": 74.0},
		{"name": "raised", "anchor": Vector3(-0.04, 1.28, -0.26), "pitch": 0.48, "fov": 80.0},
		{"name": "forward", "anchor": Vector3(-0.04, 1.22, -0.37), "pitch": 0.45, "fov": 85.0},
		{"name": "wide", "anchor": Vector3(-0.04, 1.20, -0.27), "pitch": 0.45, "fov": 90.0}
	]
	for row in variants:
		if FileAccess.file_exists(output.path_join(row.name + ".png")):
			_fail("Refusing to overwrite captures")
			return
	if FileAccess.file_exists(output.path_join("study.json")):
		_fail("Refusing to overwrite metadata")
		return
	root.size = SIZE
	root.content_scale_size = SIZE
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.hud.visible = false
	game.camera_mode = 2
	game.reset_episode(400.0)
	game._update_visual(1.0)
	if not is_finite(steering) or absf(steering) > float(game.sim.parameters.steering_limit_rad):
		_fail("Steering exceeds simulation limits")
		return
	game.bike.update_pose(0.0, steering, 0.0)
	variants[0].anchor = game.bike.ONBOARD_CAMERA_LOCAL
	variants[0].pitch = game.bike.ONBOARD_LOOK_DOWN
	variants[0].fov = game.camera.fov
	var report: Array = []
	for row in variants:
		if row.name != "production":
			game.camera.global_position = game.bike.to_global(row.anchor)
			var direction := Vector3(0, -sin(row.pitch), -cos(row.pitch))
			game.camera.look_at(
				game.camera.position + game.bike.global_basis * direction, game.bike.global_basis.y
			)
			game.camera.fov = row.fov
		for frame in 24:
			await RenderingServer.frame_post_draw
		var capture := root.get_texture().get_image()
		if (
			Validation.classify(capture, SIZE) != "nonblack"
			or capture.save_png(output.path_join(row.name + ".png")) != OK
		):
			_fail("Capture failed: " + row.name)
			return
		report.append(
			{
				"name": row.name,
				"anchor": [row.anchor.x, row.anchor.y, row.anchor.z],
				"pitch_rad": row.pitch,
				"fov": row.fov,
				"camera_transform": str(game.camera.global_transform)
			}
		)
	var file := FileAccess.open(output.path_join("study.json"), FileAccess.WRITE)
	if file == null:
		_fail("Cannot write metadata")
		return
	(
		file
		. store_string(
			(
				(
					JSON
					. stringify(
						{
							"station_m": 400,
							"visual_steering_rad": steering,
							"resolution": [SIZE.x, SIZE.y],
							"main_script_sha256": FileAccess.get_sha256("res://scripts/main.gd"),
							"study_script_sha256":
							FileAccess.get_sha256("res://tools/preview_cockpit.gd"),
							"bike_script_sha256":
							FileAccess.get_sha256("res://scripts/bike_visual.gd"),
							"variants": report,
							"limitation":
							"Original framing estimates, not recovered lens calibration; geometry and lighting held fixed"
						},
						"  "
					)
				)
				+ "\n"
			)
		)
	)
	file.flush()
	var error := file.get_error()
	file.close()
	if error != OK:
		_fail("Metadata write failed")
		return
	print("COCKPIT_STUDY output=", output)
	quit()


func _fail(message: String) -> void:
	push_error(message)
	quit(2)
