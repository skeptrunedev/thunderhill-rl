extends SceneTree
## Compare historical tone with the same geometry, view and lighting.
const SIZE := Vector2i(1280, 720)
const Validation = preload("res://scripts/image_validation.gd")


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
	if not output.is_absolute_path() or DisplayServer.get_name() == "headless":
		push_error("Use real renderer and absolute output directory")
		quit(2)
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		quit(2)
		return
	if FileAccess.file_exists(output.path_join("study.json")):
		push_error("Refusing to overwrite pavement study metadata")
		quit(2)
		return
	for station in [400, 900, 3000]:
		for label in ["procedural", "historical"]:
			if FileAccess.file_exists(output.path_join("%d_%s.png" % [station, label])):
				push_error("Refusing to overwrite pavement study")
				quit(2)
				return
	root.size = SIZE
	root.content_scale_size = SIZE
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.hud.visible = false
	game.camera_mode = 1
	var material: ShaderMaterial = game.track.get_node("RacingSurface").material_override
	var report: Array = []
	for station in [400, 900, 3000]:
		game.reset_episode(float(station))
		game._update_visual(1.0)
		for strength in [0.0, 0.65]:
			material.set_shader_parameter("pavement_tone_strength", strength)
			for frame in 24:
				await RenderingServer.frame_post_draw
			var capture := root.get_texture().get_image()
			var label := "procedural" if strength == 0.0 else "historical"
			var name := "%d_%s.png" % [station, label]
			if (
				Validation.classify(capture, SIZE) != "nonblack"
				or capture.save_png(output.path_join(name)) != OK
			):
				quit(2)
				return
			report.append(
				{
					"image": name,
					"requested_station_m": station,
					"strength": strength,
					"camera_transform": str(game.camera.global_transform)
				}
			)
	var file := FileAccess.open(output.path_join("study.json"), FileAccess.WRITE)
	if file == null:
		quit(2)
		return
	file.store_string(
		JSON.stringify(
			{
				"captures": report,
				"shader_sha256": FileAccess.get_sha256("res://shaders/asphalt.gdshader"),
				"tone_sha256": FileAccess.get_sha256("res://assets/materials/pavement_tone.png"),
				"limitation":
				"Historical appearance, not recovered pavement reflectance or surveyed friction"
			},
			"  "
		)
	)
	file.flush()
	var error := file.get_error()
	file.close()
	print("PAVEMENT_STUDY error=", error)
	quit(0 if error == OK else 2)
