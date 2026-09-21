extends SceneTree
## Compare a candidate albedo under identical geometry, lighting and camera.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 800)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var candidate := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
		elif arg.begins_with("--candidate="):
			candidate = arg.trim_prefix("--candidate=")
	if (
		not output.is_absolute_path()
		or not candidate.is_absolute_path()
		or DisplayServer.get_name() == "headless"
	):
		_fail("Use a real renderer, --output-dir and --candidate with absolute paths")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create output directory")
		return
	for filename in ["existing.png", "candidate.png", "comparison.json"]:
		if FileAccess.file_exists(output.path_join(filename)):
			_fail("Refusing to overwrite " + filename)
			return
	var source := Image.load_from_file(candidate)
	if source == null or source.is_empty() or source.generate_mipmaps() != OK:
		_fail("Cannot load candidate and generate mipmaps")
		return
	var texture := ImageTexture.create_from_image(source)
	root.size = SIZE
	root.content_scale_size = SIZE
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.set_process(false)
	game.set_physics_process(false)
	game.bike.visible = false
	game.hud.visible = false
	var track = game.track
	var nearest := 0
	for i in track.samples.size():
		if (
			absf(float(track.samples[i].s) - 1830.0)
			< absf(float(track.samples[nearest].s) - 1830.0)
		):
			nearest = i
	var center: Vector3 = track.points[nearest]
	var road: Dictionary = track.sample_world(center)
	var forward := Vector3(road.tangent.x, 0, road.tangent.z).normalized()
	var left: Vector3 = road.left
	var position: Vector3 = center + left * (float(road.width) * 0.5 + 4.0)
	position.y = track.terrain_surface_height(position) + 1.5
	var target := position + forward * 8.0 + left * 3.0 - Vector3.UP * 1.0
	game.camera.global_position = position
	game.camera.look_at(target)
	game.camera.fov = 74.0
	game.camera.current = true
	var material: ShaderMaterial = track.terrain_material
	var original: Texture2D = material.get_shader_parameter("grass_color")
	for name: String in ["existing", "candidate"]:
		material.set_shader_parameter("grass_color", original if name == "existing" else texture)
		for frame in 10:
			await RenderingServer.frame_post_draw
		var capture := root.get_texture().get_image()
		if (
			Validation.classify(capture, SIZE) != "nonblack"
			or capture.save_png(output.path_join(name + ".png")) != OK
		):
			_fail("Capture failed: " + name)
			return
	var file := FileAccess.open(output.path_join("comparison.json"), FileAccess.WRITE)
	if file == null:
		_fail("Cannot write comparison")
		return
	(
		file
		. store_string(
			(
				(
					JSON
					. stringify(
						{
							"candidate": candidate,
							"candidate_sha256": FileAccess.get_sha256(candidate),
							"source_size": [source.get_width(), source.get_height()],
							"tile_m": 2.0,
							"camera_position": [position.x, position.y, position.z],
							"camera_target": [target.x, target.y, target.z],
							"fov": 74.0,
							"limitation":
							"Albedo study only: both use existing grass normal and roughness; these are not matching PBR maps for the candidate"
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
		_fail("Comparison write failed")
		return
	print("GRASS_MATERIAL_PREVIEW output=", output)
	quit()


func _fail(message: String) -> void:
	push_error(message)
	quit(2)
