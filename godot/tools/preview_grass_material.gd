extends SceneTree
## Compare a candidate albedo under identical geometry, lighting and camera.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 800)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var candidate := ""
	var frames := 1
	var road_edge := false
	var detail_gain_exponent := 1.0
	var station_m := 1830.0
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--station="):
			var value := arg.trim_prefix("--station=")
			if not value.is_valid_float():
				_fail("Station must be numeric")
				return
			station_m = float(value)
		elif arg.begins_with("--detail-gain-exponent="):
			var value := arg.trim_prefix("--detail-gain-exponent=")
			if not value.is_valid_float():
				_fail("Detail gain exponent must be numeric")
				return
			detail_gain_exponent = float(value)
		elif arg == "--road-edge":
			road_edge = true
		elif arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
		elif arg.begins_with("--frames="):
			frames = int(arg.trim_prefix("--frames="))
		elif arg.begins_with("--candidate="):
			candidate = arg.trim_prefix("--candidate=")
	if (
		not is_finite(detail_gain_exponent)
		or detail_gain_exponent < 0.5
		or detail_gain_exponent > 3.0
	):
		_fail("Detail gain exponent must be between 0.5 and 3")
		return
	if frames < 1 or frames > 120:
		_fail("Frame count must be between 1 and 120")
		return
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
	var filenames := ["comparison.json"]
	for name: String in ["existing", "candidate"]:
		for index in frames:
			filenames.append(name + ".png" if frames == 1 else "%s_%02d.png" % [name, index])
	for filename in filenames:
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
	if not is_finite(station_m) or station_m < 0.0 or station_m >= track.length_m:
		_fail("Station must be within the lap")
		return
	var nearest := 0
	for i in track.samples.size():
		if (
			absf(float(track.samples[i].s) - station_m)
			< absf(float(track.samples[nearest].s) - station_m)
		):
			nearest = i
	var center: Vector3 = track.points[nearest]
	var road: Dictionary = track.sample_world(center)
	var forward := Vector3(road.tangent.x, 0, road.tangent.z).normalized()
	var left: Vector3 = road.left
	var position: Vector3 = center + left * (float(road.width) * 0.5 + (-0.6 if road_edge else 4.0))
	position.y = track.terrain_surface_height(position) + 1.5
	var target := position + forward * 8.0 + left * (1.5 if road_edge else 3.0) - Vector3.UP * 1.0
	game.camera.global_position = position
	game.camera.look_at(target)
	game.camera.fov = 74.0
	game.camera.current = true
	var material: ShaderMaterial = track.terrain_material
	material.set_shader_parameter("detail_gain_exponent", detail_gain_exponent)
	var original: Texture2D = material.get_shader_parameter("grass_color")
	var samples: Array = []
	for index in frames:
		var pose := position + forward * (0.30 * index)
		pose.y = track.terrain_surface_height(pose) + 1.5
		game.camera.global_position = pose
		game.camera.look_at(pose + target - position)
		for name: String in ["existing", "candidate"]:
			material.set_shader_parameter(
				"grass_color", original if name == "existing" else texture
			)
			for frame in 10:
				await RenderingServer.frame_post_draw
			var capture := root.get_texture().get_image()
			var filename := name + ".png" if frames == 1 else "%s_%02d.png" % [name, index]
			if (
				Validation.classify(capture, SIZE) != "nonblack"
				or capture.save_png(output.path_join(filename)) != OK
			):
				_fail("Capture failed: " + filename)
				return
			samples.append({"filename": filename, "position": [pose.x, pose.y, pose.z]})
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
							"existing": original.resource_path,
							"frames_per_material": frames,
							"detail_gain_exponent": detail_gain_exponent,
							"requested_station_m": station_m,
							"sample_station_m": float(track.samples[nearest].s),
							"step_m": 0.30,
							"samples": samples,
							"view": "road_edge" if road_edge else "field",
							"candidate": candidate,
							"candidate_sha256": FileAccess.get_sha256(candidate),
							"source_size": [source.get_width(), source.get_height()],
							"tile_m": 2.0,
							"camera_position": [position.x, position.y, position.z],
							"camera_target": [target.x, target.y, target.z],
							"fov": 74.0,
							"limitation":
							"Both albedos use the production shader luminance based relief and roughness, not measured PBR maps"
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
