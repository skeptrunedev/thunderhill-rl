extends SceneTree
## Fixed camera lighting study. Does not modify production environment settings.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 720)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var sky_source := ""
	var camera_mode := 2
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
		elif arg.begins_with("--sky-source="):
			sky_source = arg.trim_prefix("--sky-source=")
		elif arg.begins_with("--camera="):
			var value := arg.trim_prefix("--camera=")
			if not value.is_valid_int() or int(value) < 0 or int(value) > 2:
				_fail("Camera must be 0, 1 or 2")
				return
			camera_mode = int(value)
	if not output.is_absolute_path() or DisplayServer.get_name() == "headless":
		_fail("Use a real renderer and an absolute --output-dir")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create output directory")
		return
	var variants := [
		{
			"name": "sky_chroma",
			"mapper": Environment.TONE_MAPPER_FILMIC,
			"exposure": 1.0,
			"sky": 0.7,
			"saturation": 1.6,
			"sun": "fff0d5",
			"fog": 0.00035
		},
		{
			"name": "background_low",
			"mapper": Environment.TONE_MAPPER_FILMIC,
			"exposure": 1.0,
			"sky": 0.4,
			"sun": "fff0d5",
			"fog": 0.00035
		},
		{
			"name": "sky_only",
			"mapper": Environment.TONE_MAPPER_FILMIC,
			"exposure": 1.0,
			"sky": 0.7,
			"sun": "fff0d5",
			"fog": 0.00035
		},
		{
			"name": "baseline",
			"mapper": Environment.TONE_MAPPER_FILMIC,
			"exposure": 1.0,
			"sky": 1.0,
			"sun": "fff0d5",
			"fog": 0.00035
		},
		{
			"name": "neutral",
			"mapper": Environment.TONE_MAPPER_FILMIC,
			"exposure": 0.85,
			"sky": 0.7,
			"sun": "fff8ee",
			"fog": 0.00016
		},
		{
			"name": "aces",
			"mapper": Environment.TONE_MAPPER_ACES,
			"exposure": 1.0,
			"sky": 1.0,
			"sun": "fff0d5",
			"fog": 0.00035
		},
		{
			"name": "aces_neutral",
			"mapper": Environment.TONE_MAPPER_ACES,
			"exposure": 0.85,
			"sky": 0.7,
			"sun": "fff8ee",
			"fog": 0.00016
		}
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
	if game.bike == null:
		_fail("Game failed to initialize")
		return
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.hud.visible = false
	game.camera_mode = camera_mode
	game.reset_episode(400.0)
	game._update_visual(1.0)
	var environment: Environment
	var sun: DirectionalLight3D
	for child in game.get_children():
		if child is WorldEnvironment:
			environment = child.environment
		elif child is DirectionalLight3D:
			sun = child
	if environment == null or sun == null:
		_fail("Game lighting missing")
		return
	if not sky_source.is_empty():
		var source: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(sky_source))
		var image := Image.load_from_file(source.local_path)
		if image == null or image.is_empty():
			_fail("Cannot load candidate sky")
			return
		if FileAccess.get_sha256(source.local_path) != source.sha256:
			_fail("Candidate sky differs from manifest")
			return
		image.generate_mipmaps()
		environment.sky.sky_material.set_shader_parameter(
			"panorama", ImageTexture.create_from_image(image)
		)
		var direction: Array = source.toward_sun
		sun.look_at(-Vector3(direction[0], direction[1], direction[2]), Vector3.UP)
	for row in variants:
		environment.tonemap_mode = row.mapper
		environment.tonemap_exposure = row.exposure
		environment.fog_density = row.fog
		sun.light_color = Color(row.sun)
		environment.sky.sky_material.set_shader_parameter("background_energy", row.sky)
		environment.sky.sky_material.set_shader_parameter(
			"background_saturation", row.get("saturation", 1.0)
		)
		for frame in 24:
			await RenderingServer.frame_post_draw
		var capture := root.get_texture().get_image()
		if (
			Validation.classify(capture, SIZE) != "nonblack"
			or capture.save_png(output.path_join(row.name + ".png")) != OK
		):
			_fail("Capture failed: " + row.name)
			return
	var file := FileAccess.open(output.path_join("study.json"), FileAccess.WRITE)
	if file == null:
		_fail("Cannot write study")
		return
	file.store_string(
		(
			JSON.stringify(
				{
					"sky_source": sky_source,
					"station_m": 400,
					"camera": camera_mode,
					"fov": game.camera.fov,
					"camera_transform": str(game.camera.global_transform),
					"variants": variants,
					"limitation": "Artistic appearance study, not calibrated radiometry"
				},
				"  "
			)
			+ "\n"
		)
	)
	file.flush()
	var error := file.get_error()
	file.close()
	if error != OK:
		_fail("Study write failed")
		return
	print("LIGHTING_STUDY output=", output)
	quit()


func _fail(message: String) -> void:
	push_error(message)
	quit(2)
