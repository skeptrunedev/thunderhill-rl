extends SceneTree
## Paired visual diagnostic only, not a performance benchmark.
const Validation = preload("res://scripts/image_validation.gd")
var output := ""
var materials: Array[StandardMaterial3D] = []
var unshaded := false
var frames := 16


func _initialize() -> void:
	_run.call_deferred()


func _fail(message: String) -> void:
	push_error(message)
	quit(2)


func _collect(node: Node) -> void:
	if node is MultiMeshInstance3D and str(node.name).begins_with("DryGrass_"):
		for surface in node.multimesh.mesh.get_surface_count():
			var material: Material = node.multimesh.mesh.surface_get_material(surface)
			if material is StandardMaterial3D and material not in materials:
				materials.append(material)
	for child in node.get_children():
		_collect(child)


func _run() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
		if arg == "--unshaded":
			unshaded = true
		if arg.begins_with("--frames="):
			frames = int(arg.trim_prefix("--frames="))
	if frames < 1 or frames > 120:
		_fail("Frame count must be between 1 and 120")
		return
	if output.is_empty() or not output.is_absolute_path():
		_fail("Supply --output-dir=/absolute/path")
		return
	if DisplayServer.get_name() == "headless":
		_fail("Grass preview requires a real renderer")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create output directory")
		return
	var modes := {
		"none": BaseMaterial3D.ALPHA_ANTIALIASING_OFF,
		"coverage": BaseMaterial3D.ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE
	}
	var paths := [output.path_join("comparison.json")]
	for mode in modes:
		for index in frames:
			paths.append(output.path_join("%s_%02d.png" % [mode, index]))
	for path in paths:
		if FileAccess.file_exists(path):
			_fail("Refusing to overwrite " + path)
			return
	root.size = Vector2i(1920, 1080)
	root.content_scale_size = root.size
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.set_process(false)
	game.set_physics_process(false)
	game.bike.visible = false
	game.hud.visible = false
	_collect(game)
	if materials.is_empty():
		_fail("Baked grass materials missing")
		return
	if unshaded:
		for material in materials:
			material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	var track = game.track
	var nearest := 0
	for index in track.samples.size():
		if (
			absf(float(track.samples[index].s) - 1830.0)
			< absf(float(track.samples[nearest].s) - 1830.0)
		):
			nearest = index
	var center: Vector3 = track.points[nearest]
	var road: Dictionary = track.sample_world(center)
	var forward := Vector3(road.tangent.x, 0.0, road.tangent.z).normalized()
	var left: Vector3 = road.left
	var origin: Vector3 = center + left * (float(road.width) * 0.5 + 0.4)
	game.camera.fov = 74.0
	game.camera.current = true
	var rows: Array = []
	for frame in 10:
		await RenderingServer.frame_post_draw
	for index in frames:
		var position := origin + forward * (0.30 * index)
		position.y = track.terrain_surface_height(position) + 1.5
		game.camera.global_position = position
		game.camera.look_at(position + forward * 12.0 + left * 5.0 - Vector3.UP)
		for mode in modes:
			for material in materials:
				material.alpha_antialiasing_mode = modes[mode]
				material.alpha_antialiasing_edge = 0.30
				material.alpha_scissor_threshold = 0.35
			for frame in 3:
				await RenderingServer.frame_post_draw
			var image := root.get_texture().get_image()
			var validation := Validation.classify(image, Vector2i(root.get_texture().get_size()))
			var filename := "%s_%02d.png" % [mode, index]
			if validation != "nonblack" or image.save_png(output.path_join(filename)) != OK:
				_fail("Capture failed: " + filename + " validation=" + validation)
				return
			var basis: Basis = game.camera.global_basis
			rows.append(
				{
					"mode": mode,
					"frame": index,
					"filename": filename,
					"validation": validation,
					"position": [position.x, position.y, position.z],
					"basis":
					[
						[basis.x.x, basis.x.y, basis.x.z],
						[basis.y.x, basis.y.y, basis.y.z],
						[basis.z.x, basis.z.y, basis.z.z]
					]
				}
			)
	var report := {
		"purpose": "visual comparison, not performance benchmark",
		"station_m": track.samples[nearest].s,
		"step_m": 0.30,
		"frames_per_mode": frames,
		"unshaded_grass": unshaded,
		"alpha_modes": modes,
		"alpha_edge": 0.30,
		"alpha_cutoff": 0.35,
		"msaa_3d": root.msaa_3d,
		"fov": game.camera.fov,
		"window_size": [root.size.x, root.size.y],
		"image_size": [root.get_texture().get_width(), root.get_texture().get_height()],
		"material_count": materials.size(),
		"engine": Engine.get_version_info().string,
		"samples": rows
	}
	var file := FileAccess.open(paths[0], FileAccess.WRITE)
	if file == null:
		_fail("Cannot write comparison metadata")
		return
	file.store_string(JSON.stringify(report, "  ") + "\n")
	file.flush()
	var error := file.get_error()
	file.close()
	if error != OK:
		_fail("Failed writing comparison metadata")
		return
	print("GRASS_PREVIEW captures=", rows.size(), " output=", output)
	quit()
