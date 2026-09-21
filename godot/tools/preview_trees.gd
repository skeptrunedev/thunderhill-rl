extends SceneTree
## Paired rendered color diagnostic, not a performance benchmark.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 800)
var output := ""
var crowns: MultiMeshInstance3D


func _initialize() -> void:
	_run.call_deferred()


func _fail(message: String) -> void:
	push_error(message)
	quit(2)


func _collect(node: Node) -> void:
	if node is MultiMeshInstance3D and node.name == "ObservedTreeCrowns":
		crowns = node
	for child in node.get_children():
		_collect(child)


func _vector(value: Vector3) -> Array:
	return [value.x, value.y, value.z]


func _run() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
	if output.is_empty() or not output.is_absolute_path():
		_fail("Supply --output-dir=/absolute/path")
		return
	if DisplayServer.get_name() == "headless":
		_fail("Tree preview requires a real renderer")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create output directory")
		return
	for filename in ["linear.png", "srgb.png", "comparison.json"]:
		if FileAccess.file_exists(output.path_join(filename)):
			_fail("Refusing to overwrite " + filename)
			return
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
	_collect(game)
	if crowns == null or not crowns.material_override is StandardMaterial3D:
		_fail("Baked tree crowns or expected material missing")
		return
	var material := crowns.material_override as StandardMaterial3D
	var default_srgb := material.vertex_color_is_srgb
	var data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/landmarks.json")
	)
	# Choose the mapped crown closest to the local origin, retaining its data row.
	var selected := 0
	for index in data.trees.size():
		var p: Array = data.trees[index].p
		var best: Array = data.trees[selected].p
		if Vector2(p[0], p[1]).length_squared() < Vector2(best[0], best[1]).length_squared():
			selected = index
	var row: Dictionary = data.trees[selected]
	var base := Vector3(row.p[0], 0, row.p[1])
	base.y = game.track.terrain_surface_height(base)
	var target := base + Vector3.UP * float(row.height_m) * 0.65
	var position := target + Vector3(1, 0, -1).normalized() * float(row.radius_m) * 2.5
	game.camera.global_position = position
	game.camera.look_at(target)
	game.camera.fov = 74.0
	game.camera.current = true
	var minimum := Color(1, 1, 1, 1)
	var maximum := Color(0, 0, 0, 0)
	var count := 0
	for surface in crowns.multimesh.mesh.get_surface_count():
		var colors: PackedColorArray = crowns.multimesh.mesh.surface_get_arrays(surface)[
			Mesh.ARRAY_COLOR
		]
		for color in colors:
			for channel in 4:
				minimum[channel] = minf(minimum[channel], color[channel])
				maximum[channel] = maxf(maximum[channel], color[channel])
			count += 1
	if count == 0:
		_fail("Tree mesh has no vertex colors")
		return
	for frame in 10:
		await RenderingServer.frame_post_draw
	var captures: Array = []
	for mode in [false, true]:
		material.vertex_color_is_srgb = mode
		for frame in 3:
			await RenderingServer.frame_post_draw
		var image := root.get_texture().get_image()
		var validation := Validation.classify(image, SIZE)
		var filename := "srgb.png" if mode else "linear.png"
		if validation != "nonblack" or image.save_png(output.path_join(filename)) != OK:
			_fail("Capture failed: " + filename + " validation=" + validation)
			return
		captures.append(
			{
				"filename": filename,
				"vertex_color_is_srgb": mode,
				"validation": validation,
				"image_size": [image.get_width(), image.get_height()]
			}
		)
	var basis: Basis = game.camera.global_basis
	var report := {
		"purpose": "Tree vertex color comparison under unchanged game lighting",
		"engine": Engine.get_version_info().string,
		"default_vertex_color_is_srgb": default_srgb,
		"vertex_color_use_as_albedo": material.vertex_color_use_as_albedo,
		"tree_index": selected,
		"tree": row,
		"node_path": str(crowns.get_path()),
		"camera_position": _vector(position),
		"camera_target": _vector(target),
		"camera_basis": [_vector(basis.x), _vector(basis.y), _vector(basis.z)],
		"fov": game.camera.fov,
		"vertex_color_count": count,
		"vertex_color_min_rgba": [minimum.r, minimum.g, minimum.b, minimum.a],
		"vertex_color_max_rgba": [maximum.r, maximum.g, maximum.b, maximum.a],
		"captures": captures
	}
	var file := FileAccess.open(output.path_join("comparison.json"), FileAccess.WRITE)
	if file == null:
		_fail("Cannot open comparison metadata")
		return
	file.store_string(JSON.stringify(report, "  ") + "\n")
	file.flush()
	var error := file.get_error()
	file.close()
	if error != OK:
		_fail("Failed writing comparison metadata")
		return
	print("TREE_PREVIEW captures=", captures.size(), " output=", output)
	quit()
