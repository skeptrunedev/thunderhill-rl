extends SceneTree
## Inspect one mapped building under the game's actual lighting and materials.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 800)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var osm_way := 44958778
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
		elif arg.begins_with("--osm-way="):
			osm_way = int(arg.trim_prefix("--osm-way="))
	if output.is_empty() or not output.is_absolute_path() or DisplayServer.get_name() == "headless":
		_fail("Use a real renderer and --output-dir=/absolute/path")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create output directory")
		return
	for name in ["paddock.png", "trackside.png", "inspection.json"]:
		if FileAccess.file_exists(output.path_join(name)):
			_fail("Refusing to overwrite " + name)
			return
	var data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/landmarks.json")
	)
	var row: Dictionary = {}
	for building: Dictionary in data.buildings:
		if building.osm_way == osm_way:
			row = building
	if row.is_empty():
		_fail("Requested building is not mapped")
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
	var low := Vector2(INF, INF)
	var high := Vector2(-INF, -INF)
	for p: Array in row.points:
		low = low.min(Vector2(p[0], p[1]))
		high = high.max(Vector2(p[0], p[1]))
	var center := (low + high) * 0.5
	var base := Vector3(center.x, 0, center.y)
	base.y = game.track.terrain_surface_height(base)
	var target := base + Vector3.UP * 4.0
	var distance := (high - low).length() * 0.95
	var captures: Array = []
	for side: String in ["paddock", "trackside"]:
		var offset := Vector3(-0.65, 0.45, 0.75) if side == "paddock" else Vector3(0.85, 0.12, 0.45)
		game.camera.global_position = target + offset.normalized() * distance
		game.camera.look_at(target)
		game.camera.fov = 60.0
		game.camera.current = true
		for frame in 10:
			await RenderingServer.frame_post_draw
		var capture := root.get_texture().get_image()
		if (
			Validation.classify(capture, SIZE) != "nonblack"
			or capture.save_png(output.path_join(side + ".png")) != OK
		):
			_fail("Capture failed: " + side)
			return
		var p: Vector3 = game.camera.global_position
		captures.append(
			{
				"image": side + ".png",
				"camera_position": [p.x, p.y, p.z],
				"camera_target": [target.x, target.y, target.z],
				"fov": game.camera.fov
			}
		)
	var file := FileAccess.open(output.path_join("inspection.json"), FileAccess.WRITE)
	if file == null:
		_fail("Cannot write inspection metadata")
		return
	file.store_string(
		(
			JSON.stringify(
				{
					"osm_way": osm_way,
					"size": [SIZE.x, SIZE.y],
					"captures": captures,
					"landmarks_bake_sha256": FileAccess.get_sha256("res://data/landmarks-bake.json")
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
		_fail("Cannot save inspection metadata")
		return
	print("BUILDING_PREVIEW output=", output)
	quit()


func _fail(message: String) -> void:
	push_error(message)
	quit(2)
