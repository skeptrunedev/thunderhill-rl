extends SceneTree
## Matched shadow comparisons, including a diagnostic sun behind the rider.
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
		_fail("Use a real renderer and absolute output directory")
		return
	if DirAccess.dir_exists_absolute(output):
		_fail("Refusing to overwrite a shadow study directory")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create shadow study directory")
		return
	root.size = SIZE
	root.content_scale_size = SIZE
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.hud.visible = false
	var casters: Array[MeshInstance3D] = []
	for node in game.bike.find_children("RiderShadow", "MeshInstance3D", true, false):
		casters.append(node)
	if casters.is_empty():
		_fail("Rider shadow casters missing")
		return
	var sun: DirectionalLight3D
	for child in game.get_children():
		if child is DirectionalLight3D:
			sun = child
	var production_sun := sun.global_transform
	var report: Array = []
	for view in [
		{"name": "rider", "camera": 1, "lean": 0.0},
		{"name": "onboard", "camera": 2, "lean": 0.0},
		{"name": "policy", "camera": 1, "lean": 0.0},
		{"name": "lean", "camera": 1, "lean": 0.35},
		{"name": "production", "camera": 1, "lean": 0.0},
	]:
		game.reset_episode(1300.0)
		game.camera_mode = view.camera
		game.sim.lean = view.lean
		game._update_visual(1.0)
		if view.name == "policy":
			# Use the policy front wide mount in this presentation viewport so the
			# paired diagnostic can compare full size images without recording episodes.
			# The presentation camera is a pinhole, not the policy F-theta lens.
			game.camera.global_transform = AgentCamera.view_transform(game.sim, AgentCamera.VIEWS[1])
			game.camera.fov = 74.0
			game.camera.cull_mask = (
				((1 << 20) - 1)
				& ~(AgentCamera.BIKE_LAYER | AgentCamera.RIDER_LAYER | AgentCamera.RIDER_LIMB_LAYER)
			)
		else:
			game.camera.set_cull_mask_value(19, true)
		if view.name == "production":
			sun.global_transform = production_sun
		else:
			var forward := Vector3(sin(game.sim.heading), 0.0, -cos(game.sim.heading))
			var toward := -forward * sqrt(0.75) + Vector3.UP * 0.5
			sun.look_at(-toward, Vector3.UP)
		for enabled in [false, true]:
			for caster in casters:
				caster.visible = enabled
				caster.get_parent().cast_shadow = (
					GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
					if enabled
					else GeometryInstance3D.SHADOW_CASTING_SETTING_ON
				)
			for frame in 12:
				await RenderingServer.frame_post_draw
			var capture := root.get_texture().get_image()
			var filename: String = view.name + ("_corrected.png" if enabled else "_legacy.png")
			if (
				Validation.classify(capture, SIZE) != "nonblack"
				or capture.save_png(output.path_join(filename)) != OK
			):
				_fail("Shadow capture failed")
				return
			report.append(
				{
					"image": filename,
					"shadow_casters_enabled": enabled,
					"camera_transform": str(game.camera.global_transform),
					"camera_mask": game.camera.cull_mask,
					"fov": game.camera.fov,
					"toward_sun": str(sun.global_basis.z),
					"lean_rad": view.lean
				}
			)
	var file := FileAccess.open(output.path_join("study.json"), FileAccess.WRITE)
	if file == null:
		_fail("Cannot write shadow study")
		return
	(
		file
		. store_string(
			(
				JSON
				. stringify(
					{
						"captures": report,
						"caster_count": casters.size(),
						"near_shadow_split": sun.directional_shadow_split_1,
						"blend_shadow_splits": sun.directional_shadow_blend_splits,
						"station_m": 1300.0,
						"shadow_script_sha256":
						FileAccess.get_sha256("res://scripts/rider_shadows.gd"),
						"limitation":
						"Diagnostic light is not reconstructed footage illumination. Policy pose is reproduced in the presentation viewport, not an observation API test."
					},
					"  "
				)
			)
		)
	)
	file.flush()
	var error := file.get_error()
	file.close()
	print("RIDER_SHADOW_STUDY error=", error)
	quit(0 if error == OK else 2)


func _fail(message: String) -> void:
	push_error(message)
	quit(2)
