extends SceneTree
## Fixed geometry camera study against onboard footage. Does not change gameplay cameras.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 720)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var steering := 0.0
	var display_filtered := true
	var wall_density := 1.0
	var legacy_housing := false
	var sun_shadows := true
	var diagnostic_sun := false
	var shadow_bias := -1.0
	var normal_bias := -1.0
	var near_split := -1.0
	var shadow_blur := -1.0
	var depth32 := false
	var arguments := OS.get_cmdline_user_args()
	if "--display-filtered" in arguments and "--display-unfiltered" in arguments:
		_fail("Choose one display filtering mode")
		return
	for arg in arguments:
		if arg == "--shadow-depth32":
			depth32 = true
		elif arg == "--diagnostic-sun":
			diagnostic_sun = true
		elif arg == "--sun-shadows-off":
			sun_shadows = false
		elif arg.begins_with("--shadow-blur="):
			var value := arg.trim_prefix("--shadow-blur=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 4.0
			):
				_fail("Shadow blur must be between zero and four")
				return
			shadow_blur = float(value)
		elif arg.begins_with("--near-shadow-split="):
			var value := arg.trim_prefix("--near-shadow-split=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.005
				or float(value) > 0.2
			):
				_fail("Near shadow split must be between 0.005 and 0.2")
				return
			near_split = float(value)
		elif arg.begins_with("--shadow-normal-bias="):
			var value := arg.trim_prefix("--shadow-normal-bias=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 8.0
			):
				_fail("Normal bias must be finite and between zero and eight")
				return
			normal_bias = float(value)
		elif arg.begins_with("--shadow-bias="):
			var value := arg.trim_prefix("--shadow-bias=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 2.0
			):
				_fail("Shadow bias must be finite and between zero and two")
				return
			shadow_bias = float(value)
		elif arg == "--legacy-housing":
			legacy_housing = true
		elif arg == "--display-filtered":
			display_filtered = true
		elif arg == "--display-unfiltered":
			display_filtered = false
		elif arg.begins_with("--reservoir-wall-density="):
			var value := arg.trim_prefix("--reservoir-wall-density=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 200.0
			):
				_fail("Wall density multiplier must be between zero and 200")
				return
			wall_density = float(value)
		elif arg.begins_with("--steering="):
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
	if depth32:
		RenderingServer.directional_shadow_atlas_set_size(
			int(
				ProjectSettings.get_setting_with_override(
					"rendering/lights_and_shadows/directional_shadow/size"
				)
			),
			false
		)
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
	var sun: DirectionalLight3D
	for child in game.get_children():
		if child is DirectionalLight3D:
			sun = child
	if sun == null:
		_fail("Directional sunlight missing")
		return
	if diagnostic_sun:
		var toward: Vector3 = game.bike.global_basis * Vector3(0.3, 0.8, -0.5).normalized()
		sun.look_at(-toward, Vector3.UP)
	sun.shadow_enabled = sun_shadows
	if shadow_bias >= 0.0:
		sun.shadow_bias = shadow_bias
	if normal_bias >= 0.0:
		sun.shadow_normal_bias = normal_bias
	if near_split >= 0.0:
		sun.directional_shadow_split_1 = near_split
	if shadow_blur >= 0.0:
		sun.shadow_blur = shadow_blur
	if legacy_housing:
		var housing: MeshInstance3D = game.bike.find_child("InstrumentHousing", true, false)
		if housing == null:
			_fail("Instrument housing missing")
			return
		var temporary := Node3D.new()
		game.bike._beveled_panel(
			Vector2(0.196, 0.119), 0.025, 0.015, housing.material_override, temporary
		)
		housing.mesh = temporary.get_child(0).mesh
		temporary.free()
	var display_count := 0
	var reservoir_count := 0
	var wall_absorption := Vector3.ZERO
	for node in game.bike.find_children("*", "MeshInstance3D", true, false):
		var material = node.material_override
		if (
			material is ShaderMaterial
			and material.shader.resource_path == "res://shaders/instrument_screen.gdshader"
		):
			material.set_shader_parameter("footprint_filter", display_filtered)
			display_count += 1
		if (
			material is ShaderMaterial
			and material.shader.resource_path == "res://shaders/reservoir.gdshader"
		):
			var baseline: Vector3 = RenderingServer.shader_get_parameter_default(
				material.shader.get_rid(), "wall_absorption"
			)
			wall_absorption = baseline * wall_density
			material.set_shader_parameter("wall_absorption", wall_absorption)
			reservoir_count += 1
	if display_count != 1:
		_fail("Expected one live display material")
		return
	if reservoir_count != 2:
		_fail("Expected two reservoir materials")
		return
	variants[0].anchor = game.bike.ONBOARD_CAMERA_LOCAL
	variants[0].pitch = game.bike.ONBOARD_LOOK_DOWN
	variants[0].fov = game.camera.fov
	var report: Array = []
	var display_has_mipmaps := false
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
		if row.name == "production":
			var display_image: Image = game.bike._display_viewport.get_texture().get_image()
			display_has_mipmaps = display_image.has_mipmaps()
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
				"reservoir_wall_density": wall_density,
				"reservoir_wall_absorption":
				[wall_absorption.x, wall_absorption.y, wall_absorption.z],
				"reservoir_shader_sha256":
				FileAccess.get_sha256("res://shaders/reservoir.gdshader"),
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
							"display_filtered": display_filtered,
							"legacy_housing": legacy_housing,
							"sun_shadows": sun.shadow_enabled,
							"diagnostic_sun": diagnostic_sun,
							"toward_sun": str(sun.global_basis.z),
							"shadow_bias": sun.shadow_bias,
							"shadow_depth32_override": depth32,
							"shadow_blur": sun.shadow_blur,
							"near_shadow_split": sun.directional_shadow_split_1,
							"shadow_normal_bias": sun.shadow_normal_bias,
							"housing_builder_sha256":
							FileAccess.get_sha256("res://scripts/rounded_panel.gd"),
							"display_has_mipmaps": display_has_mipmaps,
							"display_shader_sha256":
							FileAccess.get_sha256("res://shaders/instrument_screen.gdshader"),
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
