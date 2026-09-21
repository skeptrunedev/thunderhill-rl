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
	var ground_tint := Vector3.ZERO
	var field_soil_strength := -1.0
	var field_patch_strength := -1.0
	var soil_value := -1.0
	var grass_tile_m := -1.0
	var stubble_shader_path := ""
	var candidate_grass_rotation := -1.0
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--candidate-grass-rotation="):
			var value := arg.trim_prefix("--candidate-grass-rotation=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Grass rotation spread must be between zero and one")
				return
			candidate_grass_rotation = float(value)
		elif arg.begins_with("--stubble-shader="):
			stubble_shader_path = arg.trim_prefix("--stubble-shader=")
		elif arg.begins_with("--grass-tile-m="):
			var value := arg.trim_prefix("--grass-tile-m=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.25
				or float(value) > 8
			):
				_fail("Grass tile must be between 0.25 and eight metres")
				return
			grass_tile_m = float(value)
		elif arg.begins_with("--soil-value="):
			var value := arg.trim_prefix("--soil-value=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.1
				or float(value) > 1
			):
				_fail("Soil value must be between 0.1 and one")
				return
			soil_value = float(value)
		elif arg.begins_with("--field-patch-strength="):
			var value := arg.trim_prefix("--field-patch-strength=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0
				or float(value) > 1
			):
				_fail("Field patch strength must be between zero and one")
				return
			field_patch_strength = float(value)
		elif arg.begins_with("--field-soil-strength="):
			var value := arg.trim_prefix("--field-soil-strength=")
			if not value.is_valid_float():
				_fail("Field soil strength must be numeric")
				return
			field_soil_strength = float(value)
			if (
				not is_finite(field_soil_strength)
				or field_soil_strength < 0
				or field_soil_strength > 1
			):
				_fail("Field soil strength must be between zero and one")
				return
		elif arg.begins_with("--ground-tint="):
			var values := arg.trim_prefix("--ground-tint=").split(",")
			if values.size() != 3:
				_fail("Ground tint requires three linear RGB values")
				return
			for i in 3:
				if not values[i].is_valid_float():
					_fail("Ground tint must be numeric")
					return
				ground_tint[i] = float(values[i])
				if not is_finite(ground_tint[i]) or ground_tint[i] <= 0 or ground_tint[i] > 1:
					_fail("Ground tint channels must be greater than zero and at most one")
					return
		elif arg.begins_with("--station="):
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
	var stubble_materials: Array = []
	if not stubble_shader_path.is_empty():
		if not FileAccess.file_exists(stubble_shader_path):
			_fail("Stubble shader does not exist")
			return
		var shader: Shader = load(stubble_shader_path)
		if shader == null:
			_fail("Cannot load stubble shader")
			return
		var seen: Dictionary = {}
		for node in game.find_children("DryGrass_*", "MultiMeshInstance3D", true, false):
			var mesh: Mesh = node.multimesh.mesh
			if seen.has(mesh.get_instance_id()):
				continue
			seen[mesh.get_instance_id()] = true
			for surface in mesh.get_surface_count():
				var replacement := ShaderMaterial.new()
				replacement.shader = shader
				replacement.set_shader_parameter("ground_tint", ThunderhillTrack.DRY_GROUND_TINT)
				replacement.set_shader_parameter(
					"grass_color", load("res://assets/materials/dry_cut_grass_v2.png")
				)
				stubble_materials.append(
					{
						"mesh": mesh,
						"surface": surface,
						"original": mesh.surface_get_material(surface),
						"candidate": replacement
					}
				)
		if stubble_materials.is_empty():
			_fail("No stubble surfaces found")
			return
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
	if grass_tile_m >= 0.0:
		material.set_shader_parameter("grass_tile_m", grass_tile_m)
	else:
		grass_tile_m = RenderingServer.shader_get_parameter_default(
			material.shader.get_rid(), "grass_tile_m"
		)
	if soil_value >= 0.0:
		material.set_shader_parameter("soil_value", soil_value)
	else:
		soil_value = RenderingServer.shader_get_parameter_default(
			material.shader.get_rid(), "soil_value"
		)
	if field_patch_strength >= 0.0:
		material.set_shader_parameter("field_patch_strength", field_patch_strength)
	else:
		field_patch_strength = RenderingServer.shader_get_parameter_default(
			material.shader.get_rid(), "field_patch_strength"
		)
	if ground_tint != Vector3.ZERO:
		material.set_shader_parameter("ground_tint", ground_tint)
	ground_tint = material.get_shader_parameter("ground_tint")
	if field_soil_strength >= 0.0:
		material.set_shader_parameter("field_soil_strength", field_soil_strength)
	else:
		field_soil_strength = RenderingServer.shader_get_parameter_default(
			material.shader.get_rid(), "field_soil_strength"
		)
	material.set_shader_parameter("detail_gain_exponent", detail_gain_exponent)
	var original: Texture2D = material.get_shader_parameter("grass_color")
	var original_rotation: float = RenderingServer.shader_get_parameter_default(
		material.shader.get_rid(), "grass_rotation_spread"
	)
	var samples: Array = []
	for index in frames:
		var pose := position + forward * (0.30 * index)
		pose.y = track.terrain_surface_height(pose) + 1.5
		game.camera.global_position = pose
		game.camera.look_at(pose + target - position)
		for name: String in ["existing", "candidate"]:
			for item in stubble_materials:
				item.mesh.surface_set_material(
					item.surface, item.original if name == "existing" else item.candidate
				)
			material.set_shader_parameter(
				"grass_color", original if name == "existing" else texture
			)
			material.set_shader_parameter(
				"grass_rotation_spread",
				(
					candidate_grass_rotation
					if name == "candidate" and candidate_grass_rotation >= 0.0
					else original_rotation
				)
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
							"existing_grass_rotation": original_rotation,
							"candidate_grass_rotation":
							(
								candidate_grass_rotation
								if candidate_grass_rotation >= 0.0
								else original_rotation
							),
							"detail_gain_exponent": detail_gain_exponent,
							"ground_tint_linear": [ground_tint.x, ground_tint.y, ground_tint.z],
							"field_soil_strength": field_soil_strength,
							"field_patch_strength": field_patch_strength,
							"stubble_shader": stubble_shader_path,
							"stubble_shader_sha256":
							(
								""
								if stubble_shader_path.is_empty()
								else FileAccess.get_sha256(stubble_shader_path)
							),
							"soil_value": soil_value,
							"terrain_shader_sha256":
							FileAccess.get_sha256("res://shaders/terrain.gdshader"),
							"macro_map_sha256":
							FileAccess.get_sha256("res://assets/materials/terrain_macro.png"),
							"detail_map_sha256":
							FileAccess.get_sha256("res://assets/materials/terrain_detail.png"),
							"requested_station_m": station_m,
							"sample_station_m": float(track.samples[nearest].s),
							"step_m": 0.30,
							"samples": samples,
							"view": "road_edge" if road_edge else "field",
							"candidate": candidate,
							"candidate_sha256": FileAccess.get_sha256(candidate),
							"source_size": [source.get_width(), source.get_height()],
							"tile_m": grass_tile_m,
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
