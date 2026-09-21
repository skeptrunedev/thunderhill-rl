extends SceneTree
## Fixed camera lighting study. Does not modify production environment settings.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 720)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var sky_source := ""
	var field_map_path := ""
	var field_map_sha256 := ""
	var field_map_strength := -1.0
	var camera_mode := 2
	var station_m := 400.0
	var fog_density := -1.0
	var paving_joint_strength := -1.0
	var asphalt_roughness := NAN
	var asphalt_detail := {}
	var retained_swath := NAN
	var detail_source := ""
	var lean_deg := 0.0
	var lateral_m := 0.0
	var view_yaw_deg := 0.0
	var view_roll_deg := NAN
	var match_sun_azimuth := false
	var production_only := false
	var sky_yaw := 0.0
	var minimum_elevation := -90.0
	var source_sha256 := ""
	var panorama_is_srgb := false
	var panorama_energy := 1.0
	var seam_overlap := 0.0
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--asphalt-tile-m=") or arg.begins_with("--asphalt-relief-m="):
			var value := arg.get_slice("=", 1)
			var is_tile := arg.begins_with("--asphalt-tile-m=")
			var lower := 0.05 if is_tile else 0.0
			var upper := 2.0 if is_tile else 0.002
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < lower
				or float(value) > upper
			):
				_fail("Asphalt detail must be finite and within its physical study bounds")
				return
			asphalt_detail["authored_tile_m" if is_tile else "authored_relief_m"] = float(value)
		elif arg.begins_with("--terrain-detail="):
			detail_source = arg.trim_prefix("--terrain-detail=")
		elif arg.begins_with("--retained-swath-strength="):
			var value := arg.trim_prefix("--retained-swath-strength=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Retained swath strength must be finite and within zero to one")
				return
			retained_swath = float(value)
		elif arg.begins_with("--asphalt-roughness-offset="):
			var value := arg.trim_prefix("--asphalt-roughness-offset=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < -0.3
				or float(value) > 0.2
			):
				_fail("Asphalt roughness offset must be finite and between -0.3 and 0.2")
				return
			asphalt_roughness = float(value)
		elif arg.begins_with("--paving-joint-strength="):
			var value := arg.trim_prefix("--paving-joint-strength=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Paving joint strength must be finite and within zero to one")
				return
			paving_joint_strength = float(value)
		elif arg.begins_with("--field-map-strength="):
			var value := arg.trim_prefix("--field-map-strength=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Field map strength must be finite and within zero to one")
				return
			field_map_strength = float(value)
		elif arg.begins_with("--field-map="):
			field_map_path = arg.trim_prefix("--field-map=")
		elif arg.begins_with("--fog-density="):
			var value := arg.trim_prefix("--fog-density=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 0.002
			):
				_fail("Fog density must be finite and between zero and 0.002")
				return
			fog_density = float(value)
		elif arg.begins_with("--lateral-m="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or absf(float(value)) > 20.0
			):
				_fail("Lateral position must be finite and within twenty metres")
				return
			lateral_m = float(value)
		elif (
			arg.begins_with("--lean-deg=")
			or arg.begins_with("--view-roll-deg=")
			or arg.begins_with("--view-yaw-deg=")
		):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or absf(float(value)) > 65.0
			):
				_fail("Pose angles must be finite and between minus and plus 65 degrees")
				return
			if arg.begins_with("--lean-deg="):
				lean_deg = float(value)
			elif arg.begins_with("--view-roll-deg="):
				view_roll_deg = float(value)
			else:
				view_yaw_deg = float(value)
		elif arg.begins_with("--station-m="):
			var value := arg.trim_prefix("--station-m=")
			if not value.is_valid_float() or not is_finite(float(value)) or float(value) < 0.0:
				_fail("Station must be finite and nonnegative")
				return
			station_m = float(value)
		elif arg.begins_with("--sky-min-elevation-deg="):
			var value := arg.trim_prefix("--sky-min-elevation-deg=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 30.0
			):
				_fail("Minimum sky elevation must be finite and between zero and 30 degrees")
				return
			minimum_elevation = float(value)
		elif arg == "--match-sun-azimuth":
			match_sun_azimuth = true
		elif arg == "--production-only":
			production_only = true
		elif arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
		elif arg.begins_with("--sky-source="):
			sky_source = arg.trim_prefix("--sky-source=")
		elif arg.begins_with("--camera="):
			var value := arg.trim_prefix("--camera=")
			if not value.is_valid_int() or int(value) < 0 or int(value) > 2:
				_fail("Camera must be 0, 1 or 2")
				return
			camera_mode = int(value)
	if (is_finite(view_roll_deg) or view_yaw_deg != 0.0) and camera_mode == 0:
		_fail("View roll study requires a rider or onboard camera")
		return
	if not output.is_absolute_path() or DisplayServer.get_name() == "headless":
		_fail("Use a real renderer and an absolute --output-dir")
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		_fail("Cannot create output directory")
		return
	if (match_sun_azimuth or minimum_elevation > -90.0) and sky_source.is_empty():
		_fail("Sky orientation and horizon studies require a candidate sky source")
		return
	var variants := [
		{
			"name": "production",
			"mapper": Environment.TONE_MAPPER_FILMIC,
			"exposure": 1.0,
			"sky": 0.7,
			"sun": "fff0d5",
			"fog": 0.00035
		},
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
	if production_only:
		variants = [variants[0]]
	if fog_density >= 0.0:
		for row in variants:
			row.fog = fog_density
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
	if not detail_source.is_empty():
		var detail_error := _apply_terrain_detail(game.track.terrain_material, detail_source)
		if not detail_error.is_empty():
			_fail(detail_error)
			return
	if not field_map_path.is_empty():
		if not field_map_path.is_absolute_path():
			_fail("Field map requires an absolute metadata path")
			return
		var field_data: Variant = JSON.parse_string(FileAccess.get_file_as_string(field_map_path))
		if not field_data is Dictionary:
			_fail("Invalid field map metadata")
			return
		for key in ["local_origin_xz", "local_size_xz", "output_dimensions"]:
			var values: Variant = field_data.get(key)
			if not values is Array or values.size() != 2:
				_fail("Invalid field map coordinates: " + key)
				return
			for value: Variant in values:
				if not (value is float or value is int) or not is_finite(float(value)):
					_fail("Nonfinite field map coordinate")
					return
				if key != "local_origin_xz" and float(value) <= 0.0:
					_fail("Field map dimensions must be positive")
					return
		var field_image_path := field_map_path.get_basename() + ".png"
		field_map_sha256 = FileAccess.get_sha256(field_image_path)
		if field_map_sha256.is_empty() or field_map_sha256 != field_data.get("output_sha256"):
			_fail("Field map image hash mismatch")
			return
		var field_image := Image.load_from_file(field_image_path)
		if (
			field_image == null
			or field_image.get_width() != field_data.output_dimensions[0]
			or field_image.get_height() != field_data.output_dimensions[1]
			or field_image.generate_mipmaps() != OK
		):
			_fail("Invalid field map image or mipmaps")
			return
		var field_material: ShaderMaterial = game.track.terrain_material
		field_material.set_shader_parameter(
			"field_surface_map", ImageTexture.create_from_image(field_image)
		)
		field_material.set_shader_parameter(
			"field_surface_origin",
			Vector2(field_data.local_origin_xz[0], field_data.local_origin_xz[1])
		)
		field_material.set_shader_parameter(
			"field_surface_size", Vector2(field_data.local_size_xz[0], field_data.local_size_xz[1])
		)
		field_material.set_shader_parameter("field_surface_strength", 1.0)
	if field_map_strength >= 0.0:
		game.track.terrain_material.set_shader_parameter(
			"field_surface_strength", field_map_strength
		)
	if paving_joint_strength >= 0.0:
		game.track.get_node("RacingSurface").material_override.set_shader_parameter(
			"paving_joint_strength", paving_joint_strength
		)
	else:
		var pavement_material: ShaderMaterial = (
			game.track.get_node("RacingSurface").material_override
		)
		paving_joint_strength = RenderingServer.shader_get_parameter_default(
			pavement_material.shader.get_rid(), "paving_joint_strength"
		)
	if is_finite(retained_swath):
		game.track.terrain_material.set_shader_parameter("retained_swath_strength", retained_swath)
	if is_finite(asphalt_roughness):
		game.track.get_node("RacingSurface").material_override.set_shader_parameter(
			"roughness_offset", asphalt_roughness
		)
	for parameter in asphalt_detail:
		game.track.get_node("RacingSurface").material_override.set_shader_parameter(
			parameter, asphalt_detail[parameter]
		)
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.hud.visible = false
	game.camera_mode = camera_mode
	if station_m >= game.track.length_m:
		_fail("Station exceeds lap length")
		return
	game.reset_episode(station_m)
	var pose_road: Dictionary = game.track.sample_world(game.sim.position)
	if absf(lateral_m) >= float(pose_road.width) * 0.5:
		_fail("Lateral position must remain inside the track at the selected station")
		return
	game.sim.position += pose_road.left * lateral_m
	game.sim.position.y = game.track.sample_world(game.sim.position).height
	game.sim.lean = deg_to_rad(lean_deg)
	game._update_visual(1.0)
	if view_yaw_deg != 0.0:
		game.camera.rotate(Vector3.UP, deg_to_rad(view_yaw_deg))
	if is_finite(view_roll_deg):
		# Preserve eye position and gaze. Set optical roll relative to world up
		# independently from motorcycle lean to compare reference framing.
		var gaze: Vector3 = -game.camera.global_basis.z
		game.camera.look_at(game.camera.global_position + gaze, Vector3.UP)
		game.camera.rotate_object_local(Vector3.FORWARD, deg_to_rad(view_roll_deg))
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
	if sky_source.is_empty():
		var material: ShaderMaterial = environment.sky.sky_material
		var texture: Texture2D = material.get_shader_parameter("panorama")
		source_sha256 = FileAccess.get_sha256(texture.resource_path)
		panorama_is_srgb = bool(material.get_shader_parameter("panorama_is_srgb"))
		panorama_energy = float(material.get_shader_parameter("panorama_energy"))
		seam_overlap = float(material.get_shader_parameter("panorama_seam_overlap"))
	if not sky_source.is_empty():
		var source: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(sky_source))
		var image := Image.load_from_file(source.local_path)
		if image == null or image.is_empty():
			_fail("Cannot load candidate sky")
			return
		if FileAccess.get_sha256(source.local_path) != source.sha256:
			_fail("Candidate sky differs from manifest")
			return
		if source.get("encoding", "linear") == "srgb":
			image = preload("res://scripts/color_mipmaps.gd").build(image)
		elif image.generate_mipmaps() != OK:
			_fail("Cannot generate candidate sky mipmaps")
			return
		if image == null or image.is_empty():
			_fail("Cannot build candidate sky color mipmaps")
			return
		environment.sky.sky_material.set_shader_parameter(
			"panorama", ImageTexture.create_from_image(image)
		)
		if source.get("encoding", "linear") not in ["linear", "srgb"]:
			_fail("Candidate encoding must be linear or srgb")
			return
		seam_overlap = float(source.get("seam_overlap", 0.0))
		if not is_finite(seam_overlap) or seam_overlap < 0.0 or seam_overlap > 0.1:
			_fail("Seam overlap must be finite and within zero to 0.1")
			return
		environment.sky.sky_material.set_shader_parameter("panorama_seam_overlap", seam_overlap)
		environment.sky.sky_material.set_shader_parameter(
			"use_ground_radiance", bool(source.get("use_ground_radiance", false))
		)
		panorama_is_srgb = source.get("encoding", "linear") == "srgb"
		panorama_energy = float(source.get("radiance_scale", 1.0))
		if not is_finite(panorama_energy) or panorama_energy <= 0.0 or panorama_energy > 16.0:
			_fail("Candidate radiance scale must be positive and at most 16")
			return
		environment.sky.sky_material.set_shader_parameter("panorama_is_srgb", panorama_is_srgb)
		environment.sky.sky_material.set_shader_parameter("panorama_energy", panorama_energy)
		source_sha256 = source.sha256
		var direction: Array = source.toward_sun
		var toward := Vector3(direction[0], direction[1], direction[2])
		if match_sun_azimuth:
			var previous := sun.global_basis.z
			sky_yaw = atan2(previous.x, previous.z) - atan2(toward.x, toward.z)
			toward = Basis(Vector3.UP, sky_yaw) * toward
		environment.sky.sky_material.set_shader_parameter("panorama_yaw", sky_yaw)
		sun.look_at(-toward, Vector3.UP)
	environment.sky.sky_material.set_shader_parameter(
		"minimum_source_y", sin(deg_to_rad(minimum_elevation))
	)
	for row in variants:
		if row.name == "production" and fog_density < 0.0:
			row.fog = environment.fog_density
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
	(
		file
		. store_string(
			(
				(
					JSON
					. stringify(
						{
							"sky_source": sky_source,
							"terrain_detail_source": detail_source,
							"terrain_detail_source_sha256":
							(
								FileAccess.get_sha256(detail_source)
								if not detail_source.is_empty()
								else ""
							),
							"retained_swath_strength":
							game.track.terrain_material.get_shader_parameter(
								"retained_swath_strength"
							),
							"paving_joint_strength": paving_joint_strength,
							"asphalt_roughness_offset":
							(
								game
								. track
								. get_node("RacingSurface")
								. material_override
								. get_shader_parameter("roughness_offset")
							),
							"asphalt_detail_overrides": asphalt_detail,
							"asphalt_shader_sha256":
							FileAccess.get_sha256("res://shaders/asphalt.gdshader"),
							"field_map": field_map_path,
							"field_map_strength":
							game.track.terrain_material.get_shader_parameter(
								"field_surface_strength"
							),
							"field_map_sha256": field_map_sha256,
							"terrain_shader_sha256":
							FileAccess.get_sha256("res://shaders/terrain.gdshader"),
							"runtime_panorama_path":
							(
								environment
								. sky
								. sky_material
								. get_shader_parameter("panorama")
								. resource_path
							),
							"source_sha256": source_sha256,
							"panorama_is_srgb": panorama_is_srgb,
							"panorama_energy": panorama_energy,
							"panorama_seam_overlap": seam_overlap,
							"use_ground_radiance":
							environment.sky.sky_material.get_shader_parameter(
								"use_ground_radiance"
							),
							"ground_radiance":
							str(
								environment.sky.sky_material.get_shader_parameter("ground_radiance")
							),
							"panorama_yaw_radians": sky_yaw,
							"minimum_source_elevation_deg": minimum_elevation,
							"matched_sun_azimuth": match_sun_azimuth,
							"toward_sun":
							[sun.global_basis.z.x, sun.global_basis.z.y, sun.global_basis.z.z],
							"sky_shader_sha256":
							FileAccess.get_sha256("res://shaders/sky.gdshader"),
							"station_m": station_m,
							"motorcycle_lean_deg": lean_deg,
							"lateral_m": lateral_m,
							"view_yaw_deg": view_yaw_deg,
							"view_roll_override_deg":
							view_roll_deg if is_finite(view_roll_deg) else null,
							"pose_status":
							"Artistic static framing study, not reconstructed physical lean or lens calibration",
							"camera": camera_mode,
							"fov": game.camera.fov,
							"camera_transform": str(game.camera.global_transform),
							"variants": variants,
							"limitation": "Artistic appearance study, not calibrated radiometry"
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
		_fail("Study write failed")
		return
	print("LIGHTING_STUDY output=", output)
	quit()


func _fail(message: String) -> void:
	push_error(message)
	quit(2)


func _apply_terrain_detail(material: ShaderMaterial, path: String) -> String:
	if not path.is_absolute_path():
		return "Terrain detail requires an absolute metadata path"
	var data: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not data is Dictionary:
		return "Invalid terrain detail metadata"
	for source in ["track", "surface"]:
		if data.get(source + "_sha256") != FileAccess.get_sha256("res://data/" + source + ".json"):
			return "Terrain detail geometry source mismatch"
	for key in ["local_origin_xz", "local_size_xz", "output_dimensions"]:
		var values: Variant = data.get(key)
		if not values is Array or values.size() != 2:
			return "Invalid terrain detail coordinates"
		for value: Variant in values:
			if not (value is float or value is int) or not is_finite(float(value)):
				return "Nonfinite terrain detail coordinate"
			if key != "local_origin_xz" and float(value) <= 0.0:
				return "Terrain detail dimensions must be positive"
	var origin := Vector2(data.local_origin_xz[0], data.local_origin_xz[1])
	var size := Vector2(data.local_size_xz[0], data.local_size_xz[1])
	if (
		not origin.is_equal_approx(material.get_shader_parameter("macro_origin"))
		or not size.is_equal_approx(material.get_shader_parameter("macro_size"))
	):
		return "Terrain detail mapping differs from production macro"
	var image_path := path.get_basename() + ".png"
	var digest := FileAccess.get_sha256(image_path)
	if digest.is_empty() or digest != data.get("output_sha256"):
		return "Terrain detail image hash mismatch"
	var image := Image.load_from_file(image_path)
	if (
		image == null
		or image.get_width() != data.output_dimensions[0]
		or image.get_height() != data.output_dimensions[1]
	):
		return "Terrain detail image dimensions mismatch"
	if image.generate_mipmaps() != OK:
		return "Cannot generate terrain detail mipmaps"
	material.set_shader_parameter("terrain_detail", ImageTexture.create_from_image(image))
	return ""
