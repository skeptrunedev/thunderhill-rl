extends SceneTree
## Fixed camera lighting study. Does not modify production environment settings.
## --asphalt-study=matte-aggregate selects an independent matte procedural road.
## --photographic-height-blend=0..1 selects the terrain texture overlap study.
const Validation = preload("res://scripts/image_validation.gd")
const SIZE := Vector2i(1280, 720)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var orchard_study := false
	var orchard_metadata := {}
	var asphalt_study := "production"
	var asphalt_study_metadata := {}
	var ground_study := "production"
	var ground_study_metadata := {}
	var photographic_contrast := 0.0
	var photographic_height_blend := 0.0
	var photographic_directional_composition := 0.0
	var photographic_curved_uv := true
	var sky_source := ""
	var solar_haze := 0.0
	var solar_haze_broad := false
	var sky_patch_path := ""
	var sky_patch_off := false
	var sky_secondary_off := false
	var sky_patch_metadata := {}
	var field_map_path := ""
	var field_map_sha256 := ""
	var field_map_strength := -1.0
	var camera_mode := 2
	var camera_overrides := {}
	var station_m := 400.0
	var fog_density := -1.0
	var paving_joint_strength := -1.0
	var asphalt_roughness := NAN
	var asphalt_detail := {}
	var retained_swath := NAN
	var mapped_aerial_contrast := NAN
	var mowing_band := NAN
	var mowing_detail := NAN
	var pale_straw_scale := NAN
	var detail_source := ""
	var lean_deg := 0.0
	var lateral_m := 0.0
	var view_yaw_deg := 0.0
	var view_roll_deg := NAN
	var match_sun_azimuth := false
	var production_only := false
	var sky_yaw := 0.0
	var sun_azimuth := NAN
	var sun_elevation := NAN
	var sun_energy := NAN
	var sun_color := ""
	var cloud_gain := NAN
	var authored_yaw := NAN
	var minimum_elevation := -90.0
	var source_sha256 := ""
	var panorama_is_srgb := false
	var panorama_energy := 1.0
	var seam_overlap := 0.0
	for arg in OS.get_cmdline_user_args():
		if arg == "--photographic-planar-uv":
			photographic_curved_uv = false
		elif arg == "--photographic-curved-uv":
			photographic_curved_uv = true
		elif arg.begins_with("--photographic-directional-composition="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Photographic directional composition must be finite and within zero to one")
				return
			photographic_directional_composition = float(value)
		elif arg.begins_with("--photographic-height-blend="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Photographic height blend must be finite and within zero to one")
				return
			photographic_height_blend = float(value)
		elif arg.begins_with("--photographic-contrast="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Photographic contrast preservation must be finite and within zero to one")
				return
			photographic_contrast = float(value)
		elif arg == "--solar-haze-broad":
			solar_haze_broad = true
		elif arg.begins_with("--solar-haze="):
			var value := arg.trim_prefix("--solar-haze=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 8.0
			):
				_fail("Solar haze must be finite and within zero to eight")
				return
			solar_haze = float(value)
		elif arg.begins_with("--sun-color="):
			var value := arg.trim_prefix("--sun-color=").to_lower()
			var valid := value.length() == 6
			for character in value:
				valid = valid and character in "0123456789abcdef"
			if not valid:
				_fail("Sun color must contain exactly six hexadecimal RGB digits")
				return
			sun_color = value
		elif arg.begins_with("--sun-energy="):
			var value := arg.trim_prefix("--sun-energy=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) <= 0.0
				or float(value) > 8.0
			):
				_fail("Sun energy must be finite, greater than zero and at most eight")
				return
			sun_energy = float(value)
		elif arg.begins_with("--sun-elevation-deg="):
			var value := arg.trim_prefix("--sun-elevation-deg=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) <= 0.0
				or float(value) >= 90.0
			):
				_fail("Sun elevation must be finite and strictly between zero and 90 degrees")
				return
			sun_elevation = float(value)
		elif arg.begins_with("--sun-azimuth-deg="):
			var value := arg.trim_prefix("--sun-azimuth-deg=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) >= 360.0
			):
				_fail("Sun azimuth must be finite and in [0, 360)")
				return
			sun_azimuth = float(value)
		elif arg == "--orchard-study":
			orchard_study = true
		elif arg.begins_with("--asphalt-study="):
			asphalt_study = arg.get_slice("=", 1)
			if asphalt_study not in ["production", "scan", "matte-aggregate", "hybrid-aggregate"]:
				_fail("Unknown asphalt study mode")
				return
		elif arg.begins_with("--ground-study="):
			ground_study = arg.get_slice("=", 1)
			if (
				ground_study
				not in [
					"production",
					"legacy",
					"generated",
					"regional",
					"structured",
					"composition",
					"scan",
					"geometry"
				]
			):
				_fail("Unknown ground study mode")
				return
		elif arg.begins_with("--authored-sky-yaw-deg="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or absf(float(value)) > 360.0
			):
				_fail("Authored sky yaw must be finite and within 360 degrees")
				return
			authored_yaw = deg_to_rad(float(value))
		elif arg.begins_with("--cloud-radiance-gain="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 1.0
				or float(value) > 4.0
			):
				_fail("Cloud radiance gain must be finite and between one and four")
				return
			cloud_gain = float(value)
		elif (
			arg.begins_with("--asphalt-broad-tone=")
			or arg.begins_with("--asphalt-binder-mottling=")
			or arg.begins_with("--asphalt-surface-variation=")
			or arg.begins_with("--asphalt-directional-wear=")
		):
			var name := arg.get_slice("=", 0).trim_prefix("--asphalt-")
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Asphalt tone strength must be finite and within zero to one")
				return
			var parameter: String = {
				"broad-tone": "broad_tone_strength",
				"binder-mottling": "binder_mottling_strength",
				"surface-variation": "surface_variation_strength",
				"directional-wear": "directional_wear_strength"
			}[name]
			asphalt_detail[parameter] = float(value)
		elif arg.begins_with("--asphalt-tile-m=") or arg.begins_with("--asphalt-relief-m="):
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
		elif (
			arg.begins_with("--camera-height-m=")
			or arg.begins_with("--camera-pitch-deg=")
			or arg.begins_with("--camera-fov-deg=")
		):
			var name := arg.get_slice("=", 0).trim_prefix("--camera-")
			var value := arg.get_slice("=", 1)
			var bounds: Vector2 = {
				"height-m": Vector2(0.9, 1.8),
				"pitch-deg": Vector2(0, 40),
				"fov-deg": Vector2(50, 110)
			}[name]
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < bounds.x
				or float(value) > bounds.y
			):
				_fail("Camera override outside finite study bounds: " + name)
				return
			camera_overrides[name] = float(value)
		elif arg.begins_with("--pale-straw-gain-scale="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.5
				or float(value) > 2.0
			):
				_fail("Pale straw gain scale must be finite and within 0.5 to two")
				return
			pale_straw_scale = float(value)
		elif arg.begins_with("--mapped-aerial-contrast="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Mapped aerial contrast must be finite and within zero to one")
				return
			mapped_aerial_contrast = float(value)
		elif arg.begins_with("--mowing-detail-strength="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Mowing detail strength must be finite and within zero to one")
				return
			mowing_detail = float(value)
		elif arg.begins_with("--mowing-band-strength="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				_fail("Mowing band strength must be finite and within zero to one")
				return
			mowing_band = float(value)
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
		elif arg == "--sky-secondary-off":
			sky_secondary_off = true
		elif arg == "--sky-patch-off":
			sky_patch_off = true
		elif arg.begins_with("--sky-patch="):
			sky_patch_path = arg.trim_prefix("--sky-patch=")
		elif arg.begins_with("--sky-source="):
			sky_source = arg.trim_prefix("--sky-source=")
		elif arg.begins_with("--camera="):
			var value := arg.trim_prefix("--camera=")
			if not value.is_valid_int() or int(value) < 0 or int(value) > 2:
				_fail("Camera must be 0, 1 or 2")
				return
			camera_mode = int(value)
	if (
		asphalt_study != "production"
		and (
			not asphalt_detail.is_empty()
			or is_finite(asphalt_roughness)
			or paving_joint_strength >= 0.0
		)
	):
		_fail("Independent asphalt study cannot combine production material overrides")
		return
	if sky_patch_off and not sky_patch_path.is_empty():
		_fail("Choose either a sky patch override or sky patch off")
		return
	if is_finite(authored_yaw) and not sky_source.is_empty():
		_fail("Authored yaw is only for the production panorama without a captured solar disk")
		return
	if not camera_overrides.is_empty() and camera_mode == 0:
		_fail("Camera overrides require a rider or onboard camera")
		return
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
	if is_finite(pale_straw_scale):
		var material: ShaderMaterial = game.track.terrain_material
		var gain: Vector3 = RenderingServer.shader_get_parameter_default(
			material.shader.get_rid(), "pale_straw_gain"
		)
		material.set_shader_parameter("pale_straw_gain", gain * pale_straw_scale)
	if is_finite(mapped_aerial_contrast):
		game.track.terrain_material.set_shader_parameter(
			"mapped_aerial_contrast", mapped_aerial_contrast
		)
	if is_finite(mowing_detail):
		game.track.terrain_material.set_shader_parameter("mowing_detail_strength", mowing_detail)
	if is_finite(mowing_band):
		game.track.terrain_material.set_shader_parameter("mowing_band_strength", mowing_band)
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
	var production_terrain_material: ShaderMaterial = game.track.terrain_material
	production_terrain_material.set_shader_parameter(
		"photographic_curved_uv", photographic_curved_uv
	)
	production_terrain_material.set_shader_parameter(
		"photographic_directional_composition", photographic_directional_composition
	)
	production_terrain_material.set_shader_parameter(
		"photographic_contrast_preservation", photographic_contrast
	)
	production_terrain_material.set_shader_parameter(
		"photographic_height_blend", photographic_height_blend
	)
	if ground_study == "legacy":
		game.track.terrain_material.set_shader_parameter("photographic_field_enabled", false)
		ground_study_metadata = {"mode": "legacy", "photographic_field_enabled": false}
	elif ground_study in ["generated", "regional", "structured", "composition"]:
		ground_study_metadata = load("res://scripts/ground_generated_study.gd").apply(
			game.track,
			ground_study != "generated",
			ground_study == "structured",
			ground_study == "composition"
		)
	elif ground_study == "scan":
		ground_study_metadata = load("res://scripts/ground_scan_study.gd").apply(game.track)
	elif ground_study == "geometry":
		game.track.terrain_material.set_shader_parameter("photographic_field_enabled", false)
		ground_study_metadata = load("res://scripts/straw_geometry_study.gd").setup(game)
	if ground_study_metadata.has("error"):
		_fail(ground_study_metadata.error)
		return
	if asphalt_study == "scan":
		asphalt_study_metadata = load("res://scripts/asphalt_scan_study.gd").apply(game.track)
		if asphalt_study_metadata.has("error"):
			_fail(asphalt_study_metadata.error)
			return
	if asphalt_study in ["matte-aggregate", "hybrid-aggregate"]:
		var material: ShaderMaterial = game.track.get_node("RacingSurface").material_override
		material.set_shader_parameter("matte_aggregate_study", asphalt_study == "matte-aggregate")
		material.set_shader_parameter("hybrid_aggregate_study", asphalt_study == "hybrid-aggregate")
		var normal_texture: Texture2D = material.get_shader_parameter("normal_map")
		asphalt_study_metadata = {
			"mode": asphalt_study,
			"authored_albedo_weight": 0.75 if asphalt_study == "hybrid-aggregate" else 0.0,
			"aggregate_spacing_m": [0.006, 0.018],
			"binder_spacing_m": 0.25,
			"linear_albedo": [0.032, 0.031, 0.029],
			"roughness": 0.68 if asphalt_study == "hybrid-aggregate" else 0.88,
			"specular": 0.42 if asphalt_study == "hybrid-aggregate" else 0.35,
			"normal_scale_m": 0.5,
			"normal_strength": 0.08 if asphalt_study == "hybrid-aggregate" else 0.16,
			"normal_source_path": normal_texture.resource_path,
			"normal_source_sha256": FileAccess.get_sha256(normal_texture.resource_path),
			"filter": "Derivative footprint fades unresolved stone cells to their mean",
			"limitation":
			"Original procedural color with generic scan relief. All material values and physical scales are artistic estimates. No measured Thunderhill reflectance or friction."
		}
	if orchard_study:
		orchard_metadata = load("res://scripts/orchard_study.gd").apply(game.track)
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
	if camera_overrides.has("height-m"):
		var anchor: Vector3 = (
			game.bike.ONBOARD_CAMERA_LOCAL if camera_mode == 2 else game.bike.RIDER_EYE_LOCAL
		)
		anchor.y = camera_overrides["height-m"]
		game.camera.global_position = game.bike.to_global(anchor)
	if camera_overrides.has("pitch-deg"):
		var tangent: Vector3 = game.sim.surface_forward(
			game.track.sample_world(game.sim.position).normal
		)
		var upright := Vector3.UP.slide(tangent).normalized()
		var pitch := deg_to_rad(float(camera_overrides["pitch-deg"]))
		var gaze := tangent * cos(pitch) - upright * sin(pitch)
		game.camera.look_at(game.camera.global_position + gaze * 30.0, upright)
		var roll_scale: float = (
			game.bike.ONBOARD_ROLL_SCALE if camera_mode == 2 else game.bike.RIDER_ROLL_SCALE
		)
		game.camera.rotate_object_local(Vector3.FORWARD, game.sim.lean * roll_scale)
	if camera_overrides.has("fov-deg"):
		game.camera.fov = camera_overrides["fov-deg"]
	if view_yaw_deg != 0.0:
		game.camera.rotate(Vector3.UP, deg_to_rad(view_yaw_deg))
	if is_finite(view_roll_deg):
		# Preserve eye position and gaze. Set optical roll relative to world up
		# independently from motorcycle lean to compare reference framing.
		var gaze: Vector3 = -game.camera.global_basis.z
		game.camera.look_at(game.camera.global_position + gaze, Vector3.UP)
		game.camera.rotate_object_local(Vector3.FORWARD, deg_to_rad(view_roll_deg))
	if orchard_study:
		orchard_metadata["visibility_audit"] = load("res://scripts/orchard_study.gd").audit(
			game.track, game.camera
		)
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
	if is_finite(authored_yaw):
		sky_yaw = authored_yaw
		environment.sky.sky_material.set_shader_parameter("panorama_yaw", sky_yaw)
	if is_finite(cloud_gain):
		environment.sky.sky_material.set_shader_parameter("cloud_radiance_gain", cloud_gain)
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
	if sky_patch_off:
		environment.sky.sky_material.set_shader_parameter("sky_patch_enabled", false)
	if sky_patch_off or sky_secondary_off:
		environment.sky.sky_material.set_shader_parameter("sky_patch_secondary_enabled", false)
	if not sky_patch_path.is_empty():
		sky_patch_metadata = _apply_sky_patch(environment.sky.sky_material, sky_patch_path)
		if sky_patch_metadata.has("error"):
			_fail(sky_patch_metadata.error)
			return
	environment.sky.sky_material.set_shader_parameter(
		"minimum_source_y", sin(deg_to_rad(minimum_elevation))
	)
	if is_finite(sun_energy):
		sun.light_energy = sun_energy
	if is_finite(sun_azimuth):
		# Geographic azimuth, north is local negative Z. Preserve light elevation.
		var toward := sun.global_basis.z.normalized()
		var horizontal := Vector2(toward.x, toward.z).length()
		var azimuth := deg_to_rad(sun_azimuth)
		sun.look_at(
			-Vector3(sin(azimuth) * horizontal, toward.y, -cos(azimuth) * horizontal), Vector3.UP
		)
	if is_finite(sun_elevation):
		var toward := sun.global_basis.z.normalized()
		var azimuth := atan2(toward.x, -toward.z)
		var elevation := deg_to_rad(sun_elevation)
		sun.look_at(
			-Vector3(sin(azimuth) * cos(elevation), sin(elevation), -cos(azimuth) * cos(elevation)),
			Vector3.UP
		)
	environment.sky.sky_material.set_shader_parameter("solar_haze_strength", solar_haze)
	environment.sky.sky_material.set_shader_parameter("solar_haze_broad", solar_haze_broad)
	environment.sky.sky_material.set_shader_parameter(
		"solar_haze_direction", sun.global_basis.z.normalized()
	)
	var sun_point: Vector3 = game.camera.global_position + sun.global_basis.z.normalized() * 100.0
	var projected_sun: Vector2 = game.camera.unproject_position(sun_point)
	var sun_in_front: bool = -game.camera.to_local(sun_point).z > 0.0
	var pavement: ShaderMaterial = game.track.get_node("RacingSurface").material_override
	var effective_wear = pavement.get_shader_parameter("directional_wear_strength")
	if effective_wear == null and asphalt_study == "production":
		effective_wear = RenderingServer.shader_get_parameter_default(
			pavement.shader.get_rid(), "directional_wear_strength"
		)
	if asphalt_study in ["matte-aggregate", "hybrid-aggregate"]:
		effective_wear = 0.0
	if asphalt_study == "matte-aggregate":
		paving_joint_strength = 0.0
	for row in variants:
		if row.name == "production" and fog_density < 0.0:
			row.fog = environment.fog_density
		environment.tonemap_mode = row.mapper
		environment.tonemap_exposure = row.exposure
		environment.fog_density = row.fog
		sun.light_color = Color(row.sun if sun_color.is_empty() else sun_color)
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
							"orchard_study": orchard_metadata,
							"solar_haze_strength": solar_haze,
							"direct_sun_energy": sun.light_energy,
							"direct_sun_color": sun.light_color.to_html(false),
							"direct_sun_color_override": sun_color,
							"direct_sun_energy_override":
							sun_energy if is_finite(sun_energy) else null,
							"solar_haze_broad": solar_haze_broad,
							"direct_sun_elevation_override_deg":
							sun_elevation if is_finite(sun_elevation) else null,
							"sun_projection_pixels": [projected_sun.x, projected_sun.y],
							"sun_projection_in_front": sun_in_front,
							"asphalt_study": asphalt_study,
							"asphalt_study_metadata": asphalt_study_metadata,
							"ground_study": ground_study,
							"photographic_contrast_preservation": photographic_contrast,
							"photographic_height_blend": photographic_height_blend,
							"photographic_directional_composition":
							photographic_directional_composition,
							"ground_study_metadata": ground_study_metadata,
							"photographic_sparse_enabled":
							_material_parameter(
								game.track.terrain_material, "photographic_sparse_enabled"
							),
							"photographic_disturbance_strength":
							_material_parameter(
								game.track.terrain_material, "photographic_disturbance_strength"
							),
							"sky_source": sky_source,
							"sky_patch": sky_patch_metadata,
							"runtime_secondary_sky_patch_enabled":
							environment.sky.sky_material.get_shader_parameter(
								"sky_patch_secondary_enabled"
							),
							"runtime_sky_patch_enabled":
							environment.sky.sky_material.get_shader_parameter("sky_patch_enabled"),
							"production_sky_metadata_sha256":
							FileAccess.get_sha256("res://data/sky.json"),
							"sky_patch_setup_sha256":
							FileAccess.get_sha256("res://scripts/sky_patch.gd"),
							"terrain_detail_source": detail_source,
							"terrain_detail_source_sha256":
							(
								FileAccess.get_sha256(detail_source)
								if not detail_source.is_empty()
								else ""
							),
							"pale_straw_gain":
							str(
								_material_parameter(production_terrain_material, "pale_straw_gain")
							),
							"mapped_aerial_contrast":
							_material_parameter(
								production_terrain_material, "mapped_aerial_contrast"
							),
							"mowing_detail_strength":
							_material_parameter(
								production_terrain_material, "mowing_detail_strength"
							),
							"mowing_band_strength":
							production_terrain_material.get_shader_parameter(
								"mowing_band_strength"
							),
							"retained_swath_strength":
							production_terrain_material.get_shader_parameter(
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
							"directional_wear_strength": effective_wear,
							"asphalt_shader_sha256":
							FileAccess.get_sha256("res://shaders/asphalt.gdshader"),
							"field_map": field_map_path,
							"field_map_strength":
							production_terrain_material.get_shader_parameter(
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
							"cloud_radiance_gain":
							environment.sky.sky_material.get_shader_parameter(
								"cloud_radiance_gain"
							),
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
							"direct_sun_azimuth_override_deg":
							sun_azimuth if is_finite(sun_azimuth) else null,
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
							"camera_overrides": camera_overrides,
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


func _apply_sky_patch(material: ShaderMaterial, path: String) -> Dictionary:
	if not path.is_absolute_path() or not FileAccess.file_exists(path):
		return {"error": "Sky patch requires an existing absolute manifest path"}
	var data: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not data is Dictionary or data.get("encoding") != "srgb":
		return {"error": "Sky patch manifest must be a dictionary with srgb encoding"}
	var image_path: Variant = data.get("local_path")
	if not image_path is String or not image_path.is_absolute_path():
		return {"error": "Sky patch image requires an absolute local_path"}
	if not FileAccess.file_exists(image_path):
		return {"error": "Sky patch image does not exist"}
	var digest := FileAccess.get_sha256(image_path)
	if digest.is_empty() or digest != data.get("sha256"):
		return {"error": "Sky patch image hash mismatch"}
	var bounds := {
		"horizontal_fov_degrees": [1.0, 170.0],
		"center_azimuth_degrees": [-360.0, 360.0],
		"center_elevation_degrees": [-90.0, 90.0],
		"feather_fraction": [0.001, 0.5]
	}
	for key in bounds:
		var value: Variant = data.get(key)
		if not (value is float or value is int):
			return {"error": "Sky patch requires a numeric " + key}
		if not is_finite(float(value)) or value < bounds[key][0] or value > bounds[key][1]:
			return {"error": "Sky patch projection outside finite bounds: " + key}
	var image := Image.load_from_file(image_path)
	if image == null or image.is_empty():
		return {"error": "Cannot load sky patch image"}
	var dimensions := [image.get_width(), image.get_height()]
	if dimensions[0] < 2 or dimensions[1] < 2 or dimensions[0] > 8192 or dimensions[1] > 8192:
		return {"error": "Sky patch dimensions must be between 2 and 8192 pixels"}
	if data.has("output_dimensions") and data.output_dimensions != dimensions:
		return {"error": "Sky patch dimensions differ from manifest"}
	image = preload("res://scripts/color_mipmaps.gd").build(image)
	if image == null or image.is_empty():
		return {"error": "Cannot build sky patch color mipmaps (requires RGB8 or RGBA8)"}
	var projection := preload("res://scripts/sky_patch.gd").configure(
		material,
		ImageTexture.create_from_image(image),
		float(data.horizontal_fov_degrees),
		float(data.center_azimuth_degrees),
		float(data.center_elevation_degrees),
		float(data.feather_fraction)
	)
	projection.merge(
		{
			"manifest_path": path,
			"manifest_sha256": FileAccess.get_sha256(path),
			"local_path": image_path,
			"sha256": digest,
			"encoding": "srgb",
			"output_dimensions": dimensions
		}
	)
	return projection


func _material_parameter(material: ShaderMaterial, parameter: String) -> Variant:
	var value: Variant = material.get_shader_parameter(parameter)
	if value != null:
		return value
	return RenderingServer.shader_get_parameter_default(material.shader.get_rid(), parameter)
