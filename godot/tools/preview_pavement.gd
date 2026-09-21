extends SceneTree
## Compare pavement materials with the same geometry, view and lighting.
const SIZE := Vector2i(1280, 720)
const Validation = preload("res://scripts/image_validation.gd")


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	var candidate := ""
	var surface_variation := -1.0
	var asphalt_warmth := -1.0
	var production_baseline := false
	var binder_mottling := -1.0
	var edge_paint_width := -1.0
	var frames := 1
	var flat_relief := false
	var unlit := false
	var fixed_mip := false
	var linear_mips := false
	var world_uv := false
	var raw_albedo := false
	var variation_only := false
	var offset_m := 0.0
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--asphalt-warmth="):
			var value := arg.trim_prefix("--asphalt-warmth=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				push_error("Asphalt warmth must be finite and within zero to one")
				quit(2)
				return
			asphalt_warmth = float(value)
		if arg == "--production-baseline":
			production_baseline = true
		if arg.begins_with("--edge-paint-width="):
			var value := arg.trim_prefix("--edge-paint-width=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.05
				or float(value) > 0.5
			):
				push_error("Paint width must be finite and within 0.05 to 0.5 metres")
				quit(2)
				return
			edge_paint_width = float(value)
		if arg.begins_with("--binder-mottling="):
			var value := arg.trim_prefix("--binder-mottling=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				push_error("Binder mottling must be finite and within zero to one")
				quit(2)
				return
			binder_mottling = float(value)
		if arg.begins_with("--surface-variation="):
			var value := arg.trim_prefix("--surface-variation=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				push_error("Surface variation must be finite and within zero to one")
				quit(2)
				return
			surface_variation = float(value)
		if arg == "--flat-relief":
			flat_relief = true
		if arg == "--unlit":
			unlit = true
		if arg == "--fixed-mip":
			fixed_mip = true
		if arg == "--linear-mips":
			linear_mips = true
		if arg == "--world-uv":
			world_uv = true
		if arg == "--raw-albedo":
			raw_albedo = true
		if arg == "--variation-only":
			variation_only = true
		if arg.begins_with("--offset-m="):
			var value := arg.trim_prefix("--offset-m=")
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or absf(float(value)) > 100
			):
				push_error("Offset must be finite and within 100 metres")
				quit(2)
				return
			offset_m = float(value)
		if arg.begins_with("--output-dir="):
			output = arg.trim_prefix("--output-dir=")
		elif arg.begins_with("--candidate="):
			candidate = arg.trim_prefix("--candidate=")
		elif arg.begins_with("--frames="):
			var value := arg.trim_prefix("--frames=")
			if not value.is_valid_int() or int(value) < 1 or int(value) > 60:
				push_error("Frames must be an integer between 1 and 60")
				quit(2)
				return
			frames = int(value)
	if (
		production_baseline
		and surface_variation < 0.0
		and binder_mottling < 0.0
		and asphalt_warmth < 0.0
	):
		push_error("Production baseline requires a surface variation, binder, or warmth comparison")
		quit(2)
		return
	if (
		(
			surface_variation >= 0.0
			or binder_mottling >= 0.0
			or edge_paint_width >= 0.0
			or asphalt_warmth >= 0.0
		)
		and (
			not candidate.is_empty()
			or flat_relief
			or unlit
			or fixed_mip
			or linear_mips
			or world_uv
			or raw_albedo
			or variation_only
		)
	):
		push_error(
			"Surface variation comparison requires the unchanged production texture and lighting"
		)
		quit(2)
		return
	if (
		(
			int(binder_mottling >= 0.0)
			+ int(surface_variation >= 0.0)
			+ int(edge_paint_width >= 0.0)
			+ int(asphalt_warmth >= 0.0)
		)
		> 1
	):
		push_error("Choose one independent material or marking comparison")
		quit(2)
		return
	if not output.is_absolute_path() or DisplayServer.get_name() == "headless":
		push_error("Use real renderer and absolute output directory")
		quit(2)
		return
	if DirAccess.make_dir_recursive_absolute(output) != OK:
		quit(2)
		return
	if FileAccess.file_exists(output.path_join("study.json")):
		push_error("Refusing to overwrite pavement study metadata")
		quit(2)
		return
	var labels := ["procedural", "historical"] if candidate.is_empty() else ["existing", "authored"]
	if surface_variation >= 0.0:
		labels = ["baseline", "variation"]
	if binder_mottling >= 0.0:
		labels = ["baseline", "mottling"]
	if edge_paint_width >= 0.0:
		labels = ["baseline", "paint"]
	if asphalt_warmth >= 0.0:
		labels = ["baseline", "warmth"]
	var candidate_texture: ImageTexture
	if not candidate.is_empty():
		if not candidate.is_absolute_path():
			push_error("Candidate requires an absolute path")
			quit(2)
			return
		var source := Image.load_from_file(candidate)
		if source == null or source.is_empty():
			push_error("Cannot load candidate")
			quit(2)
			return
		if linear_mips:
			source = load("res://scripts/color_mipmaps.gd").build(source)
		elif source.generate_mipmaps() != OK:
			push_error("Cannot generate candidate mipmaps")
			quit(2)
			return
		if source == null:
			push_error("Cannot filter candidate")
			quit(2)
			return
		candidate_texture = ImageTexture.create_from_image(source)
	var views: Array = []
	for station in [400, 900, 3000]:
		for index in frames:
			views.append(
				{
					"station": station + index * 0.5 + offset_m,
					"prefix": str(station) if frames == 1 else "%d_%02d" % [station, index]
				}
			)
	for view in views:
		for label in labels:
			if FileAccess.file_exists(output.path_join("%s_%s.png" % [view.prefix, label])):
				push_error("Refusing to overwrite pavement study")
				quit(2)
				return
	root.size = SIZE
	root.content_scale_size = SIZE
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	game.paused = true
	game.process_mode = Node.PROCESS_MODE_DISABLED
	game.hud.visible = false
	game.camera_mode = 1
	var material: ShaderMaterial = game.track.get_node("RacingSurface").material_override
	if unlit or fixed_mip or world_uv or raw_albedo or variation_only:
		var diagnostic_shader := Shader.new()
		var code := material.shader.code
		if variation_only:
			code = (
				code
				. insert(
					code.rfind("}"),
					"if (authored_surface) { ALBEDO = vec3(0.03) * variation * mix(0.84, 1.12, broad); }\n"
				)
			)
		if raw_albedo:
			code = code.insert(
				code.rfind("}"),
				"if (authored_surface) { ALBEDO = authored * authored_color_gain; }\n"
			)
		if world_uv:
			code = code.replace(
				"vec2 authored_uv = UV / authored_tile_m;",
				"vec2 authored_uv = world_position.xz / authored_tile_m;"
			)
		if unlit:
			code = code.replace(
				"render_mode cull_disabled;", "render_mode cull_disabled, unshaded;"
			)
		if fixed_mip:
			code = code.replace(
				"textureGrad(authored_color, authored_uv, authored_dx, authored_dy)",
				"textureLod(authored_color, authored_uv, 2.0)"
			)
		diagnostic_shader.code = code
		material.shader = diagnostic_shader
	if candidate_texture != null:
		material.set_shader_parameter("authored_color", candidate_texture)
	if flat_relief:
		material.set_shader_parameter("authored_relief_m", 0.0)
	material.set_shader_parameter("surface_variation_strength", 0.0)
	material.set_shader_parameter("binder_mottling_strength", 0.0)
	var baseline_surface: float = 0.0
	var baseline_binder: float = 0.0
	if production_baseline:
		baseline_surface = RenderingServer.shader_get_parameter_default(
			material.shader.get_rid(), "surface_variation_strength"
		)
		baseline_binder = RenderingServer.shader_get_parameter_default(
			material.shader.get_rid(), "binder_mottling_strength"
		)
	var authored_texture: Texture2D = material.get_shader_parameter("authored_color")
	var report: Array = []
	for view in views:
		game.reset_episode(float(view.station))
		game._update_visual(1.0)
		for strength in [0.0, 0.65]:
			material.set_shader_parameter("surface_variation_strength", baseline_surface)
			material.set_shader_parameter("binder_mottling_strength", baseline_binder)
			material.set_shader_parameter(
				"pavement_tone_strength", strength if candidate.is_empty() else 0.0
			)
			material.set_shader_parameter(
				"authored_surface", not candidate.is_empty() and strength > 0.0
			)
			if asphalt_warmth >= 0.0:
				material.set_shader_parameter("pavement_tone_strength", 0.0)
				material.set_shader_parameter("authored_surface", true)
				for parameter in ["surface_variation_strength", "binder_mottling_strength"]:
					material.set_shader_parameter(
						parameter,
						RenderingServer.shader_get_parameter_default(
							material.shader.get_rid(), parameter
						)
					)
				material.set_shader_parameter(
					"authored_warmth", 0.0 if strength == 0.0 else asphalt_warmth
				)
			if surface_variation >= 0.0:
				material.set_shader_parameter("pavement_tone_strength", 0.0)
				material.set_shader_parameter("authored_surface", true)
				material.set_shader_parameter(
					"surface_variation_strength",
					baseline_surface if strength == 0.0 else surface_variation
				)
			if binder_mottling >= 0.0:
				material.set_shader_parameter("pavement_tone_strength", 0.0)
				material.set_shader_parameter("authored_surface", true)
				material.set_shader_parameter(
					"surface_variation_strength", baseline_surface if production_baseline else 1.0
				)
				material.set_shader_parameter(
					"binder_mottling_strength",
					baseline_binder if strength == 0.0 else binder_mottling
				)
			if edge_paint_width >= 0.0:
				material.set_shader_parameter("pavement_tone_strength", 0.0)
				material.set_shader_parameter("authored_surface", true)
				for parameter in ["surface_variation_strength", "binder_mottling_strength"]:
					material.set_shader_parameter(
						parameter,
						RenderingServer.shader_get_parameter_default(
							material.shader.get_rid(), parameter
						)
					)
				var width := 0.12 if strength == 0.0 else edge_paint_width
				var paint_surface: SurfaceTool = game.track.edge_paint_surface(width)
				paint_surface.generate_normals()
				paint_surface.generate_tangents()
				var paint_mesh: MeshInstance3D = game.track.get_node("EdgePaint")
				paint_mesh.mesh = paint_surface.commit()
				paint_mesh.material_override.set_shader_parameter("edge_paint_width_m", width)
			for frame in 4:
				await RenderingServer.frame_post_draw
			var capture := root.get_texture().get_image()
			var label: String = labels[0 if strength == 0.0 else 1]
			var name := "%s_%s.png" % [view.prefix, label]
			if (
				Validation.classify(capture, SIZE) != "nonblack"
				or capture.save_png(output.path_join(name)) != OK
			):
				quit(2)
				return
			report.append(
				{
					"image": name,
					"edge_paint_width_m":
					game.track.get_node("EdgePaint").material_override.get_shader_parameter(
						"edge_paint_width_m"
					),
					"requested_station_m": view.station,
					"historical_strength": material.get_shader_parameter("pavement_tone_strength"),
					"authored_warmth": material.get_shader_parameter("authored_warmth"),
					"surface_variation_strength":
					material.get_shader_parameter("surface_variation_strength"),
					"binder_mottling_strength":
					material.get_shader_parameter("binder_mottling_strength"),
					"authored_surface": material.get_shader_parameter("authored_surface"),
					"camera_transform": str(game.camera.global_transform)
				}
			)
	var file := FileAccess.open(output.path_join("study.json"), FileAccess.WRITE)
	if file == null:
		quit(2)
		return
	(
		file
		. store_string(
			(
				JSON
				. stringify(
					{
						"captures": report,
						"frames_per_station": frames,
						"production_baseline": production_baseline,
						"runtime_authored_texture": authored_texture.resource_path,
						"runtime_authored_texture_sha256":
						(
							""
							if authored_texture.resource_path.is_empty()
							else FileAccess.get_sha256(authored_texture.resource_path)
						),
						"flat_relief": flat_relief,
						"unlit": unlit,
						"fixed_mip": fixed_mip,
						"linear_mips": linear_mips,
						"world_uv": world_uv,
						"raw_albedo": raw_albedo,
						"variation_only": variation_only,
						"step_m": 0.5,
						"candidate": candidate,
						"candidate_sha256":
						"" if candidate.is_empty() else FileAccess.get_sha256(candidate),
						"track_script_sha256": FileAccess.get_sha256("res://scripts/track.gd"),
						"paint_shader_sha256":
						FileAccess.get_sha256("res://shaders/painted_concrete.gdshader"),
						"shader_sha256": FileAccess.get_sha256("res://shaders/asphalt.gdshader"),
						"tone_sha256":
						FileAccess.get_sha256("res://assets/materials/pavement_tone.png"),
						"limitation":
						"Authored and historical appearance estimates, not recovered reflectance or surveyed friction. Sampled stationary views do not establish temporal stability."
					},
					"  "
				)
			)
		)
	)
	file.flush()
	var error := file.get_error()
	file.close()
	print("PAVEMENT_STUDY error=", error)
	quit(0 if error == OK else 2)
