extends SceneTree
## Render a recorded trajectory with isolated ground or pavement treatments.
## No training or human session is accepted by this replay study.


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var mode := ""
	var replay := ""
	var directional_wear := NAN
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--ground-study="):
			mode = arg.get_slice("=", 1)
		elif arg.begins_with("--replay="):
			replay = arg.trim_prefix("--replay=")
		elif arg.begins_with("--asphalt-directional-wear="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				push_error("Directional wear must be finite and within zero to one")
				quit(2)
				return
			directional_wear = float(value)
		elif arg.begins_with("--agent-port=") or arg == "--qa-controls":
			push_error("Ground replay study cannot run agent or control test sessions")
			quit(2)
			return
	if (
		(
			mode
			not in [
				"production",
				"generated",
				"regional",
				"structured",
				"composition",
				"scan",
				"geometry"
			]
		)
		or not FileAccess.file_exists(replay)
	):
		push_error("Ground study requires a valid mode and an existing replay")
		quit(2)
		return
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	var report: Dictionary
	if mode == "production":
		report = {"mode": "production"}
	elif mode in ["generated", "regional", "structured", "composition"]:
		report = load("res://scripts/ground_generated_study.gd").apply(
			game.track, mode != "generated", mode == "structured", mode == "composition"
		)
	elif mode == "scan":
		report = load("res://scripts/ground_scan_study.gd").apply(game.track)
	else:
		game.track.terrain_material.set_shader_parameter("photographic_field_enabled", false)
		report = load("res://scripts/straw_geometry_study.gd").setup(game)
	if report.has("error"):
		push_error(report.error)
		quit(2)
		return
	var pavement: ShaderMaterial = game.track.get_node("RacingSurface").material_override
	if is_finite(directional_wear):
		pavement.set_shader_parameter("directional_wear_strength", directional_wear)
	var effective_wear = pavement.get_shader_parameter("directional_wear_strength")
	if effective_wear == null:
		effective_wear = RenderingServer.shader_get_parameter_default(
			pavement.shader.get_rid(), "directional_wear_strength"
		)
	report["directional_wear_strength"] = effective_wear
	print("GROUND_REPLAY_STUDY ", JSON.stringify({"mode": mode, "material": report}))
