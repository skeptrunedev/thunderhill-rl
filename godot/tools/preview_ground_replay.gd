extends SceneTree
## Render a recorded trajectory with isolated ground or pavement treatments.
## No training or human session is accepted by this replay study.


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var mode := ""
	var canopy_strength := NAN
	var canopy_additive := false
	var replay := ""
	var directional_wear := NAN
	var exit_scuff := NAN
	var hybrid_aggregate := false
	var curved_uv := true
	var directional_composition := 0.0
	for arg in OS.get_cmdline_user_args():
		if arg == "--canopy-additive":
			canopy_additive = true
		elif arg.begins_with("--canopy-backscatter="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 4.0
			):
				push_error("Canopy strength must be finite and within zero to four")
				quit(2)
				return
			canopy_strength = float(value)
		elif arg == "--photographic-planar-uv":
			curved_uv = false
		elif arg == "--photographic-curved-uv":
			curved_uv = true
		elif arg.begins_with("--photographic-directional-composition="):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				push_error("Directional composition must be finite and within zero to one")
				quit(2)
				return
			directional_composition = float(value)
		elif arg == "--hybrid-aggregate":
			hybrid_aggregate = true
		elif arg.begins_with("--ground-study="):
			mode = arg.get_slice("=", 1)
		elif arg.begins_with("--replay="):
			replay = arg.trim_prefix("--replay=")
		elif (
			arg.begins_with("--asphalt-directional-wear=")
			or arg.begins_with("--asphalt-exit-scuff=")
		):
			var value := arg.get_slice("=", 1)
			if (
				not value.is_valid_float()
				or not is_finite(float(value))
				or float(value) < 0.0
				or float(value) > 1.0
			):
				push_error("Pavement study strength must be finite and within zero to one")
				quit(2)
				return
			if arg.begins_with("--asphalt-exit-scuff="):
				exit_scuff = float(value)
			else:
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
	if hybrid_aggregate and (is_finite(directional_wear) or is_finite(exit_scuff)):
		push_error("Hybrid aggregate replaces pavement overrides; choose one study")
		quit(2)
		return
	if canopy_additive and not is_finite(canopy_strength):
		push_error("Canopy additive mode requires an explicit strength")
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
	game.track.terrain_material.set_shader_parameter("photographic_curved_uv", curved_uv)
	game.track.terrain_material.set_shader_parameter(
		"photographic_directional_composition", directional_composition
	)
	report["photographic_curved_uv"] = curved_uv
	report["photographic_directional_composition"] = directional_composition
	report["terrain_shader_sha256"] = FileAccess.get_sha256(
		game.track.terrain_material.shader.resource_path
	)
	if is_finite(canopy_strength):
		if mode != "production":
			push_error("Canopy study requires production terrain")
			quit(2)
			return
		var canopy: Dictionary = load("res://scripts/canopy_light_study.gd").apply(
			game.track, canopy_strength, canopy_additive
		)
		if canopy.has("error"):
			push_error(canopy.error)
			quit(2)
			return
		report["canopy_study"] = canopy
	var pavement: ShaderMaterial = game.track.get_node("RacingSurface").material_override
	if is_finite(directional_wear):
		pavement.set_shader_parameter("directional_wear_strength", directional_wear)
	if is_finite(exit_scuff):
		pavement.set_shader_parameter("exit_scuff_strength", exit_scuff)
	var effective_wear = pavement.get_shader_parameter("directional_wear_strength")
	if effective_wear == null:
		effective_wear = RenderingServer.shader_get_parameter_default(
			pavement.shader.get_rid(), "directional_wear_strength"
		)
	if hybrid_aggregate:
		pavement.set_shader_parameter("hybrid_aggregate_study", true)
		effective_wear = 0.0
	report["hybrid_aggregate"] = hybrid_aggregate
	report["pavement_shader_sha256"] = FileAccess.get_sha256(pavement.shader.resource_path)
	report["directional_wear_strength"] = effective_wear
	report["exit_scuff_strength_override"] = exit_scuff if is_finite(exit_scuff) else null
	print("GROUND_REPLAY_STUDY ", JSON.stringify({"mode": mode, "material": report}))
