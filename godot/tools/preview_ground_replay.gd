extends SceneTree
## Render a recorded trajectory with an isolated ground treatment.
## No training or human session is accepted by this replay study.


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var mode := ""
	var replay := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--ground-study="):
			mode = arg.get_slice("=", 1)
		elif arg.begins_with("--replay="):
			replay = arg.trim_prefix("--replay=")
		elif arg.begins_with("--agent-port=") or arg == "--qa-controls":
			push_error("Ground replay study cannot run agent or control test sessions")
			quit(2)
			return
	if (
		mode not in ["generated", "regional", "structured", "composition", "scan", "geometry"]
		or not FileAccess.file_exists(replay)
	):
		push_error("Ground study requires a valid mode and an existing replay")
		quit(2)
		return
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	var report: Dictionary
	if mode in ["generated", "regional", "structured", "composition"]:
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
	print("GROUND_REPLAY_STUDY ", JSON.stringify({"mode": mode, "material": report}))
