extends SceneTree
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func _initialize() -> void:
	run.call_deferred()


func run() -> void:
	var game = load("res://main.tscn").instantiate()
	root.add_child(game)
	await process_frame
	game.paused = true
	game.camera_mode = 1
	# Discontinuous positions expose any world-space chase lag in the rider view.
	# Sloped track locations, mirrored lean and frame rates exercise pose attachment.
	for index in [0, 300, 750, 1200]:
		game.sim.position = game.track.points[index]
		var road: Dictionary = game.track.sample_world(game.sim.position)
		game.sim.heading = atan2(road.tangent.x, -road.tangent.z)
		for lean in [-0.5, 0.0, 0.5]:
			game.sim.lean = lean
			for dt in [1.0 / 30.0, 1.0 / 120.0]:
				game._update_visual(dt)
				var local_eye: Vector3 = game.bike.to_local(game.camera.global_position)
				check(
					local_eye.distance_to(game.bike.RIDER_EYE_LOCAL) < 0.0001,
					"Rider camera detached from helmet"
				)
				check(game.camera.global_basis.is_finite(), "Rider camera basis is not finite")
				check(not game.camera.get_cull_mask_value(20), "Helmet obscures rider view")
	check(game.camera.fov == 90.0, "Rider field of view was not applied")
	game.camera_mode = 0
	game._update_visual(1.0 / 60.0)
	check(game.camera.fov == 64.0, "Chase field of view was not restored")
	check(game.camera.get_cull_mask_value(20), "Chase view lost rider mesh")
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--screenshot="):
			game.camera_mode = 1
			game.sim.lean = 0.4
			game._update_visual(1.0 / 60.0)
			game.hud.visible = false
			await RenderingServer.frame_post_draw
			check(
				root.get_texture().get_image().save_png(arg.trim_prefix("--screenshot=")) == OK,
				"Camera render capture failed"
			)
	print("RIDER_CAMERA_CHECK failures=", failures)
	game.queue_free()
	await process_frame
	quit(failures)
