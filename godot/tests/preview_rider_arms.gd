extends SceneTree


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("Arm preview requires a renderer; use test_rider_pose.gd for headless checks")
		quit(2)
		return
	var steering := 0.0
	var path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--steering="):
			steering = float(arg.trim_prefix("--steering="))
		if arg.begins_with("--screenshot="):
			path = arg.trim_prefix("--screenshot=")
	var stage := Node3D.new()
	root.add_child(stage)
	var bike := preload("res://scripts/bike_visual.gd").new()
	stage.add_child(bike)
	bike.set_rider_visible(false)
	bike.update_pose(0, steering, 0)
	bike._arms.visible = true
	var camera := Camera3D.new()
	stage.add_child(camera)
	camera.position = bike.ONBOARD_CAMERA_LOCAL
	camera.look_at(camera.position + Vector3(0, -sin(0.38), -cos(0.38)))
	camera.fov = 74.0
	camera.near = 0.02
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color("5f7581")
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color("c5d3dd")
	environment.environment.ambient_light_energy = 0.6
	stage.add_child(environment)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-50, -30, 0)
	light.light_energy = 1.6
	stage.add_child(light)
	for i in range(12):
		await process_frame
	await RenderingServer.frame_post_draw
	if not path.is_empty():
		assert(root.get_texture().get_image().save_png(path) == OK)
	print("RIDER_ARMS_PREVIEW steering=", steering)
	quit()
