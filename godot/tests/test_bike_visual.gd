extends SceneTree
## Functional dashboard validation and optional close cockpit visual QA.

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var stage := Node3D.new()
	root.add_child(stage)
	var bike = load("res://scripts/bike_visual.gd").new()
	stage.add_child(bike)
	bike.set_rider_visible(false)
	bike.update_instruments(25.0, 7200.0, 3)
	assert(bike._speed_label.text == "090")
	assert(bike._gear_label.text == "3")
	assert(bike._rpm_label.text == "7200 RPM")
	assert(not bike.rider.visible)
	var camera := Camera3D.new()
	stage.add_child(camera)
	camera.position = Vector3(0, 1.37, 0.12)
	camera.look_at(Vector3(0, 0.98, -0.55))
	camera.fov = 66
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
	for i in range(8):
		await process_frame
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--screenshot="):
			await RenderingServer.frame_post_draw
			var path := arg.trim_prefix("--screenshot=")
			assert(root.get_texture().get_image().save_png(path) == OK)
	print("PASS: motorcycle cockpit readouts and rider visibility")
	quit()
