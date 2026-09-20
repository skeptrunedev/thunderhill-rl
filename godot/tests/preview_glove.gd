extends SceneTree


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var stage := Node3D.new()
	root.add_child(stage)
	var material := StandardMaterial3D.new()
	material.vertex_color_use_as_albedo = true
	material.vertex_color_is_srgb = true
	material.roughness = 0.65
	for side in [-1.0, 1.0]:
		var glove := MeshInstance3D.new()
		glove.mesh = preload("res://scripts/rider_glove.gd").build(side)
		var arrays := glove.mesh.surface_get_arrays(0)
		var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
		assert(
			preload("res://scripts/bike_collision_envelope.gd")._has_volume(vertices),
			"Glove has no hull volume"
		)
		for index in vertices.size():
			assert(vertices[index].is_finite(), "Nonfinite glove vertex")
			assert(
				normals[index].is_finite() and absf(normals[index].length() - 1.0) < 0.001,
				"Invalid glove normal"
			)
		glove.material_override = material
		glove.position.x = side * 0.13
		stage.add_child(glove)
		var grip := MeshInstance3D.new()
		var cylinder := CylinderMesh.new()
		cylinder.top_radius = 0.021
		cylinder.bottom_radius = 0.021
		cylinder.height = 0.11
		grip.mesh = cylinder
		grip.position = glove.position
		grip.rotation.z = PI * 0.5
		var rubber := StandardMaterial3D.new()
		rubber.albedo_color = Color("101112")
		rubber.roughness = 0.9
		grip.material_override = rubber
		stage.add_child(grip)
	var camera := Camera3D.new()
	camera.position = Vector3(0, 0.22, 0.35)
	stage.add_child(camera)
	camera.look_at(Vector3(0, 0, 0.04))
	camera.near = 0.01
	camera.fov = 55
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color("687c87")
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color.WHITE
	environment.environment.ambient_light_energy = 0.5
	stage.add_child(environment)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-45, -35, 0)
	stage.add_child(light)
	if DisplayServer.get_name() == "headless":
		print("GLOVE_GEOMETRY_CHECK passed")
		quit()
		return
	for i in range(12):
		await process_frame
	await RenderingServer.frame_post_draw
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--screenshot="):
			assert(root.get_texture().get_image().save_png(arg.trim_prefix("--screenshot=")) == OK)
	print("GLOVE_PREVIEW_COMPLETE")
	quit()
