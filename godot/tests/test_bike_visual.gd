extends SceneTree
## Functional dashboard validation and optional close cockpit visual QA.


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	var stage := Node3D.new()
	root.add_child(stage)
	var bike = load("res://scripts/bike_visual.gd").new()
	stage.add_child(bike)
	for cap in [false, true]:
		_check_reservoir(bike._reservoir_shell(cap), cap)
	# The reflective face must be planar and face the rider, not inherit side normals.
	var housing: MeshInstance3D = bike._front.get_node("LiveInstrumentCluster").get_child(0)
	var panel_arrays := housing.mesh.surface_get_arrays(0)
	var panel_vertices: PackedVector3Array = panel_arrays[Mesh.ARRAY_VERTEX]
	var panel_normals: PackedVector3Array = panel_arrays[Mesh.ARRAY_NORMAL]
	var checked := 0
	for triangle in range(0, panel_vertices.size(), 3):
		if (
			panel_vertices[triangle].z > 0
			and panel_vertices[triangle + 1].z > 0
			and panel_vertices[triangle + 2].z > 0
		):
			for vertex in range(triangle, triangle + 3):
				assert(
					panel_normals[vertex].distance_to(Vector3.BACK) < 0.0001,
					"Front normal: " + str(panel_normals[vertex])
				)
			checked += 1
	assert(checked == 8)
	bike.set_rider_visible(false)
	bike.update_instruments(25.0, 7200.0, 3, 72.34)
	assert(bike._display.speed_text == "090")
	assert(bike._display.gear_text == "3")
	assert(bike._display.rpm_text == "7200 RPM")
	assert(bike._display.lap_text == "01:12.34")
	assert(not bike._display.set_readings(25.0, 7200.0, 3, 72.34))
	bike.update_instruments(-10.0, 1400.0, 0, 0)
	assert(bike._display.speed_text == "036" and bike._display.gear_text == "N")
	assert(bike._display.lap_text == "00:00.00")
	bike.update_instruments(25.0, 7200.0, 3, 72.34)
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


func _check_reservoir(mesh: ArrayMesh, cap: bool) -> void:
	var arrays := mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
	var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
	var radius := 0.030 if cap else 0.028
	var half_height := 0.004 if cap else 0.0225
	var edges := {}
	var signed_volume := 0.0
	for i in vertices.size():
		assert(vertices[i].is_finite() and normals[i].is_finite())
		assert(absf(normals[i].length() - 1.0) < 0.0001)
		assert(absf(vertices[i].y) <= half_height + 0.000001)
		assert(Vector2(vertices[i].x, vertices[i].z).length() <= radius + 0.000001)
	for i in range(0, indices.size(), 3):
		var a := vertices[indices[i]]
		var b := vertices[indices[i + 1]]
		var c := vertices[indices[i + 2]]
		var cross := (b - a).cross(c - a)
		assert(cross.length() > 0.00000001)
		assert(
			cross.dot(normals[indices[i]] + normals[indices[i + 1]] + normals[indices[i + 2]]) < 0
		)
		signed_volume -= a.dot(b.cross(c)) / 6.0
		for edge in [[a, b], [b, c], [c, a]]:
			# Weld identical positions across hard normal boundaries.
			var key: Array = edge.duplicate()
			key.sort()
			edges[key] = edges.get(key, 0) + 1
	for count in edges.values():
		assert(count == 2)
	assert(signed_volume > 0 and signed_volume <= PI * radius * radius * 2 * half_height)
