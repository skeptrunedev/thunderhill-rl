extends SceneTree
## Functional dashboard validation and optional close cockpit visual QA.


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	var stage := Node3D.new()
	root.add_child(stage)
	var bike = load("res://scripts/bike_visual.gd").new()
	stage.add_child(bike)
	if not _check_tank(bike.get_node("FuelTank").mesh):
		quit(1)
		return
	var control_rings: Array[Vector4] = [
		Vector4(-0.5, 0.95, 0.12, 0.10), Vector4(0.4, 0.95, 0.10, 0.10)
	]
	var control: MeshInstance3D = bike._body_panel(control_rings, StandardMaterial3D.new(), stage)
	# Fault injection verifies a rejected helper cannot fall through to PASS.
	var inject_invalid := "--inject-invalid-tank" in OS.get_cmdline_user_args()
	if not _check_tank(control.mesh, inject_invalid):
		quit(1)
		return
	control.free()
	for cap in [false, true]:
		if not _check_reservoir(bike._reservoir_shell(cap), cap):
			quit(1)
			return
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


func _check_reservoir(mesh: ArrayMesh, cap: bool) -> bool:
	var arrays := mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
	var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
	var radius := 0.030 if cap else 0.028
	var bottom := -0.004 if cap else -0.0225
	var top := 0.010 if cap else 0.0225
	var edges := {}
	var signed_volume := 0.0
	for i in vertices.size():
		assert(vertices[i].is_finite() and normals[i].is_finite())
		assert(absf(normals[i].length() - 1.0) < 0.0001)
		assert(vertices[i].y >= bottom - 0.000001 and vertices[i].y <= top + 0.000001)
		assert(Vector2(vertices[i].x, vertices[i].z).length() <= radius + 0.000001)
	for i in range(0, indices.size(), 3):
		var a := vertices[indices[i]]
		var b := vertices[indices[i + 1]]
		var c := vertices[indices[i + 2]]
		var cross := (b - a).cross(c - a)
		# Flat molded rings must not inherit sloped normals from adjacent bevels.
		if cap and absf(a.y - b.y) < 0.0000001 and absf(a.y - c.y) < 0.0000001:
			for j in [i, i + 1, i + 2]:
				assert(
					absf(normals[indices[j]].y) > 0.999, "Planar cap face inherited bevel normals"
				)
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
	assert(signed_volume > 0 and signed_volume <= PI * radius * radius * (top - bottom))

	return true


func _check_tank(mesh: ArrayMesh, hollow_expected := true) -> bool:
	var arrays := mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
	var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
	var edges := {}
	var volume := 0.0
	var front_caps := 0
	var rear_caps := 0
	for i in range(0, indices.size(), 3):
		var a := vertices[indices[i]]
		var b := vertices[indices[i + 1]]
		var c := vertices[indices[i + 2]]
		var cross := (b - a).cross(c - a)
		assert(cross.length() > 0.000000001, "Degenerate tank triangle")
		assert(
			cross.dot(normals[indices[i]] + normals[indices[i + 1]] + normals[indices[i + 2]]) < 0,
			"Tank winding disagrees with normals"
		)
		volume -= a.dot(b.cross(c)) / 6.0
		if absf(a.z - b.z) < 0.000001 and absf(a.z - c.z) < 0.000001:
			var expected := Vector3.FORWARD if a.z < 0.0 else Vector3.BACK
			for j in [i, i + 1, i + 2]:
				assert(normals[indices[j]].dot(expected) > 0.999, "Tank cap faces inward")
			if a.z < 0.0:
				front_caps += 1
			else:
				rear_caps += 1
		for edge in [[a, b], [b, c], [c, a]]:
			var key: Array = edge.duplicate()
			key.sort()
			edges[key] = edges.get(key, 0) + 1
	for count in edges.values():
		assert(count == 2, "Tank shell is not watertight")
	assert(front_caps > 0 and rear_caps > 0)
	assert(volume > 0.0, "Tank has inward winding")
	# The rear crown stays convex and points outward after hollowing the nose.
	var checked_crown := false
	var front_center := -INF
	var front_shoulder := -INF
	var front_z := mesh.get_aabb().position.z
	for i in vertices.size():
		if absf(vertices[i].z - front_z) < 0.000001:
			if absf(vertices[i].x) < 0.001:
				front_center = maxf(front_center, vertices[i].y)
			elif absf(vertices[i].x) > 0.05 and absf(vertices[i].x) < 0.09:
				front_shoulder = maxf(front_shoulder, vertices[i].y)
		if vertices[i].y > 1.01 and absf(vertices[i].x) < 0.03 and absf(normals[i].z) < 0.999:
			assert(normals[i].y > 0.0, "Tank crown normal faces inward")
			checked_crown = true
	assert(checked_crown)
	assert(is_finite(front_center) and is_finite(front_shoulder))
	if hollow_expected:
		assert(front_center < front_shoulder - 0.03, "Tank nose lost its hollow")
	else:
		assert(front_center > front_shoulder, "Control loft should remain convex")
	return true
