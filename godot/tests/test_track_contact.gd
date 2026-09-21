extends SceneTree
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


const Track = preload("res://scripts/track.gd")


func _initialize():
	call_deferred("run")


func run():
	var track = Track.new()
	root.add_child(track)
	var upward = 0
	var downward = 0
	for child in track.get_children():
		if child is MeshInstance3D:
			var arrays = child.mesh.surface_get_arrays(0)
			for normal in arrays[Mesh.ARRAY_NORMAL]:
				if normal.y > 0:
					upward += 1
				else:
					downward += 1
	check(downward == 0, "All ground mesh normals must face up")
	var offroad_tests = 0
	var max_terrain_error = 0.0
	for z in range(-400, 900, 50):
		for x in range(-500, 500, 50):
			var p = Vector3(x, 0, z)
			var sampled = track.sample_world(p)
			check(is_finite(sampled.height) and sampled.normal.y > 0, "Invalid terrain contact")
			if sampled.planar_distance > sampled.width * .5 + track.SHOULDER_WIDTH:
				var difference = abs(sampled.height - track.terrain_height(p))
				max_terrain_error = max(max_terrain_error, difference)
				if difference >= 0.0002:
					print(
						"TERRAIN_DIFFERENCE ",
						p,
						" delta=",
						difference,
						" edge=",
						sampled.planar_distance - sampled.width * 0.5
					)
					quit(1)
					return
				offroad_tests += 1
	# Independent triangle centroid and cross-product checks verify that visible
	# offroad triangles, not a second interpolated heightfield, supply contact.
	var mesh_tests := 0
	for i in range(0, track.offroad_surface.triangles.size(), 97):
		var triangle: Array = track.offroad_surface.triangles[i]
		var a: Vector3 = track.offroad_surface.vertices[int(triangle[0])]
		var b: Vector3 = track.offroad_surface.vertices[int(triangle[1])]
		var c: Vector3 = track.offroad_surface.vertices[int(triangle[2])]
		var center := (a + b + c) / 3.0
		var sampled: Dictionary = track.sample_world(center)
		if sampled.planar_distance <= sampled.width * 0.5 + 1.0:
			continue
		var normal := (b - a).cross(c - a).normalized()
		if normal.y < 0.0:
			normal = -normal
		check(absf(sampled.height - center.y) < 0.0002, "Contact differs from visible triangle")
		check(
			sampled.normal.distance_to(normal) < 0.0002,
			"Contact normal differs from visible triangle"
		)
		mesh_tests += 1
	check(mesh_tests > 1000, "Insufficient ground contact coverage")
	var road_tests = 0
	for i in range(0, track.points.size(), 50):
		var sample = track.sample_world(track.points[i])
		check(sample.on_track, "Centerline must be on track")
		check(
			abs(sample.height - track.points[i].y - track.ROAD_LIFT) < 0.0001,
			"Centerline elevation changed"
		)
		road_tests += 1
	print(
		"Track contact audit: upward normals=",
		upward,
		" downward=",
		downward,
		" offroad=",
		offroad_tests,
		" mesh=",
		mesh_tests,
		" road=",
		road_tests,
		" max terrain error=",
		max_terrain_error
	)
	_test_rendered_pavement(track)
	_test_rendered_curbs(track)
	_test_removed_t2_curbs(track)
	track.queue_free()
	quit(failures)


func _test_rendered_curbs(track: Node3D) -> void:
	var mesh_node := track.get_node_or_null("ProvisionalCurbs") as MeshInstance3D
	check(mesh_node != null, "Rendered curb mesh is present")
	if mesh_node == null:
		return
	var arrays := mesh_node.mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var indices := PackedInt32Array()
	if arrays[Mesh.ARRAY_INDEX] != null:
		indices = arrays[Mesh.ARRAY_INDEX]
	if indices.is_empty():
		for index in vertices.size():
			indices.append(index)
	check(indices.size() == 522 * 3, "Reviewed fixture covers all 522 remaining curb triangles")
	var count := 0
	var ground_above := 0
	var curb_flags := 0
	var max_error := 0.0
	var max_normal_error := 0.0
	for index in range(0, indices.size(), 3):
		var a := mesh_node.transform * vertices[indices[index]]
		var b := mesh_node.transform * vertices[indices[index + 1]]
		var c := mesh_node.transform * vertices[indices[index + 2]]
		# Derive expectations from rendered arrays, never CurbSurface internals.
		var normal := -(b - a).cross(c - a).normalized()
		check(normal.y > 0.0, "Rendered curb winding faces upward")
		for weight in [
			Vector3.ONE / 3.0, Vector3(.8, .1, .1), Vector3(.1, .8, .1), Vector3(.1, .1, .8)
		]:
			var point: Vector3 = a * weight.x + b * weight.y + c * weight.z
			var ground: Dictionary = track.offroad_surface.sample(point)
			check(not ground.has("error"), "Underlying curb ground query succeeds")
			if ground.has("error"):
				continue
			var expected_height := float(point.y)
			var expected_normal := normal
			var separation := -INF
			if ground.has("height"):
				separation = float(ground.height) - expected_height
				if separation > 0.0:
					expected_height = float(ground.height)
					expected_normal = ground.normal
			var sampled: Dictionary = track.sample_world(point)
			var label := "triangle %d, weights %s" % [index / 3, weight]
			var height_error := absf(float(sampled.height) - expected_height)
			max_error = maxf(max_error, height_error)
			check(height_error < .0002, "Curb visible upper height differs: " + label)
			# Float32 world positions can perturb the side of an exact height tie.
			# Away from ties, both face normal and exposed surface identity must agree.
			if absf(separation) > .0002:
				var normal_error: float = sampled.normal.distance_to(expected_normal)
				max_normal_error = maxf(max_normal_error, normal_error)
				check(normal_error < .0002, "Curb visible upper normal differs: " + label)
				check(sampled.on_curb == (separation < 0.0), "Curb exposure flag differs: " + label)
				if separation < 0.0:
					curb_flags += 1
					check(
						sampled.get("curb_triangle_id", -1) == index / 3,
						"Exposed curb triangle identity differs: " + label
					)
				else:
					ground_above += 1
					check(
						not sampled.has("curb_triangle_id"),
						"Covered curb leaks triangle identity: " + label
					)
			count += 1
	check(count == 2088, "Every curb triangle supplies four contact samples")
	check(curb_flags > 0 and ground_above > 0, "Both exposed and ground covered curbs exercised")
	print(
		"CURB_TRACK_CONTACT samples=",
		count,
		" exposed=",
		curb_flags,
		" ground_above=",
		ground_above,
		" max_height_error_m=",
		max_error,
		" max_normal_error=",
		max_normal_error,
		" failures=",
		failures
	)


func _test_rendered_pavement(track: Node3D) -> void:
	var mesh_node := track.get_node_or_null("RacingSurface") as MeshInstance3D
	check(mesh_node != null, "Rendered pavement mesh is present")
	if mesh_node == null:
		return
	var arrays := mesh_node.mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var indices := PackedInt32Array()
	if arrays[Mesh.ARRAY_INDEX] != null:
		indices = arrays[Mesh.ARRAY_INDEX]
	if indices.is_empty():
		for index in vertices.size():
			indices.append(index)
	var pavement_data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/pavement.json")
	)
	check(
		indices.size() == pavement_data.triangles.size() * 3,
		"Every authored pavement triangle rendered"
	)
	check(indices.size() >= 3072 * 3, "Historical pavement coverage preserved")
	var boundary_points: Dictionary = {}
	var count := 0
	var max_error := 0.0
	var max_normal_error := 0.0
	for index in range(0, indices.size(), 3):
		var a := mesh_node.transform * vertices[indices[index]]
		var b := mesh_node.transform * vertices[indices[index + 1]]
		var c := mesh_node.transform * vertices[indices[index + 2]]
		for point in [(a + b) * .5, (b + c) * .5, (c + a) * .5]:
			if not boundary_points.has(point):
				boundary_points[point] = "edge midpoint"
		for point in [a, b, c]:
			boundary_points[point] = "vertex"
		# Derive expectations directly from rendered arrays, independent of the sampler.
		var normal := -(b - a).cross(c - a).normalized()
		check(normal.y > 0.0, "Rendered pavement winding faces upward")
		for weight in [
			Vector3.ONE / 3.0, Vector3(.8, .1, .1), Vector3(.1, .8, .1), Vector3(.1, .1, .8)
		]:
			var point: Vector3 = a * weight.x + b * weight.y + c * weight.z
			var sampled: Dictionary = track.sample_world(point)
			var label := "triangle %d, weights %s" % [index / 3, weight]
			var height_error := absf(float(sampled.height) - float(point.y))
			var normal_error: float = sampled.normal.distance_to(normal)
			max_error = maxf(max_error, height_error)
			max_normal_error = maxf(max_normal_error, normal_error)
			check(height_error < .0002, "Pavement visible height differs: " + label)
			check(normal_error < .0002, "Pavement visible normal differs: " + label)
			check(sampled.get("on_pavement", false), "Exposed pavement flag missing: " + label)
			check(not sampled.on_curb, "Pavement incorrectly reports curb: " + label)
			count += 1
	check(count == indices.size() / 3 * 4, "Every pavement triangle supplies four contact samples")
	print(
		"PAVEMENT_TRACK_CONTACT samples=",
		count,
		" max_height_error_m=",
		max_error,
		" max_normal_error=",
		max_normal_error,
		" failures=",
		failures
	)

	var max_boundary_error := 0.0
	for point: Vector3 in boundary_points:
		# Shared edges can select either face normal. Height still has to agree
		# with the rendered road or an actually higher curb or terrain triangle.
		var expected_height := float(point.y)
		for surface in [track.pavement_surface, track.curb_surface, track.offroad_surface]:
			var covering: Dictionary = (
				surface.sample_mesh(point)
				if surface == track.offroad_surface
				else surface.sample(point)
			)
			check(not covering.has("error"), "Boundary covering surface query succeeds")
			if covering.has("height"):
				expected_height = maxf(expected_height, float(covering.height))
		var sampled: Dictionary = track.sample_world(point)
		check(
			is_finite(sampled.height),
			"Road boundary %s contact must be finite at %s" % [boundary_points[point], point]
		)
		if not is_finite(sampled.height):
			continue
		var error := absf(float(sampled.height) - expected_height)
		max_boundary_error = maxf(max_boundary_error, error)
		check(error < .0002, "Road boundary visible height differs at %s by %s" % [point, error])
	check(boundary_points.size() >= 3072, "All rendered road vertices and edge midpoints tested")
	print(
		"PAVEMENT_BOUNDARY_CONTACT samples=",
		boundary_points.size(),
		" max_height_error_m=",
		max_boundary_error,
		" failures=",
		failures
	)


func _test_removed_t2_curbs(track: Node3D) -> void:
	var count := 0
	var original: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://tests/fixtures/curb-curvature-baseline.json")
	)
	for item: Array in original.segments_and_sides:
		var index := int(item[0])
		if int(item[1]) != 1 or index < 324 or index >= 416:
			continue
		var next := index + 1
		var a: Vector3 = track.edge_point(
			index, float(track.samples[index].width) * 0.5 + 0.45, 0.08
		)
		var b: Vector3 = track.edge_point(next, float(track.samples[next].width) * 0.5 + 0.45, 0.08)
		var point := (a + b) * 0.5
		check(
			track.curb_surface.sample(point).is_empty(), "Removed T2 curb still supplies geometry"
		)
		check(not track.sample_world(point).on_curb, "Removed T2 curb still affects contact")
		count += 1
	check(count == 36, "All removed curb segment footprints tested")
	print("REMOVED_T2_CURB_CONTACT samples=", count, " failures=", failures)
