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
	track.queue_free()
	quit(failures)
