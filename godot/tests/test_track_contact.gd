extends SceneTree
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
	assert(downward == 0, "All ground mesh normals must face up")
	var offroad_tests = 0
	var max_terrain_error = 0.0
	for z in range(-400, 900, 50):
		for x in range(-500, 500, 50):
			var p = Vector3(x, 0, z)
			var sampled = track.sample_world(p)
			assert(is_finite(sampled.height) and sampled.normal.y > 0)
			if sampled.planar_distance > sampled.width * .5 + track.SHOULDER_WIDTH:
				var difference = abs(sampled.height - track.terrain_height(p))
				max_terrain_error = max(max_terrain_error, difference)
				assert(difference < 0.00001, "Offroad contact must be terrain")
				assert(sampled.normal.distance_to(track.terrain_normal(p)) < 0.00001)
				offroad_tests += 1
	var road_tests = 0
	for i in range(0, track.points.size(), 50):
		var sample = track.sample_world(track.points[i])
		assert(sample.on_track)
		assert(abs(sample.height - track.points[i].y - track.ROAD_LIFT) < 0.0001)
		road_tests += 1
	print(
		"Track contact audit: upward normals=",
		upward,
		" downward=",
		downward,
		" offroad=",
		offroad_tests,
		" road=",
		road_tests,
		" max terrain error=",
		max_terrain_error
	)
	track.queue_free()
	quit()
