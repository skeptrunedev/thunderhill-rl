extends SceneTree
const ENVELOPE := preload("res://scripts/bike_collision_envelope.gd")
const SWEEP := preload("res://scripts/bike_sweep.gd")
const BIKE := preload("res://scripts/bike_visual.gd")


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var bike := BIKE.new()
	root.add_child(bike)
	await process_frame
	var envelope := ENVELOPE.new()
	assert(envelope.build(bike).is_empty())
	var sweep := SWEEP.new()
	assert(sweep.build(envelope).is_empty())
	var failures := 0
	var checks := 0
	var rng := RandomNumberGenerator.new()
	rng.seed = 123419
	for part in sweep._parts:
		for pose_index in 3:
			var rotation := Basis.from_euler(
				Vector3(rng.randf_range(-3, 3), rng.randf_range(-3, 3), rng.randf_range(-3, 3))
			)
			var center: Vector3 = [
				Vector3.ZERO, Vector3(4096, -4096, 4096), Vector3(-4096, 4096, -4096)
			][pose_index]
			var points := PackedVector3Array()
			for p in part.points:
				points.append(rotation * p + center)
			var padding: float = [0.0005, 0.01, 0.5][pose_index]
			var expanded: PackedVector3Array = SWEEP._expand_masked(
				points, SWEEP._corner_masks(points, part.witnesses), padding, center
			)
			var directions := PackedVector3Array()
			for x in [-1.0, 0.0, 1.0]:
				for y in [-1.0, 0.0, 1.0]:
					for z in [-1.0, 0.0, 1.0]:
						if x != 0 or y != 0 or z != 0:
							directions.append(Vector3(x, y, z))
			for k in 32:
				directions.append(
					Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1))
				)
			for direction in directions:
				var offset := (
					Vector3(
						1.0 if direction.x >= 0 else -1.0,
						1.0 if direction.y >= 0 else -1.0,
						1.0 if direction.z >= 0 else -1.0
					)
					* padding
				)
				var expected := -INF
				for p in points:
					expected = maxf(expected, (p - center + offset).dot(direction))
				var actual := -INF
				for p in expanded:
					actual = maxf(actual, p.dot(direction))
				checks += 1
				if expected != actual:
					failures += 1
	var first: Dictionary = sweep._endpoint_cloud(0, 1.0 - pow(2.0, -30), Transform3D.IDENTITY)
	var second: Dictionary = sweep._endpoint_cloud(
		0, 1.0 - pow(2.0, -31), Transform3D(Basis.IDENTITY, Vector3.ONE)
	)
	checks += 1
	if first.points[0] == second.points[0] or sweep._endpoint_clouds[0].size() != 2:
		failures += 1
	print("REDUCTION_SUPPORT checks=", checks, " failures=", failures)
	bike.queue_free()
	quit(1 if failures else 0)
