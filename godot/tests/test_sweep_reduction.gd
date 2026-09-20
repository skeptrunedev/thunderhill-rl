extends SceneTree
const ENVELOPE := preload("res://scripts/bike_collision_envelope.gd")
const SWEEP := preload("res://scripts/bike_sweep.gd")
const BIKE := preload("res://scripts/bike_visual.gd")


func _initialize() -> void:
	_run.call_deferred()


func _union_checks(sweep: RefCounted) -> Dictionary:
	var checks := 0
	var failures := 0
	# One source vertex exposes all four discarded corners for either sign of
	# each axial translation. No within-cloud witness can hide a mask error.
	for axis in 3:
		for sign_value in [-1.0, 1.0]:
			var delta := Vector3.ZERO
			delta[axis] = sign_value
			var masks: Array = SWEEP._paired_masks(
				PackedVector3Array([Vector3.ZERO]),
				PackedVector3Array([delta]),
				PackedInt32Array([255]),
				PackedInt32Array([255])
			)
			var retained := 0
			for mask in masks:
				var count := 0
				for corner in 8:
					count += 1 if mask[0] & (1 << corner) else 0
				checks += 1
				if count != 4:
					failures += 1
				retained += count
			checks += 1
			if retained != 8:
				failures += 1
	var directions := PackedVector3Array()
	for x in [-1.0, 0.0, 1.0]:
		for y in [-1.0, 0.0, 1.0]:
			for z in [-1.0, 0.0, 1.0]:
				if x != 0 or y != 0 or z != 0:
					directions.append(Vector3(x, y, z))
	var rng := RandomNumberGenerator.new()
	rng.seed = 548102
	for index in 32:
		directions.append(
			Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1))
		)
	var sources: Array[PackedVector3Array] = []
	for part in sweep._parts:
		sources.append(part.points)
	# Repeated vertices within clouds, equal coordinates, and different witness
	# paths exercise the acyclic dominance argument together with endpoint masks.
	sources.append(PackedVector3Array([Vector3.ZERO, Vector3.ZERO, Vector3.ONE, Vector3(0, 1, 0)]))
	for source_index in sources.size():
		var source := sources[source_index]
		var witnesses: Array[PackedInt32Array] = SWEEP._witness_indices(source)
		for case_index in 4:
			var center: Vector3 = [
				Vector3.ZERO, Vector3(4096, -4096, 4096), Vector3(-4096, 4096, -4096), Vector3.ZERO
			][case_index]
			var padding: float = [0.0005, 0.01, 0.5, 0.0005][case_index]
			var base := Basis.from_euler(Vector3(0.27, -0.59, 0.81))
			var finish := Transform3D(base, center)
			if case_index == 0:
				finish.origin[source_index % 3] += -0.3 if source_index % 2 else 0.3
			elif case_index == 1:
				finish.origin += Vector3(0.13, -0.27, 0.19)
			elif case_index == 3:
				finish.basis = Basis.from_euler(Vector3(-0.63, 0.92, 0.17))
			var first: PackedVector3Array = Transform3D(base, center) * source
			var second: PackedVector3Array = finish * source
			var first_masks: PackedInt32Array = SWEEP._corner_masks(first, witnesses)
			var second_masks: PackedInt32Array = SWEEP._corner_masks(second, witnesses)
			var saved_first := first_masks.duplicate()
			var saved_second := second_masks.duplicate()
			var paired: Array = SWEEP._paired_masks(first, second, first_masks, second_masks)
			checks += 1
			if first_masks != saved_first or second_masks != saved_second:
				failures += 1
			var expanded: PackedVector3Array = SWEEP._expand_masked(
				first, paired[0], padding, center
			)
			expanded.append_array(SWEEP._expand_masked(second, paired[1], padding, center))
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
				for cloud in [first, second]:
					for point in cloud:
						expected = maxf(expected, (point - center + offset).dot(direction))
				var actual := -INF
				for point in expanded:
					actual = maxf(actual, point.dot(direction))
				checks += 1
				if actual != expected:
					failures += 1
	# Use actual cache entries as well, including reusing the first endpoint
	# against a different endpoint after the first call.
	var cached_a: Dictionary = sweep._endpoint_cloud(0, 0.25, Transform3D.IDENTITY)
	var cached_b: Dictionary = sweep._endpoint_cloud(
		0, 0.5, Transform3D(Basis.IDENTITY, Vector3.RIGHT)
	)
	var saved_a: PackedInt32Array = cached_a.masks.duplicate()
	var saved_b: PackedInt32Array = cached_b.masks.duplicate()
	SWEEP._paired_masks(cached_a.points, cached_b.points, cached_a.masks, cached_b.masks)
	SWEEP._paired_masks(cached_a.points, cached_a.points, cached_a.masks, cached_a.masks)
	checks += 1
	if cached_a.masks != saved_a or cached_b.masks != saved_b:
		failures += 1
	print("UNION_SUPPORT checks=", checks, " failures=", failures)
	return {"checks": checks, "failures": failures}


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
	var union_result := _union_checks(sweep)
	checks += union_result.checks
	failures += union_result.failures
	sweep._endpoint_clouds.clear()
	var first: Dictionary = sweep._endpoint_cloud(0, 1.0 - pow(2.0, -30), Transform3D.IDENTITY)
	var second: Dictionary = sweep._endpoint_cloud(
		0, 1.0 - pow(2.0, -31), Transform3D(Basis.IDENTITY, Vector3.ONE)
	)
	checks += 1
	if first.points[0] == second.points[0] or sweep._endpoint_clouds[0].size() != 2:
		failures += 1
	for limit in ["points", "entries"]:
		sweep._endpoint_clouds.clear()
		sweep._cached_points = SWEEP.MAX_CACHED_POINTS if limit == "points" else 0
		sweep._cached_endpoints = SWEEP.MAX_CACHED_ENDPOINTS if limit == "entries" else 0
		var uncached: Dictionary = sweep._endpoint_cloud(0, 0.5, Transform3D.IDENTITY)
		checks += 1
		if not sweep._endpoint_clouds[0].is_empty() or uncached != first:
			failures += 1
	print("REDUCTION_SUPPORT checks=", checks, " failures=", failures)
	bike.queue_free()
	quit(1 if failures else 0)
