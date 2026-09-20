extends SceneTree
const Surface = preload("res://scripts/offroad_surface.gd")
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func _initialize() -> void:
	_test_ring_contains()
	_test_actual_road_rings()
	var surface := Surface.new()
	surface.configure(
		{
			"bounds": {"x0": 0.0, "z0": 0.0, "x1": 2.0, "z1": 2.0},
			"vertices": [[0, 0, 0], [2, 2, 0], [0, 0, 2]],
			"triangles": [[0, 1, 2]],
			"road_rings": [[[2, 0], [2, 2], [0, 2]]]
		}
	)
	for point in [Vector3(0.5, 100, 0.5), Vector3(1, -100, 0), Vector3(0, 0, 2)]:
		var result := surface.sample(point)
		check(not result.has("error") and not result.road_cutout, "Known triangle point missing")
		check(
			absf(result.height - point.x) < 0.00001, "Height does not match independent plane y=x"
		)
		check(
			result.normal.distance_to(Vector3(-1, 1, 0).normalized()) < 0.00001,
			"Normal does not match analytic plane"
		)
	check(
		surface.sample(Vector3(1.5, 0, 1.5)).get("road_cutout", false), "Road cutout not identified"
	)
	check(absf(surface.sample(Vector3(-1, 0, 0)).height) < 0.00001, "World edge clamp failed")
	# A second ring is a hole in the cutout. This fixture intentionally has no
	# terrain inside that hole, so it must report missing terrain, not road.
	surface.road_rings.append(
		PackedVector2Array(
			[Vector2(1.3, 1.3), Vector2(1.6, 1.3), Vector2(1.6, 1.6), Vector2(1.3, 1.6)]
		)
	)
	check(
		surface.sample(Vector3(1.5, 0, 1.5)).has("error"),
		"Hole was incorrectly filled by the road cutout"
	)
	check(
		surface.sample(Vector3(1.7, 0, 1.7)).get("road_cutout", false),
		"Adding a hole removed the surrounding road cutout"
	)
	surface.road_rings.clear()
	check(
		surface.sample(Vector3(1.5, 0, 1.5)).has("error"), "Uncovered terrain was silently accepted"
	)
	print("OFFROAD_SURFACE_CHECK failures=", failures)
	quit(failures)


func _test_ring_contains() -> void:
	# The notch makes the horizontal ray cross vertices and horizontal edges.
	var ring := PackedVector2Array(
		[Vector2(0, 0), Vector2(4, 0), Vector2(4, 4), Vector2(2, 2), Vector2(0, 4)]
	)
	var cases := [
		[Vector2(2, 1), true],
		[Vector2(2, 3), false],
		[Vector2(1, 2), true],
		[Vector2(3, 2), true],
		[Vector2(-1, 2), false],
		[Vector2(5, 2), false],
		[Vector2(2, 0), true],
		[Vector2(2, -0.0001), false],
		[Vector2(0, 2), true],
		[Vector2(4, 2), true],
		[Vector2(3, 3), true],
		[Vector2(3, 3.0001), false],
		[Vector2(3, 2.9999), true],
	]
	for vertex in ring:
		cases.append([vertex, true])
	for offset in [Vector2.ZERO, Vector2(128, -256), Vector2(-128, 256)]:
		var translated := PackedVector2Array()
		for vertex in ring:
			translated.append(vertex + offset)
		for reverse in [false, true]:
			if reverse:
				translated.reverse()
			for row in cases:
				# Small offsets above remain distinct at these float32 coordinates.
				var query: Vector2 = row[0] + offset
				check(
					Surface.ring_contains(query, translated) == row[1],
					(
						"Ring membership differs for %s, offset %s, reversed %s"
						% [row[0], offset, reverse]
					)
				)
	check(
		not Surface.ring_contains(Vector2.ZERO, PackedVector2Array()), "Empty ring contains a point"
	)
	check(
		not Surface.ring_contains(Vector2.ZERO, PackedVector2Array([Vector2.ZERO, Vector2.ONE])),
		"Two point ring contains a point"
	)
	var closed := ring.duplicate()
	closed.append(closed[0])
	check(
		Surface.ring_contains(Vector2(2, 1), closed), "Repeated closing vertex changed containment"
	)


func _test_actual_road_rings() -> void:
	var data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/surface.json")
	)
	check(data.road_rings.size() == 2, "Historical road ring fixture changed")
	var query := Vector2(7.99179458618164, 69.0894088745117)
	for i in data.road_rings.size():
		var ring := PackedVector2Array()
		for point in data.road_rings[i]:
			ring.append(Vector2(point[0], point[1]))
		# Independent Shapely classification agrees before and after float32
		# quantization: outer ring true, inner ring false. The old oblique ray
		# double counted a distant vertex of the outer ring.
		check(
			Surface.ring_contains(query, ring) == (i == 0),
			"Actual road cutout membership failed for ring %d" % i
		)
