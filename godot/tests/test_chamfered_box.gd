extends SceneTree
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	for size: Vector3 in [Vector3(0.036, 0.026, 0.05), Vector3(0.245, 0.028, 0.084), Vector3.ONE]:
		var mesh := preload("res://scripts/chamfered_box.gd").build(size, 0.004)
		var arrays := mesh.surface_get_arrays(0)
		var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
		check(vertices.size() == 132, "Chamfered box must have 44 triangles")
		var bounds := mesh.get_aabb()
		check(bounds.position.is_equal_approx(-size * 0.5), "Minimum extent changed")
		check(bounds.size.is_equal_approx(size), "Maximum extent changed")
		var edges := {}
		var volume := 0.0
		for index in range(0, vertices.size(), 3):
			var a := vertices[index]
			var b := vertices[index + 1]
			var c := vertices[index + 2]
			var cross_value := (b - a).cross(c - a)
			check(cross_value.length() > 0.00000001, "Degenerate triangle")
			check(cross_value.dot(a + b + c) < 0.0, "Inverted triangle winding")
			volume -= a.dot(b.cross(c)) / 6.0
			for offset in range(3):
				var normal := normals[index + offset]
				check(normal.is_finite() and absf(normal.length() - 1.0) < 0.0001, "Invalid normal")
				check(normal.dot(-cross_value.normalized()) > 0.9999, "Normal disagrees with face")
				var pair := [str(vertices[index + offset]), str(vertices[index + (offset + 1) % 3])]
				pair.sort()
				var key := str(pair)
				edges[key] = edges.get(key, 0) + 1
		for count in edges.values():
			check(count == 2, "Mesh is not a closed manifold")
		var core := size - Vector3.ONE * 0.008
		check(
			volume > core.x * core.y * core.z and volume < size.x * size.y * size.z,
			"Chamfer volume falls outside its analytic bounds"
		)
	print("CHAMFERED_BOX_CHECK failures=", failures)
	quit(failures)
