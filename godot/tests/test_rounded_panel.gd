extends SceneTree

var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	for dimensions: Vector3 in [
		Vector3(0.196, 0.119, 0.025), Vector3(0.119, 0.196, 0.04), Vector3.ONE
	]:
		var mesh := preload("res://scripts/rounded_panel.gd").build(
			Vector2(dimensions.x, dimensions.y), dimensions.z, 0.012, 0.0025
		)
		var arrays := mesh.surface_get_arrays(0)
		var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
		check(vertices.size() == 1344, "Expected 448 triangles")
		check(mesh.get_aabb().position.is_equal_approx(-dimensions * 0.5), "Minimum bounds changed")
		check(mesh.get_aabb().size.is_equal_approx(dimensions), "Maximum bounds changed")
		var edges := {}
		var front_faces := 0
		var back_faces := 0
		for index in range(0, vertices.size(), 3):
			var a := vertices[index]
			var b := vertices[index + 1]
			var c := vertices[index + 2]
			var cross_value := (b - a).cross(c - a)
			check(cross_value.length() > 1e-10, "Degenerate triangle")
			check(cross_value.dot(a + b + c) < 0.0, "Inverted winding")
			var face_normal := -cross_value.normalized()
			if absf(a.z - b.z) < 1e-7 and absf(a.z - c.z) < 1e-7:
				var side := signf(a.z)
				check(
					face_normal.dot(Vector3(0, 0, side)) > 0.9999, "Cap face normal is not planar"
				)
				if side > 0:
					front_faces += 1
				else:
					back_faces += 1
			for offset in range(3):
				var normal := normals[index + offset]
				check(
					normal.is_finite() and absf(normal.length() - 1.0) < 0.0001,
					"Invalid unit normal"
				)
				check(normal.dot(face_normal) > 0.95, "Normal disagrees with face")
				var vertex := vertices[index + offset]
				var planar := Vector2(vertex.x, vertex.y)
				if planar.length_squared() > 1e-12:
					var corner_core := (
						Vector2(dimensions.x, dimensions.y) * 0.5 - Vector2.ONE * 0.012
					)
					var offset_from_core := planar - planar.clamp(-corner_core, corner_core)
					var normal_radial := Vector2(normal.x, normal.y).length()
					check(
						(
							absf(
								(
									offset_from_core.length()
									- (0.012 - 0.0025 + 0.0025 * normal_radial)
								)
							)
							< 1e-6
						),
						"Corner or rim radius changed"
					)
					if normal_radial > 0.0001:
						check(
							(
								offset_from_core.normalized().dot(
									Vector2(normal.x, normal.y).normalized()
								)
								> 0.9999
							),
							"Radial normal is not analytic"
						)
				# Position welding checks topology despite separate cap/rim vertices.
				var first := vertices[index + offset].snapped(Vector3.ONE * 1e-7)
				var second := vertices[index + (offset + 1) % 3].snapped(Vector3.ONE * 1e-7)
				var pair := [str(first), str(second)]
				pair.sort()
				var key := str(pair)
				edges[key] = edges.get(key, 0) + 1
		check(front_faces == 28 and back_faces == 28, "Both caps must be closed")
		for count in edges.values():
			check(count == 2, "Mesh has an open or nonmanifold edge")
	print("ROUNDED_PANEL_CHECK failures=", failures)
	quit(failures)
