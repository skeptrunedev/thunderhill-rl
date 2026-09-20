extends RefCounted
## Closed flat shaded chamfers stay within the requested box dimensions.


static func build(size: Vector3, bevel: float) -> ArrayMesh:
	assert(size.is_finite() and is_finite(bevel) and bevel > 0.0)
	var half_size := size * 0.5
	assert(bevel < minf(half_size.x, minf(half_size.y, half_size.z)))
	var core := half_size - Vector3.ONE * bevel
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	# Six inset rectangular faces.
	for axis in range(3):
		var u := (axis + 1) % 3
		var v := (axis + 2) % 3
		for side in [-1.0, 1.0]:
			var face: Array[Vector3] = []
			for corner in [Vector2(-1, -1), Vector2(-1, 1), Vector2(1, 1), Vector2(1, -1)]:
				var point := Vector3.ZERO
				point[axis] = side * half_size[axis]
				point[u] = corner.x * core[u]
				point[v] = corner.y * core[v]
				face.append(point)
			_face(surface, face)
	# Twelve chamfer strips, each joining a pair of inset faces.
	for a in range(3):
		for b in range(a + 1, 3):
			var along := 3 - a - b
			for sa in [-1.0, 1.0]:
				for sb in [-1.0, 1.0]:
					var p := Vector3.ZERO
					p[a] = sa * half_size[a]
					p[b] = sb * core[b]
					p[along] = -core[along]
					var q := p
					q[a] = sa * core[a]
					q[b] = sb * half_size[b]
					var r := q
					r[along] = core[along]
					var s := p
					s[along] = core[along]
					_face(surface, [p, q, r, s])
	# Eight triangular corner patches close the strips without overlapping faces.
	for sx in [-1.0, 1.0]:
		for sy in [-1.0, 1.0]:
			for sz in [-1.0, 1.0]:
				var signs := Vector3(sx, sy, sz)
				var face: Array[Vector3] = []
				for axis in range(3):
					var point := signs * core
					point[axis] = signs[axis] * half_size[axis]
					face.append(point)
				_face(surface, face)
	return surface.commit()


static func _face(surface: SurfaceTool, points: Array[Vector3]) -> void:
	var center := Vector3.ZERO
	for point in points:
		center += point / points.size()
	# Godot front faces wind clockwise; the conventional cross points inward.
	if (points[1] - points[0]).cross(points[2] - points[0]).dot(center) > 0:
		points.reverse()
	var normal := -(points[1] - points[0]).cross(points[2] - points[0]).normalized()
	for index in range(1, points.size() - 1):
		for vertex in [points[0], points[index], points[index + 1]]:
			surface.set_normal(normal)
			surface.add_vertex(vertex)
