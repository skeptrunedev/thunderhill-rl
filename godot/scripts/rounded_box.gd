extends RefCounted
## Smooth fillets on a closed box, with exact requested outer dimensions.


static func build(size: Vector3, radius: float, segments := 4) -> ArrayMesh:
	assert(size.is_finite() and is_finite(radius) and radius > 0 and segments >= 2)
	var half := size * 0.5
	assert(radius < minf(half.x, minf(half.y, half.z)))
	var core := half - Vector3.ONE * radius
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	for axis in 3:
		var u := (axis + 1) % 3
		var v := (axis + 2) % 3
		var us := _coordinates(half[u], radius, segments)
		var vs := _coordinates(half[v], radius, segments)
		for sign_value in [-1.0, 1.0]:
			for i in us.size() - 1:
				for j in vs.size() - 1:
					var points: Array[Vector3] = []
					for corner in [
						Vector2i(i, j),
						Vector2i(i, j + 1),
						Vector2i(i + 1, j + 1),
						Vector2i(i + 1, j)
					]:
						var p := Vector3.ZERO
						p[axis] = half[axis] * sign_value
						p[u] = us[corner.x]
						p[v] = vs[corner.y]
						points.append(p)
					if sign_value < 0:
						points.reverse()
					for index in [0, 1, 2, 0, 2, 3]:
						var p := points[index]
						var closest := p.clamp(-core, core)
						var normal := (p - closest).normalized()
						surface.set_normal(normal)
						surface.add_vertex(closest + normal * radius)
	return surface.commit()


static func _coordinates(half: float, radius: float, segments: int) -> PackedFloat32Array:
	var values := PackedFloat32Array()
	for i in segments + 1:
		values.append(-half + radius * float(i) / segments)
	for i in segments + 1:
		values.append(half - radius + radius * float(i) / segments)
	return values
