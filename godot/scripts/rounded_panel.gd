extends RefCounted
## Closed molded panel with circular XY corners and rounded front and back rims.

const CORNER_STEPS := 6
const RIM_STEPS := 3


static func build(
	size: Vector2, depth: float, corner_radius: float, edge_radius: float
) -> ArrayMesh:
	assert(size.is_finite() and is_finite(depth))
	assert(is_finite(corner_radius) and is_finite(edge_radius))
	assert(size.x > 0.0 and size.y > 0.0 and depth > 0.0)
	assert(corner_radius > 0.0 and corner_radius < minf(size.x, size.y) * 0.5)
	assert(edge_radius > 0.0 and edge_radius < minf(corner_radius, depth * 0.5))
	var half_size := size * 0.5
	var centers: Array[Vector2] = []
	var directions: Array[Vector2] = []
	# Counterclockwise outline; each arc includes both tangent endpoints.
	# Consecutive arcs connect with the four straight sides.
	for corner in range(4):
		var middle := (float(corner) + 0.5) * PI * 0.5
		var center := Vector2(signf(cos(middle)), signf(sin(middle)))
		center *= half_size - Vector2.ONE * corner_radius
		for step in range(CORNER_STEPS + 1):
			var angle := (float(corner) + float(step) / CORNER_STEPS) * PI * 0.5
			centers.append(center)
			directions.append(Vector2(cos(angle), sin(angle)))
	var rings: Array[PackedVector3Array] = []
	var normals: Array[PackedVector3Array] = []
	for side: float in [-1.0, 1.0]:
		for step in range(RIM_STEPS + 1):
			var fraction := float(step) / RIM_STEPS
			var angle := lerpf(-PI * 0.5, 0.0, fraction) if side < 0 else fraction * PI * 0.5
			var radial := cos(angle)
			var axial := sin(angle)
			var radius := corner_radius - edge_radius + edge_radius * radial
			var z := side * (depth * 0.5 - edge_radius) + edge_radius * axial
			var ring := PackedVector3Array()
			var ring_normals := PackedVector3Array()
			for index in range(centers.size()):
				var point := centers[index] + directions[index] * radius
				ring.append(Vector3(point.x, point.y, z))
				var direction := directions[index]
				ring_normals.append(Vector3(direction.x * radial, direction.y * radial, axial))
			rings.append(ring)
			normals.append(ring_normals)
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	for ring in range(rings.size() - 1):
		for index in range(centers.size()):
			var next := (index + 1) % centers.size()
			_triangle(
				surface,
				[rings[ring][index], rings[ring][next], rings[ring + 1][next]],
				[normals[ring][index], normals[ring][next], normals[ring + 1][next]]
			)
			_triangle(
				surface,
				[rings[ring][index], rings[ring + 1][next], rings[ring + 1][index]],
				[normals[ring][index], normals[ring + 1][next], normals[ring + 1][index]]
			)
	for ring in [0, rings.size() - 1]:
		var side := -1.0 if ring == 0 else 1.0
		var normal := Vector3(0, 0, side)
		for index in range(centers.size()):
			_triangle(
				surface,
				[
					Vector3(0, 0, side * depth * 0.5),
					rings[ring][index],
					rings[ring][(index + 1) % centers.size()]
				],
				[normal, normal, normal]
			)
	return surface.commit()


static func _triangle(
	surface: SurfaceTool, points: Array[Vector3], normals: Array[Vector3]
) -> void:
	# Godot front faces wind clockwise when viewed from outside.
	if (
		(points[1] - points[0]).cross(points[2] - points[0]).dot(
			normals[0] + normals[1] + normals[2]
		)
		> 0.0
	):
		points.reverse()
		normals.reverse()
	for index in range(3):
		surface.set_normal(normals[index])
		surface.add_vertex(points[index])
