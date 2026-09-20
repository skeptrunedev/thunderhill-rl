class_name OffroadSurface
extends RefCounted
## The same indexed triangles supply offroad rendering and contact height.
const CELL := 16.0
var vertices := PackedVector3Array()
var triangles: Array = []
var grid: Dictionary = {}
var road_rings: Array[PackedVector2Array] = []
var bounds: Dictionary


func configure(data: Dictionary) -> void:
	vertices.clear()
	triangles.clear()
	grid.clear()
	road_rings.clear()
	bounds = data.bounds
	for p in data.vertices:
		vertices.append(Vector3(p[0], p[1], p[2]))
	triangles = data.triangles
	for ring in data.road_rings:
		var polygon := PackedVector2Array()
		for p in ring:
			polygon.append(Vector2(p[0], p[1]))
		road_rings.append(polygon)
	for index in triangles.size():
		var t: Array = triangles[index]
		var a := vertices[int(t[0])]
		var b := vertices[int(t[1])]
		var c := vertices[int(t[2])]
		for z in range(
			floori(minf(a.z, minf(b.z, c.z)) / CELL), floori(maxf(a.z, maxf(b.z, c.z)) / CELL) + 1
		):
			for x in range(
				floori(minf(a.x, minf(b.x, c.x)) / CELL),
				floori(maxf(a.x, maxf(b.x, c.x)) / CELL) + 1
			):
				var key := Vector2i(x, z)
				if not grid.has(key):
					grid[key] = []
				grid[key].append(index)


func sample_mesh(p: Vector3) -> Dictionary:
	var q := Vector2(clampf(p.x, bounds.x0, bounds.x1), clampf(p.z, bounds.z0, bounds.z1))
	var key := Vector2i(floori(q.x / CELL), floori(q.y / CELL))
	for index in grid.get(key, []):
		var t: Array = triangles[index]
		var a := vertices[int(t[0])]
		var b := vertices[int(t[1])]
		var c := vertices[int(t[2])]
		var point := Vector3(q.x, 0.0, q.y)
		var determinant := TriangleRibbon.projected_cross(a, b, c)
		if determinant == 0.0:
			continue
		var orientation := signf(determinant)
		if (
			orientation * TriangleRibbon.projected_cross(a, b, point) < 0.0
			or orientation * TriangleRibbon.projected_cross(b, c, point) < 0.0
			or orientation * TriangleRibbon.projected_cross(c, a, point) < 0.0
		):
			continue
		var u := TriangleRibbon.projected_cross(a, point, c) / determinant
		var v := TriangleRibbon.projected_cross(a, b, point) / determinant
		var normal := (b - a).cross(c - a).normalized()
		if normal.y < 0.0:
			normal = -normal
		return {
			"height": float(a.y) + u * (float(b.y) - float(a.y)) + v * (float(c.y) - float(a.y)),
			"normal": normal,
			"road_cutout": false
		}
	return {}


func sample(p: Vector3) -> Dictionary:
	var result := sample_mesh(p)
	if not result.is_empty():
		return result
	var q := Vector2(clampf(p.x, bounds.x0, bounds.x1), clampf(p.z, bounds.z0, bounds.z1))
	var inside_road := false
	for ring in road_rings:
		if ring_contains(q, ring):
			inside_road = not inside_road
	if inside_road:
		return {"road_cutout": true}
	return {"error": "Offroad mesh has an uncovered point", "position": [q.x, q.y]}


func surface_tool() -> SurfaceTool:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for t in triangles:
		var a := vertices[int(t[0])]
		var b := vertices[int(t[1])]
		var c := vertices[int(t[2])]
		var order := [a, b, c] if (b - a).cross(c - a).y < 0.0 else [a, c, b]
		for p in order:
			st.set_uv(Vector2(p.x, p.z))
			st.add_vertex(p)
	return st


static func ring_contains(point: Vector2, ring: PackedVector2Array) -> bool:
	if ring.size() < 3:
		return false
	# Scalar arithmetic uses float64 even with Godot's float32 Vector2 storage.
	# A horizontal ray owns the lower endpoint of each edge, never both ends.
	# This avoids counting a shared vertex twice on large, detailed road rings.
	var inside := false
	var px: float = point.x
	var py: float = point.y
	for i in ring.size():
		var a := ring[i]
		var b := ring[(i + 1) % ring.size()]
		var ax: float = a.x
		var ay: float = a.y
		var bx: float = b.x
		var by: float = b.y
		var cross_value := (bx - ax) * (py - ay) - (by - ay) * (px - ax)
		if (
			cross_value == 0.0
			and px >= minf(ax, bx)
			and px <= maxf(ax, bx)
			and py >= minf(ay, by)
			and py <= maxf(ay, by)
		):
			return true
		if (ay > py) != (by > py):
			var crossing_x := ax + (py - ay) * (bx - ax) / (by - ay)
			if px < crossing_x:
				inside = not inside
	return inside
