class_name TriangleRibbon
extends RefCounted
## One triangle stream supplies ribbon rendering and contact. No boundary expansion.
const CELL := 16.0
const MAX_INDEX_CELLS_PER_QUAD := 65536
var vertices := PackedVector3Array()
var uvs := PackedVector2Array()
var triangles: Array[PackedInt32Array] = []
var grid: Dictionary = {}


func clear() -> void:
	vertices.clear()
	uvs.clear()
	triangles.clear()
	grid.clear()


static func projected_cross(a: Vector3, b: Vector3, c: Vector3) -> float:
	# Promote components before subtraction; Vector arithmetic uses float32.
	var ax: float = a.x
	var az: float = a.z
	return (float(b.x) - ax) * (float(c.z) - az) - (float(b.z) - az) * (float(c.x) - ax)


func add_quad(
	a: Vector3,
	b: Vector3,
	c: Vector3,
	d: Vector3,
	uv_a: Vector2,
	uv_b: Vector2,
	uv_c: Vector2,
	uv_d: Vector2
) -> String:
	var points := [a, b, c, d]
	var coords := [uv_a, uv_b, uv_c, uv_d]
	for p: Vector3 in points:
		if not p.is_finite():
			return "Nonfinite ribbon vertex"
	for uv: Vector2 in coords:
		if not uv.is_finite():
			return "Nonfinite ribbon UV"
	var first := projected_cross(a, b, c)
	var second := projected_cross(a, c, d)
	if first == 0.0 or second == 0.0 or signf(first) != signf(second):
		return "Degenerate or folded ribbon quad"
	# Same a,c diagonal and clockwise order as ThunderhillTrack._quad.
	var orders := [[0, 1, 2], [0, 2, 3]] if first > 0.0 else [[0, 3, 2], [0, 2, 1]]
	return _append_faces(points, coords, orders)


func add_triangle(
	a: Vector3, b: Vector3, c: Vector3, uv_a: Vector2, uv_b: Vector2, uv_c: Vector2
) -> String:
	for p: Vector3 in [a, b, c]:
		if not p.is_finite():
			return "Nonfinite ribbon vertex"
	for uv: Vector2 in [uv_a, uv_b, uv_c]:
		if not uv.is_finite():
			return "Nonfinite ribbon UV"
	var area := projected_cross(a, b, c)
	if area == 0.0:
		return "Degenerate ribbon triangle"
	return _append_faces([a, b, c], [uv_a, uv_b, uv_c], [[0, 1, 2]] if area > 0.0 else [[0, 2, 1]])


func _append_faces(points: Array, coords: Array, orders: Array) -> String:
	var cells: Array = []
	var count := 0
	for order: Array in orders:
		var p: Vector3 = points[order[0]]
		var q: Vector3 = points[order[1]]
		var r: Vector3 = points[order[2]]
		var x0 := floorf(minf(p.x, minf(q.x, r.x)) / CELL)
		var x1 := floorf(maxf(p.x, maxf(q.x, r.x)) / CELL)
		var z0 := floorf(minf(p.z, minf(q.z, r.z)) / CELL)
		var z1 := floorf(maxf(p.z, maxf(q.z, r.z)) / CELL)
		if maxf(absf(x0), maxf(absf(x1), maxf(absf(z0), absf(z1)))) > 2147483646.0:
			return "Ribbon spatial index coordinate exceeds integer range"
		var size: float = (x1 - x0 + 1.0) * (z1 - z0 + 1.0)
		if size > MAX_INDEX_CELLS_PER_QUAD - count:
			return "Ribbon quad exceeds spatial index budget"
		count += int(size)
		cells.append([int(x0), int(x1), int(z0), int(z1)])
	# Validate both triangles before mutating any owned state.
	var base := vertices.size()
	vertices.append_array(PackedVector3Array(points))
	uvs.append_array(PackedVector2Array(coords))
	for i in orders.size():
		var index := triangles.size()
		var order: Array = orders[i]
		triangles.append(PackedInt32Array([base + order[0], base + order[1], base + order[2]]))
		var bounds: Array = cells[i]
		for z in range(bounds[2], bounds[3] + 1):
			for x in range(bounds[0], bounds[1] + 1):
				var key := Vector2i(x, z)
				if not grid.has(key):
					grid[key] = []
				grid[key].append(index)
	return ""


func sample(p: Vector3) -> Dictionary:
	if not p.is_finite() or absf(p.x / CELL) > 2147483646.0 or absf(p.z / CELL) > 2147483646.0:
		return {}
	var result := {}
	for index: int in grid.get(Vector2i(floori(p.x / CELL), floori(p.z / CELL)), []):
		var t := triangles[index]
		var a := vertices[t[0]]
		var b := vertices[t[1]]
		var c := vertices[t[2]]
		# Oriented edge tests include boundaries without an epsilon halo.
		if (
			projected_cross(a, b, p) < 0.0
			or projected_cross(b, c, p) < 0.0
			or projected_cross(c, a, p) < 0.0
		):
			continue
		var determinant := projected_cross(a, b, c)
		var u := projected_cross(a, p, c) / determinant
		var v := projected_cross(a, b, p) / determinant
		var by: float = float(b.y) - float(a.y)
		var cy: float = float(c.y) - float(a.y)
		var height := float(a.y) + u * by + v * cy
		if not result.is_empty() and height <= float(result.height):
			continue
		var dx := (by * (float(c.z) - float(a.z)) - cy * (float(b.z) - float(a.z))) / determinant
		var dz := ((float(b.x) - float(a.x)) * cy - (float(c.x) - float(a.x)) * by) / determinant
		var scale := maxf(1.0, maxf(absf(dx), absf(dz)))
		var normal := Vector3(-dx / scale, 1.0 / scale, -dz / scale).normalized()
		result = {"height": height, "normal": normal, "triangle_id": index}
	return result


func surface_tool() -> SurfaceTool:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for t in triangles:
		for index in t:
			st.set_uv(uvs[index])
			st.add_vertex(vertices[index])
	return st
