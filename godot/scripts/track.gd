class_name ThunderhillTrack
extends Node3D

var data: Dictionary
var samples: Array
var terrain: Dictionary
var points: PackedVector3Array
var length_m: float
var grid: Dictionary = {}
var candidate_cache: Dictionary = {}
var terrain_material: ShaderMaterial
const CELL: float = 25.0
const ROAD_LIFT: float = 0.04
const SHOULDER_WIDTH: float = 6.0


func _ready() -> void:
	data = JSON.parse_string(FileAccess.get_file_as_string("res://data/track.json"))
	samples = data.samples
	assert(samples.size() >= 3, "A track requires at least three samples")
	length_m = data.length_m
	assert(length_m > 0.0, "Track length must be positive")
	terrain = JSON.parse_string(FileAccess.get_file_as_string("res://data/terrain.json"))
	for i in samples.size():
		var p: Array = samples[i].p
		points.append(Vector3(p[0], p[1], p[2]))
		var cell := Vector2i(floori(p[0] / CELL), floori(p[2] / CELL))
		if not grid.has(cell):
			grid[cell] = []
		grid[cell].append(i)
	_build_terrain()
	_build_road()


func material(color: Color, roughness: float = 0.8) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = roughness
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	return m


func _mesh(st: SurfaceTool, mat: Material, label: String) -> void:
	st.generate_normals()
	st.generate_tangents()
	var instance := MeshInstance3D.new()
	instance.name = label
	instance.mesh = st.commit()
	instance.material_override = mat
	add_child(instance)


func _vertex(st: SurfaceTool, p: Vector3, uv: Vector2, color: Color = Color.WHITE) -> void:
	st.set_color(color)
	st.set_uv(uv)
	st.add_vertex(p)


func _quad(
	st: SurfaceTool,
	a: Vector3,
	b: Vector3,
	c: Vector3,
	d: Vector3,
	u0: Vector2,
	u1: Vector2,
	u2: Vector2,
	u3: Vector2,
	color: Color = Color.WHITE
) -> void:
	# Godot front faces use clockwise winding. Ensure every ground ribbon faces up.
	var vertices := [[a, u0], [b, u1], [c, u2], [a, u0], [c, u2], [d, u3]]
	if (b - a).cross(c - a).y > 0.0:
		vertices = [[a, u0], [d, u3], [c, u2], [a, u0], [c, u2], [b, u1]]
	for v in vertices:
		_vertex(st, v[0], v[1], color)


func _build_terrain() -> void:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var nx: int = terrain.nx
	var nz: int = terrain.nz
	var spacing: float = terrain.step
	var verts := PackedVector3Array()
	for z in nz:
		for x in nx:
			var p := Vector3(
				terrain.x0 + x * spacing, terrain.heights[z * nx + x], terrain.z0 + z * spacing
			)
			verts.append(p)
	for z in nz - 1:
		for x in nx - 1:
			var a := z * nx + x
			_quad(
				st,
				verts[a],
				verts[a + 1],
				verts[a + nx + 1],
				verts[a + nx],
				Vector2(x, z),
				Vector2(x + 1, z),
				Vector2(x + 1, z + 1),
				Vector2(x, z + 1)
			)
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/terrain.gdshader")
	mat.set_shader_parameter(
		"grass_color", load("res://assets/materials/withered_grass_diff_1k.jpg")
	)
	mat.set_shader_parameter("soil_color", load("res://assets/materials/brown_mud_dry_diff_1k.jpg"))
	mat.set_shader_parameter(
		"grass_normal", load("res://assets/materials/withered_grass_nor_gl_1k.jpg")
	)
	terrain_material = mat
	_mesh(st, mat, "MeasuredTerrain")


func edge_point(i: int, offset: float, lift: float = 0.04) -> Vector3:
	var next: int = (i + 1) % points.size()
	var prev: int = posmod(i - 1, points.size())
	var tangent := (points[next] - points[prev]).normalized()
	var left := Vector3(tangent.z, 0, -tangent.x).normalized()
	return points[i] + left * offset + Vector3.UP * (tan(float(samples[i].bank)) * offset + lift)


func _build_road() -> void:
	var road := SurfaceTool.new()
	var paint := SurfaceTool.new()
	var curb := SurfaceTool.new()
	var shoulder := SurfaceTool.new()
	road.begin(Mesh.PRIMITIVE_TRIANGLES)
	paint.begin(Mesh.PRIMITIVE_TRIANGLES)
	curb.begin(Mesh.PRIMITIVE_TRIANGLES)
	shoulder.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in points.size():
		var j: int = (i + 1) % points.size()
		var w: float = samples[i].width * 0.5
		var wj: float = samples[j].width * 0.5
		var s: float = samples[i].s
		var sj: float = samples[j].s if j > 0 else length_m
		_quad(
			road,
			edge_point(i, -w),
			edge_point(i, w),
			edge_point(j, wj),
			edge_point(j, -wj),
			Vector2(0, s),
			Vector2(w * 2, s),
			Vector2(wj * 2, sj),
			Vector2(0, sj)
		)
		for side in [-1.0, 1.0]:
			var offsets := [0.0, 1.0, 2.0, 4.0, SHOULDER_WIDTH]
			for ring in offsets.size() - 1:
				var a := edge_point(i, side * (w + offsets[ring]), 0.0)
				var b := edge_point(i, side * (w + offsets[ring + 1]), 0.0)
				var c := edge_point(j, side * (wj + offsets[ring + 1]), 0.0)
				var d := edge_point(j, side * (wj + offsets[ring]), 0.0)
				a.y = _surface_height(a, _road_sample(a), false)
				b.y = _surface_height(b, _road_sample(b), false)
				c.y = _surface_height(c, _road_sample(c), false)
				d.y = _surface_height(d, _road_sample(d), false)
				_quad(
					shoulder,
					a,
					b,
					c,
					d,
					Vector2(a.x, a.z),
					Vector2(b.x, b.z),
					Vector2(c.x, c.z),
					Vector2(d.x, d.z)
				)
			_quad(
				paint,
				edge_point(i, side * (w - 0.18), 0.047),
				edge_point(i, side * (w - 0.06), 0.047),
				edge_point(j, side * (wj - 0.06), 0.047),
				edge_point(j, side * (wj - 0.18), 0.047),
				Vector2.ZERO,
				Vector2.RIGHT,
				Vector2.ONE,
				Vector2.DOWN
			)
			# Provisional curb placement follows curvature. Real profile and placement remain editable.
			if (
				absf(float(samples[i].curvature)) > 0.012
				and side * float(samples[i].curvature) > 0.0
			):
				var col := Color("1760a2") if int(s / 2.5) % 2 == 0 else Color("e5e5dc")
				_quad(
					curb,
					edge_point(i, side * w, 0.05),
					edge_point(i, side * (w + 0.9), 0.11),
					edge_point(j, side * (wj + 0.9), 0.11),
					edge_point(j, side * wj, 0.05),
					Vector2.ZERO,
					Vector2.RIGHT,
					Vector2.ONE,
					Vector2.DOWN,
					col
				)
	var asphalt := ShaderMaterial.new()
	asphalt.shader = load("res://shaders/asphalt.gdshader")
	for pair in [["color_map", "Color"], ["normal_map", "NormalGL"], ["rough_map", "Roughness"]]:
		asphalt.set_shader_parameter(
			pair[0], load("res://assets/materials/Asphalt010_1K-JPG_%s.jpg" % pair[1])
		)
	_mesh(road, asphalt, "RacingSurface")
	_mesh(shoulder, terrain_material, "RoadShoulders")
	_mesh(paint, material(Color("e8e3ce")), "EdgePaint")
	var curb_mat := material(Color.WHITE)
	curb_mat.vertex_color_use_as_albedo = true
	curb_mat.vertex_color_is_srgb = true
	_mesh(curb, curb_mat, "ProvisionalCurbs")


func terrain_height(p: Vector3) -> float:
	assert(not terrain.is_empty(), "Terrain must be loaded before sampling")
	var nx: int = terrain.nx
	var nz: int = terrain.nz
	var gx := clampf((p.x - float(terrain.x0)) / float(terrain.step), 0.0, float(nx - 1))
	var gz := clampf((p.z - float(terrain.z0)) / float(terrain.step), 0.0, float(nz - 1))
	var x := mini(floori(gx), nx - 2)
	var z := mini(floori(gz), nz - 2)
	var fx := gx - float(x)
	var fz := gz - float(z)
	var top := lerpf(terrain.heights[z * nx + x], terrain.heights[z * nx + x + 1], fx)
	var bottom := lerpf(
		terrain.heights[(z + 1) * nx + x], terrain.heights[(z + 1) * nx + x + 1], fx
	)
	return lerpf(top, bottom, fz)


func terrain_normal(p: Vector3) -> Vector3:
	const DELTA: float = 0.15
	var dx := (
		(terrain_height(p + Vector3.RIGHT * DELTA) - terrain_height(p - Vector3.RIGHT * DELTA))
		/ (2.0 * DELTA)
	)
	var dz := (
		(terrain_height(p + Vector3.BACK * DELTA) - terrain_height(p - Vector3.BACK * DELTA))
		/ (2.0 * DELTA)
	)
	return Vector3(-dx, 1.0, -dz).normalized()


func _surface_height(p: Vector3, road: Dictionary, include_curb: bool = true) -> float:
	var delta: Vector3 = p - road.center
	var lateral: float = delta.dot(road.left)
	var along: Vector3 = Vector3(road.tangent.x, 0.0, road.tangent.z).normalized()
	var grade: float = (
		road.tangent.y / maxf(Vector2(road.tangent.x, road.tangent.z).length(), 0.001)
	)
	var plane: float = (
		road.center.y + tan(road.bank) * lateral + delta.dot(along) * grade + ROAD_LIFT
	)
	var edge: float = Vector2(delta.x, delta.z).length() - road.width * 0.5
	if edge <= 0.0:
		return plane
	if (
		include_curb
		and edge <= 0.9
		and absf(road.curvature) > 0.012
		and lateral * road.curvature > 0.0
	):
		return plane + lerpf(0.01, 0.07, edge / 0.9)
	if edge < SHOULDER_WIDTH:
		return lerpf(plane, terrain_height(p), smoothstep(0.0, SHOULDER_WIDTH, edge))
	return terrain_height(p)


# Shared visible surface for scenery placement, including the finite road shoulder.
func terrain_surface_height(p: Vector3) -> float:
	return _surface_height(p, _road_sample(p))


func sample_world(p: Vector3) -> Dictionary:
	var road := _road_sample(p)
	road.height = _surface_height(p, road)
	var edge: float = road.planar_distance - road.width * 0.5
	if edge >= SHOULDER_WIDTH:
		road.normal = terrain_normal(p)
	elif edge > 0.0:
		const DELTA: float = 0.10
		var dx := (
			(
				_surface_height(p + Vector3.RIGHT * DELTA, road)
				- _surface_height(p - Vector3.RIGHT * DELTA, road)
			)
			/ (2.0 * DELTA)
		)
		var dz := (
			(
				_surface_height(p + Vector3.BACK * DELTA, road)
				- _surface_height(p - Vector3.BACK * DELTA, road)
			)
			/ (2.0 * DELTA)
		)
		road.normal = Vector3(-dx, 1.0, -dz).normalized()
	return road


func _road_sample(p: Vector3) -> Dictionary:
	assert(points.size() >= 3 and length_m > 0.0, "Track must be initialized before sampling")
	var cell := Vector2i(floori(p.x / CELL), floori(p.z / CELL))
	var candidates: Array = candidate_cache.get(cell, [])
	if candidates.is_empty():
		for z in range(-1, 2):
			for x in range(-1, 2):
				candidates.append_array(grid.get(cell + Vector2i(x, z), []))
		if not candidates.is_empty():
			candidate_cache[cell] = candidates
	if candidates.is_empty():
		for i in points.size():
			candidates.append(i)
	var best := INF
	var index := 0
	var fraction := 0.0
	for i in candidates:
		var a: Vector3 = points[i]
		var b: Vector3 = points[(i + 1) % points.size()]
		var ab := Vector2(b.x - a.x, b.z - a.z)
		var t := clampf(
			Vector2(p.x - a.x, p.z - a.z).dot(ab) / maxf(ab.length_squared(), 0.001), 0, 1
		)
		var q := a.lerp(b, t)
		var d := Vector2(p.x - q.x, p.z - q.z).length_squared()
		if d < best:
			best = d
			index = i
			fraction = t
	var j: int = (index + 1) % points.size()
	var center := points[index].lerp(points[j], fraction)
	var tangent := (points[j] - points[index]).normalized()
	var left := Vector3(tangent.z, 0, -tangent.x).normalized()
	var lateral := (p - center).dot(left)
	var bank := lerpf(samples[index].bank, samples[j].bank, fraction)
	var width := lerpf(samples[index].width, samples[j].width, fraction)
	var height := center.y + tan(bank) * lateral
	var normal := (
		(
			Vector3.UP
			- left * tan(bank)
			- (
				Vector3(tangent.x, 0, tangent.z).normalized()
				* (tangent.y / maxf(Vector2(tangent.x, tangent.z).length(), 0.001))
			)
		)
		. normalized()
	)
	return {
		"height": height,
		"normal": normal,
		"on_track": sqrt(best) <= width * 0.5,
		"distance": lateral,
		"planar_distance": sqrt(best),
		"progress":
		(
			fposmod(
				float(samples[index].s) + fraction * (points[j] - points[index]).length(), length_m
			)
			/ length_m
		),
		"tangent": tangent,
		"width": width,
		"index": index,
		"center": center,
		"left": left,
		"bank": bank,
		"curvature": float(samples[index].curvature)
	}
