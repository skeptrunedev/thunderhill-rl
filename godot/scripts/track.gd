class_name ThunderhillTrack
extends Node3D

var initialization_error := ""
var startup_observer := Callable()
var data: Dictionary
var samples: Array
var terrain: Dictionary
var points: PackedVector3Array
var length_m: float
var grid: Dictionary = {}
var candidate_cache: Dictionary = {}
var terrain_material: ShaderMaterial
var offroad_surface := preload("res://scripts/offroad_surface.gd").new()
var pavement_surface := preload("res://scripts/triangle_ribbon.gd").new()
var curb_surface := preload("res://scripts/curb_surface.gd").new()
const CELL: float = 25.0
const ROAD_LIFT: float = 0.04
const SHOULDER_WIDTH: float = 6.0
## Shared linear albedo multiplier for dry ground and standing vegetation.
## Artistic palette matching, not measured Thunderhill reflectance.
const DRY_GROUND_TINT := Vector3(0.65, 0.56, 0.43)


func _ready() -> void:
	data = JSON.parse_string(FileAccess.get_file_as_string("res://data/track.json"))
	samples = data.samples
	assert(samples.size() >= 3, "A track requires at least three samples")
	length_m = data.length_m
	assert(length_m > 0.0, "Track length must be positive")
	terrain = JSON.parse_string(FileAccess.get_file_as_string("res://data/terrain.json"))
	_startup_mark("track_sources_parsed")
	for i in samples.size():
		var p: Array = samples[i].p
		points.append(Vector3(p[0], p[1], p[2]))
		var cell := Vector2i(floori(p[0] / CELL), floori(p[2] / CELL))
		if not grid.has(cell):
			grid[cell] = []
		grid[cell].append(i)
	_startup_mark("track_index_complete")
	_build_terrain()
	if not initialization_error.is_empty():
		push_error(initialization_error)
		get_tree().quit(2)
		return
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


func validate_surface_sources(mesh_data: Dictionary) -> String:
	var metadata: Dictionary = mesh_data.get("metadata", {})
	for source in ["track", "terrain"]:
		if (
			metadata.get(source + "_sha256", "")
			!= FileAccess.get_sha256("res://data/" + source + ".json")
		):
			return "Offroad mesh " + source + " source mismatch"
	return ""


func _build_terrain() -> void:
	var mesh_data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/surface.json")
	)
	initialization_error = validate_surface_sources(mesh_data)
	if not initialization_error.is_empty():
		return
	_startup_mark("surface_sources_validated")
	offroad_surface.configure(mesh_data)
	_startup_mark("surface_contact_index_complete")
	var st := offroad_surface.surface_tool()
	_startup_mark("terrain_triangle_stream_complete")
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/terrain.gdshader")
	mat.set_shader_parameter("ground_tint", DRY_GROUND_TINT)
	mat.set_shader_parameter("grass_color", load("res://assets/materials/dry_cut_grass_v1.png"))
	mat.set_shader_parameter("soil_color", load("res://assets/materials/brown_mud_dry_diff_1k.jpg"))
	mat.set_shader_parameter(
		"soil_normal", load("res://assets/materials/brown_mud_dry_nor_gl_1k.jpg")
	)
	mat.set_shader_parameter(
		"soil_roughness", load("res://assets/materials/brown_mud_dry_rough_1k.jpg")
	)
	var macro: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://assets/materials/terrain_macro.json")
	)
	var detail: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://assets/materials/terrain_detail.json")
	)
	for source: String in ["track", "surface"]:
		if (
			detail.get(source + "_sha256", "")
			!= FileAccess.get_sha256("res://data/" + source + ".json")
		):
			initialization_error = "Terrain detail " + source + " source mismatch"
			return
	if (
		detail.get("local_origin_xz") != macro.local_origin_xz
		or detail.get("local_size_xz") != macro.local_size_xz
	):
		initialization_error = "Terrain detail and macro mapping mismatch"
		return
	mat.set_shader_parameter("terrain_detail", load("res://assets/materials/terrain_detail.png"))
	mat.set_shader_parameter("terrain_macro", load("res://assets/materials/terrain_macro.png"))
	mat.set_shader_parameter(
		"macro_origin", Vector2(macro.local_origin_xz[0], macro.local_origin_xz[1])
	)
	mat.set_shader_parameter("macro_size", Vector2(macro.local_size_xz[0], macro.local_size_xz[1]))
	terrain_material = mat
	_mesh(st, mat, "MeasuredTerrain")
	_startup_mark("terrain_render_mesh_complete")


func edge_point(i: int, offset: float, lift: float = 0.04) -> Vector3:
	var next: int = (i + 1) % points.size()
	var prev: int = posmod(i - 1, points.size())
	# Match the offline terrain builder: retain source coordinates in float64
	# through edge construction, then round once into the render vertex.
	var dx: float = float(samples[next].p[0]) - float(samples[prev].p[0])
	var dz: float = float(samples[next].p[2]) - float(samples[prev].p[2])
	var length := sqrt(dx * dx + dz * dz)
	return Vector3(
		float(samples[i].p[0]) + dz / length * offset,
		float(samples[i].p[1]) + tan(float(samples[i].bank)) * offset + lift,
		float(samples[i].p[2]) - dx / length * offset
	)


func _load_pavement() -> String:
	var mesh_data: Variant = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/pavement.json")
	)
	if not mesh_data is Dictionary or mesh_data.get("schema_version") != 1:
		return "Invalid pavement mesh schema"
	var metadata: Variant = mesh_data.get("metadata")
	if not metadata is Dictionary:
		return "Missing pavement mesh metadata"
	for source: String in ["track", "surface"]:
		if (
			metadata.get(source + "_sha256", "")
			!= FileAccess.get_sha256("res://data/" + source + ".json")
		):
			return "Pavement mesh " + source + " source mismatch"
	for key: String in ["vertices", "uvs", "triangles"]:
		if not mesh_data.get(key) is Array or mesh_data[key].is_empty():
			return "Missing pavement " + key
	if mesh_data.vertices.size() != mesh_data.uvs.size():
		return "Pavement UV count differs from vertices"
	var vertices := PackedVector3Array()
	var uvs := PackedVector2Array()
	for key: String in ["vertices", "uvs"]:
		for value: Variant in mesh_data[key]:
			if not value is Array or value.size() != (3 if key == "vertices" else 2):
				return "Malformed pavement " + key
			for component: Variant in value:
				if not (component is float or component is int) or not is_finite(float(component)):
					return "Nonfinite pavement component"
			if key == "vertices":
				vertices.append(Vector3(value[0], value[1], value[2]))
			else:
				uvs.append(Vector2(value[0], value[1]))
	var candidate := preload("res://scripts/triangle_ribbon.gd").new()
	for face: Variant in mesh_data.triangles:
		if not face is Array or face.size() != 3:
			return "Malformed pavement triangle"
		for index: Variant in face:
			if not (index is float or index is int) or not is_finite(float(index)):
				return "Invalid pavement vertex index"
			if float(index) != floorf(float(index)) or index < 0 or index >= vertices.size():
				return "Pavement vertex index out of range"
		var error: String = candidate.add_triangle(
			vertices[int(face[0])],
			vertices[int(face[1])],
			vertices[int(face[2])],
			uvs[int(face[0])],
			uvs[int(face[1])],
			uvs[int(face[2])]
		)
		if not error.is_empty():
			return error
	pavement_surface = candidate
	return ""


func _build_road() -> void:
	initialization_error = _load_pavement()
	if not initialization_error.is_empty():
		push_error(initialization_error)
		get_tree().quit(2)
		return
	var paint := SurfaceTool.new()
	curb_surface.clear()
	paint.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in points.size():
		var j: int = (i + 1) % points.size()
		var w: float = samples[i].width * 0.5
		var wj: float = samples[j].width * 0.5
		var s: float = samples[i].s
		var sj: float = samples[j].s if j > 0 else length_m
		for side in [-1.0, 1.0]:
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
				var error: String = curb_surface.add_quad(
					edge_point(i, side * w, 0.05),
					edge_point(i, side * (w + 0.9), 0.11),
					edge_point(j, side * (wj + 0.9), 0.11),
					edge_point(j, side * wj, 0.05),
					Vector2(0.0, s),
					Vector2(0.9, s),
					Vector2(0.9, sj),
					Vector2(0.0, sj)
				)
				if not error.is_empty():
					initialization_error = "Invalid curb segment %d: %s" % [i, error]
					push_error(initialization_error)
					get_tree().quit(2)
					return
	var asphalt := ShaderMaterial.new()
	asphalt.shader = load("res://shaders/asphalt.gdshader")
	for pair in [["color_map", "Color"], ["normal_map", "NormalGL"], ["rough_map", "Roughness"]]:
		asphalt.set_shader_parameter(
			pair[0], load("res://assets/materials/Asphalt010_1K-JPG_%s.jpg" % pair[1])
		)
	_mesh(pavement_surface.surface_tool(), asphalt, "RacingSurface")
	var paint_mat := ShaderMaterial.new()
	paint_mat.shader = preload("res://shaders/painted_concrete.gdshader")
	paint_mat.set_shader_parameter("paint_tint", Color("e8e3ce"))
	paint_mat.set_shader_parameter("wear_amount", 0.12)
	_mesh(paint, paint_mat, "EdgePaint")
	var curb_mat := ShaderMaterial.new()
	curb_mat.shader = preload("res://shaders/painted_concrete.gdshader")
	curb_mat.set_shader_parameter("wear_amount", 0.28)
	curb_mat.set_shader_parameter("curb_stripes", true)
	curb_mat.set_shader_parameter("use_mesh_uv", true)
	for pair in [
		["concrete_color", "diff"], ["concrete_normal", "nor_gl"], ["concrete_roughness", "rough"]
	]:
		curb_mat.set_shader_parameter(
			pair[0], load("res://assets/materials/rough_concrete_%s_1k.jpg" % pair[1])
		)
	curb_mat.set_shader_parameter("concrete_detail", true)
	_mesh(curb_surface.surface_tool(), curb_mat, "ProvisionalCurbs")


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
	var a: float = terrain.heights[z * nx + x]
	var b: float = terrain.heights[z * nx + x + 1]
	var c: float = terrain.heights[(z + 1) * nx + x + 1]
	var d: float = terrain.heights[(z + 1) * nx + x]
	# Match _build_terrain's a,b,c / a,c,d triangle planes. Bilinear
	# interpolation invents a curved surface between noncoplanar corners.
	if fx >= fz:
		return a + (b - a) * fx + (c - b) * fz
	return a + (c - d) * fx + (d - a) * fz


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


func _surface_sample(p: Vector3, road: Dictionary, include_curb: bool = true) -> Dictionary:
	var surface: Dictionary = pavement_surface.sample(p)
	if not surface.is_empty():
		surface.on_pavement = true
	if include_curb:
		var curb: Dictionary = curb_surface.sample(p)
		if (
			not curb.is_empty()
			and (surface.is_empty() or float(curb.height) > float(surface.height))
		):
			surface = curb
	# Avoid polygon cutout classification when a rendered ribbon already covers
	# the point. Mesh triangles alone determine any overlying terrain contact.
	var ground: Dictionary = offroad_surface.sample_mesh(p)
	if (
		not ground.is_empty()
		and (surface.is_empty() or float(ground.height) > float(surface.height))
	):
		surface = ground
	if not surface.is_empty():
		return surface
	# No analytic plane may fill a hole in the rendered geometry.
	push_error("No rendered ground surface covers point %s" % p)
	return {"height": NAN, "normal": road.normal}


func _surface_height(p: Vector3, road: Dictionary, include_curb: bool = true) -> float:
	return float(_surface_sample(p, road, include_curb).height)


# Shared visible surface for scenery placement, including the finite road shoulder.
func terrain_surface_height(p: Vector3) -> float:
	return _surface_height(p, _road_sample(p))


func sample_world(p: Vector3) -> Dictionary:
	var road := _road_sample(p)
	# Height and normal come from the same triangle search. This matters for
	# both contact consistency and the many scenery placement queries at startup.
	var surface := _surface_sample(p, road)
	road.height = surface.height
	road.normal = surface.normal
	road.on_pavement = surface.get("on_pavement", false)
	road.on_curb = surface.get("on_curb", false)
	if road.on_curb:
		road.curb_triangle_id = surface.triangle_id
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


func _startup_mark(stage: String) -> void:
	if startup_observer.is_valid():
		startup_observer.call(stage)
