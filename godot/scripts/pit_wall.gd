extends Node3D
## Historical wall profile with solid collision matching its triangulated shell.
## Width remains estimated. Motorcycle contact response is integrated separately.

const COLLISION_LAYER := 1 << 4
const QUAD_TRIANGLES := [0, 1, 2, 0, 2, 3]

var initialization_error := ""


func build(track: Node3D, profile: Dictionary) -> void:
	if profile.get("track_sha256", "") != FileAccess.get_sha256("res://data/track.json"):
		initialization_error = "Pit wall track provenance mismatch"
		return
	if profile.get("origin", {}) != track.data.origin:
		initialization_error = "Pit wall coordinate origin differs from track"
		return
	var rows: Array = profile.get("rows", [])
	var width: float = profile.get("visual_width_m", 0.0)
	if rows.size() < 2 or not is_finite(width) or width <= 0.0:
		initialization_error = "Invalid pit wall profile"
		return
	var rings: Array = []
	var stations: Array[float] = [0.0]
	for row: Dictionary in rows:
		var face: Array = row.get("face_xz_m", [])
		var inward: Array = row.get("road_inward_xz", [])
		var height: float = row.get("height_m", 0.0)
		if face.size() != 2 or inward.size() != 2 or not is_finite(height) or height <= 0:
			initialization_error = "Invalid pit wall section"
			return
		var p := Vector3(float(face[0]), 0.0, float(face[1]))
		var normal := Vector3(float(inward[0]), 0.0, float(inward[1]))
		if not p.is_finite() or not normal.is_finite() or normal.length_squared() < 0.5:
			initialization_error = "Invalid pit wall face coordinates"
			return
		normal = normal.normalized()
		p.y = track.terrain_surface_height(p)
		var back := p - normal * width
		back.y = track.terrain_surface_height(back)
		if not is_finite(p.y) or not is_finite(back.y):
			initialization_error = "Pit wall has no ground support"
			return
		# Both faces remain vertical and share one top elevation at each section.
		var top := p.y + height
		if back.y >= top:
			initialization_error = "Pit wall top lies below its back support"
			return
		var ring := [p, Vector3(p.x, top, p.z), Vector3(back.x, top, back.z), back]
		if not rings.is_empty():
			var previous: Vector3 = rings[-1][0]
			var distance := Vector2(p.x - previous.x, p.z - previous.z).length()
			if distance < 0.001:
				initialization_error = "Coincident pit wall sections"
				return
			stations.append(stations[-1] + distance)
		rings.append(ring)
	var collision := _build_collision(rings)
	if collision == null:
		return
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	surface.set_smooth_group(-1)
	for i in rings.size() - 1:
		for side in 4:
			var next := (side + 1) % 4
			var a: Vector3 = rings[i][side]
			var b: Vector3 = rings[i][next]
			var c: Vector3 = rings[i + 1][next]
			var d: Vector3 = rings[i + 1][side]
			_quad(
				surface,
				[a, d, c, b],
				[
					Vector2(stations[i], 0),
					Vector2(stations[i + 1], 0),
					Vector2(stations[i + 1], d.distance_to(c)),
					Vector2(stations[i], a.distance_to(b))
				]
			)
	for end in [0, rings.size() - 1]:
		var cap: Array = rings[end].duplicate()
		if end > 0:
			cap.reverse()
		_quad(surface, cap, [Vector2.ZERO, Vector2.UP, Vector2.ONE, Vector2.RIGHT])
	surface.generate_normals()
	surface.generate_tangents()
	var material := ShaderMaterial.new()
	material.shader = preload("res://shaders/painted_concrete.gdshader")
	material.set_shader_parameter("paint_tint", Color("deddd0"))
	material.set_shader_parameter("wear_amount", 0.08)
	material.set_shader_parameter("use_mesh_uv", true)
	for pair in [
		["concrete_color", "diff"], ["concrete_normal", "nor_gl"], ["concrete_roughness", "rough"]
	]:
		material.set_shader_parameter(
			pair[0], load("res://assets/materials/rough_concrete_%s_1k.jpg" % pair[1])
		)
	material.set_shader_parameter("concrete_detail", true)
	var instance := MeshInstance3D.new()
	instance.name = "ObservedPitDivider"
	instance.mesh = surface.commit()
	instance.material_override = material
	add_child(instance)
	add_child(collision)


func _quad(surface: SurfaceTool, points: Array, uv: Array) -> void:
	for corner in QUAD_TRIANGLES:
		surface.set_color(Color.WHITE)
		surface.set_uv(uv[corner])
		surface.add_vertex(points[corner])


func _build_collision(rings: Array) -> StaticBody3D:
	var pieces: Array = []
	for section in rings.size() - 1:
		var first: Array = rings[section]
		var second: Array = rings[section + 1]
		var footprint := PackedVector2Array()
		for p: Vector3 in [first[0], second[0], second[3], first[3]]:
			footprint.append(Vector2(p.x, p.z))
		var orientation := 0.0
		for i in 4:
			var cross_value := (footprint[(i + 1) % 4] - footprint[i]).cross(
				footprint[(i + 2) % 4] - footprint[(i + 1) % 4]
			)
			if absf(cross_value) < 0.00000001 or (i > 0 and cross_value * orientation <= 0):
				initialization_error = "Pit wall section has a folded or degenerate footprint"
				return null
			orientation = cross_value
		var center := Vector3.ZERO
		for p: Vector3 in first + second:
			center += p * 0.125
		var faces: Array = []
		for side in 4:
			var next := (side + 1) % 4
			faces.append([first[side], second[side], second[next], first[next]])
		faces.append(first)
		var end_cap: Array = second.duplicate()
		end_cap.reverse()
		faces.append(end_cap)
		for face: Array in faces:
			for triangle in [0, 3]:
				var a: Vector3 = face[QUAD_TRIANGLES[triangle]]
				var b: Vector3 = face[QUAD_TRIANGLES[triangle + 1]]
				var c: Vector3 = face[QUAD_TRIANGLES[triangle + 2]]
				# Godot front faces wind clockwise. The conventional cross product
				# points inward, so this is positive for a center in the shell kernel.
				var six_volume := (b - a).cross(c - a).dot(center - a)
				if not is_finite(six_volume) or six_volume <= 0.000000001:
					initialization_error = "Pit wall section cannot form a solid collision volume"
					return null
				pieces.append(
					{
						"center": center,
						"points":
						PackedVector3Array([Vector3.ZERO, a - center, b - center, c - center])
					}
				)
	var body := StaticBody3D.new()
	body.name = "PitDividerCollision"
	body.collision_layer = COLLISION_LAYER
	body.collision_mask = 0
	body.set_meta("obstacle_kind", "pit_wall")
	for piece: Dictionary in pieces:
		var shape := ConvexPolygonShape3D.new()
		shape.margin = 0.0
		shape.points = piece.points
		var instance := CollisionShape3D.new()
		instance.shape = shape
		instance.position = piece.center
		body.add_child(instance)
	return body
