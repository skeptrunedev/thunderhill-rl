extends Node3D
## Historical visual wall profile. Width is estimated and collision is not modeled.

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
	var material := ShaderMaterial.new()
	material.shader = preload("res://shaders/painted_concrete.gdshader")
	material.set_shader_parameter("paint_tint", Color("deddd0"))
	material.set_shader_parameter("wear_amount", 0.08)
	material.set_shader_parameter("use_mesh_uv", true)
	var instance := MeshInstance3D.new()
	instance.name = "ObservedPitDivider"
	instance.mesh = surface.commit()
	instance.material_override = material
	add_child(instance)


func _quad(surface: SurfaceTool, points: Array, uv: Array) -> void:
	for corner in [0, 1, 2, 0, 2, 3]:
		surface.set_color(Color.WHITE)
		surface.set_uv(uv[corner])
		surface.add_vertex(points[corner])
