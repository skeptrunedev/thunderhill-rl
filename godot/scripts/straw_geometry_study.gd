extends RefCounted
## Original visual experiment only. No collision, training, or production changes.
const SEED := 629137
const CLUMPS := 9000
const CELL := 16.0
const LYING_STEMS := 24
const STUBBLE_STEMS := 4


static func setup(game: Node) -> Dictionary:
	if game.has_node("StrawGeometryStudy"):
		return {"error": "StrawGeometryStudy already installed"}
	var random := RandomNumberGenerator.new()
	random.seed = SEED
	var track: Node = game.track
	var segments: Array[int] = []
	for i in track.samples.size() - 1:
		if float(track.samples[i].s) >= 900.0 and float(track.samples[i].s) <= 1250.0:
			segments.append(i)
	if segments.is_empty():
		return {"error": "No track samples in study interval"}
	var patches := {}
	var accepted := 0
	var minimum_clearance := INF
	for attempt in CLUMPS * 4:
		var i: int = segments[random.randi_range(0, segments.size() - 1)]
		var tangent: Vector3 = (track.points[i + 1] - track.points[i]).normalized()
		var left := Vector3(tangent.z, 0, -tangent.x).normalized()
		var p: Vector3 = track.points[i].lerp(track.points[i + 1], random.randf())
		# Positive curvature in Turn 2 places its inside on the left.
		p += left * (float(track.samples[i].width) * 0.5 + random.randf_range(1.8, 17.0))
		var contact: Dictionary = track.sample_world(p)
		var clearance := absf(float(contact.distance)) - float(contact.width) * 0.5
		if clearance < 1.5 or contact.on_pavement or contact.on_curb:
			continue
		if not is_finite(float(contact.height)):
			return {"error": "Nonfinite terrain sample"}
		p.y = float(contact.height) + 0.004
		var cell := Vector2i(floori(p.x / CELL), floori(p.z / CELL))
		if not patches.has(cell):
			patches[cell] = []
		patches[cell].append({"p": p, "normal": contact.normal, "yaw": random.randf() * TAU})
		minimum_clearance = minf(minimum_clearance, clearance)
		accepted += 1
		if accepted == CLUMPS:
			break
	var study := Node3D.new()
	study.name = "StrawGeometryStudy"
	game.add_child(study)
	var meshes: Array[ArrayMesh] = []
	for variant in 3:
		meshes.append(_mesh(random))
	for cell: Vector2i in patches:
		var rows: Array = patches[cell]
		var origin: Vector3 = rows[0].p
		var multi := MultiMesh.new()
		multi.transform_format = MultiMesh.TRANSFORM_3D
		multi.mesh = meshes[absi(cell.x * 73 + cell.y * 131) % meshes.size()]
		multi.instance_count = rows.size()
		for i in rows.size():
			var basis := (
				Basis(Quaternion(Vector3.UP, rows[i].normal)) * Basis(Vector3.UP, rows[i].yaw)
			)
			multi.set_instance_transform(i, Transform3D(basis, rows[i].p - origin))
		var instance := MultiMeshInstance3D.new()
		instance.name = "FlattenedStraw_%d_%d" % [cell.x, cell.y]
		instance.position = origin
		instance.multimesh = multi
		instance.visibility_range_end = 65.0
		instance.visibility_range_end_margin = 10.0
		instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
		study.add_child(instance)
	var report := {
		"seed": SEED,
		"clumps": accepted,
		"patches": patches.size(),
		"triangles_per_clump": (LYING_STEMS + STUBBLE_STEMS) * 4,
		"total_triangles": accepted * (LYING_STEMS + STUBBLE_STEMS) * 4,
		"station_interval_m": [900, 1250],
		"minimum_center_road_clearance_m": minimum_clearance,
		"maximum_clump_radius_m": 0.77,
		"standing_grass_preserved": true,
		"casts_shadows": true,
		"fragment_length_m": [0.055, 0.19],
		"fragment_full_width_m": [0.002, 0.005],
		"limitation": "Artistic flattened straw study; stem density and dimensions are not surveyed"
	}
	study.set_meta("study_report", report)
	return report


static func _mesh(random: RandomNumberGenerator) -> ArrayMesh:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	surface.set_smooth_group(-1)
	for stem in LYING_STEMS + STUBBLE_STEMS:
		var angle := random.randf() * TAU
		var radial := sqrt(random.randf()) * 0.38
		var base := Vector3(cos(angle) * radial, 0.0, sin(angle) * radial)
		var heading := random.randf() * TAU
		var direction := Vector3(cos(heading), 0, sin(heading))
		var across := Vector3(-direction.z, 0, direction.x)
		var length_m := random.randf_range(0.055, 0.19)
		var width_m := random.randf_range(0.001, 0.0025)
		var middle := (
			base
			+ direction * length_m * 0.48
			+ across * random.randf_range(-0.012, 0.012)
			+ Vector3.UP * random.randf_range(0.004, 0.014)
		)
		var tip := base + direction * length_m + Vector3.UP * random.randf_range(0.001, 0.006)
		if stem >= LYING_STEMS:
			middle = base + direction * 0.025 + Vector3.UP * 0.02
			tip = base + direction * 0.045 + Vector3.UP * random.randf_range(0.035, 0.065)
			width_m *= 0.65
		var color := Color("62503a").lerp(Color("a28b64"), random.randf())
		surface.set_color(color)
		var vertices := [
			base - across * width_m,
			base + across * width_m,
			middle - across * width_m * 0.8,
			middle + across * width_m * 0.8,
			tip - across * width_m * 0.2,
			tip + across * width_m * random.randf_range(0.05, 0.45)
		]
		for index in [0, 2, 1, 1, 2, 3, 2, 4, 3, 3, 4, 5]:
			surface.add_vertex(vertices[index])
	surface.generate_normals()
	var material := StandardMaterial3D.new()
	material.vertex_color_use_as_albedo = true
	material.vertex_color_is_srgb = true
	material.roughness = 1.0
	material.metallic_specular = 0.0
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	surface.set_material(material)
	return surface.commit()
