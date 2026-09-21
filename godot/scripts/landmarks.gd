class_name TrackLandmarks
extends Node3D
## Mapped historical footprints with explicitly estimated architectural massing.
## Positions derive from OSM and inspected NAIP crowns, never random scattering.

var _track: Node3D
var _ground_height_cache: Dictionary = {}
var initialization_error := ""
var _walls := SurfaceTool.new()
var _roofs := SurfaceTool.new()
var _glass := SurfaceTool.new()
var _metal := SurfaceTool.new()
var _concrete := SurfaceTool.new()
var _paving := SurfaceTool.new()


func build(track: Node3D, include_divider := true) -> void:
	_track = track
	_ground_height_cache.clear()
	var data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/landmarks.json")
	)
	var divider_data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/pit-wall.json")
	)
	if include_divider:
		_build_divider(track, divider_data)
		if not initialization_error.is_empty():
			return
	for surface in [_walls, _roofs, _glass, _metal, _concrete, _paving]:
		surface.begin(Mesh.PRIMITIVE_TRIANGLES)
		if surface != _paving:
			surface.set_smooth_group(-1)
	for area in data.paving:
		_build_paving(area)
	var clubhouse: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/clubhouse-profile.json")
	)
	if (
		clubhouse.get("source_landmarks_sha256", "")
		!= FileAccess.get_sha256("res://data/landmarks.json")
	):
		initialization_error = "Clubhouse profile mapped footprint source mismatch"
		return
	for building in data.buildings:
		if building.osm_way == clubhouse.osm_way:
			_build_clubhouse(building, clubhouse)
		else:
			_build_building(building)
	for fence in data.fences:
		if fence.osm_way == divider_data.replaces_osm_way:
			continue
		_build_fence(fence)
	for post in data.marshal_posts:
		_build_marshal(post)
	_commit(_walls, _material(Color("c5c1ae")), "MappedBuildingWalls")
	_commit(_roofs, _material(Color("aeb8b8")), "MappedMetalRoofs")
	_commit(_glass, _material(Color("334951"), 0.35), "EstimatedFacadePanels")
	_commit(_metal, _material(Color("6a7374")), "MappedFencesAndSupports")
	_commit(_concrete, _material(Color("deddd0")), "VideoReferencedPitWall")
	_commit(_paving, _material(Color("373d3d")), "MappedPaddockPaving")
	_build_trees(data.trees)
	_ground_height_cache.clear()


func build_prepared(track: Node3D) -> void:
	initialization_error = validate_bake()
	if not initialization_error.is_empty():
		return
	# Keep the wall's existing provenance checks and solid collision construction.
	# Only decorative static landmarks are loaded from the prepared scene.
	var profile: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/pit-wall.json")
	)
	_build_divider(track, profile)
	if not initialization_error.is_empty():
		return
	var packed := load("res://assets/generated/landmarks.scn") as PackedScene
	if packed == null:
		initialization_error = "Cannot load prepared landmarks"
		return
	add_child(packed.instantiate())


func _build_divider(track: Node3D, profile: Dictionary) -> void:
	var divider := preload("res://scripts/pit_wall.gd").new()
	add_child(divider)
	divider.build(track, profile)
	initialization_error = divider.initialization_error


static func validate_bake() -> String:
	var manifest: Variant = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/landmarks-bake.json")
	)
	if (
		not manifest is Dictionary
		or manifest.get("schema_version") != 1
		or not manifest.get("sources") is Dictionary
		or manifest.sources.is_empty()
	):
		return "Missing or invalid landmark bake manifest"
	for source: String in manifest.sources:
		# Packaging checks all sources before export converts scripts and textures.
		if OS.has_feature("editor") or source.begins_with("data/"):
			if FileAccess.get_sha256("res://" + source) != manifest.sources[source]:
				return "Stale landmark bake: " + source + ". Run res://tools/bake_landmarks.gd."
	return ""


func _material(color: Color, roughness: float = 0.85) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = roughness
	mat.metallic_specular = 0.12
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	return mat


func _commit(surface: SurfaceTool, mat: Material, label: String) -> void:
	var vertices: Variant = surface.commit_to_arrays()[Mesh.ARRAY_VERTEX]
	# Godot represents an unused surface's vertex slot as null.
	if vertices == null or vertices.is_empty():
		return
	surface.generate_normals()
	var instance := MeshInstance3D.new()
	instance.name = label
	instance.mesh = surface.commit()
	instance.material_override = mat
	add_child(instance)


func _point(p: Array, lift: float = 0.0) -> Vector3:
	var result := Vector3(float(p[0]), 0, float(p[1]))
	# Subdivided pavement triangles share most vertices. The track is immutable
	# during this build, so reuse the exact sampled height rather than resampling.
	var key := Vector2(result.x, result.z)
	if not _ground_height_cache.has(key):
		_ground_height_cache[key] = _track.terrain_surface_height(result)
	result.y = float(_ground_height_cache[key]) + lift
	return result


func _tri(surface: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, upward: bool = false) -> void:
	if upward and (b - a).cross(c - a).y > 0:
		var swap := b
		b = c
		c = swap
	for p in [a, b, c]:
		surface.add_vertex(p)


func _quad(surface: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, d: Vector3) -> void:
	_tri(surface, a, b, c)
	_tri(surface, a, c, d)


func _beam(surface: SurfaceTool, a: Vector3, b: Vector3, width: float, depth: float = -1.0) -> void:
	if a.distance_squared_to(b) < 0.00001:
		return
	var along := (b - a).normalized()
	var across := along.cross(Vector3.UP).normalized()
	if across.length_squared() < 0.5:
		across = Vector3.RIGHT
	var up := across.cross(along).normalized()
	across *= width * 0.5
	up *= (width if depth < 0.0 else depth) * 0.5
	var ring := [across + up, -across + up, -across - up, across - up]
	for i in 4:
		var j := (i + 1) % 4
		_quad(surface, a + ring[i], a + ring[j], b + ring[j], b + ring[i])
	_quad(surface, a + ring[3], a + ring[2], a + ring[1], a + ring[0])
	_quad(surface, b + ring[0], b + ring[1], b + ring[2], b + ring[3])


func _polygon(row: Dictionary) -> PackedVector2Array:
	var polygon := PackedVector2Array()
	for p in row.points:
		polygon.append(Vector2(float(p[0]), float(p[1])))
	if polygon.size() > 1 and polygon[0].distance_to(polygon[-1]) < 0.001:
		polygon.remove_at(polygon.size() - 1)
	return polygon


# The mapped outline contains both covered patio and enclosed rooms. A single
# extrusion incorrectly made the patio a tall solid block. Split only this
# observed building into low roof volumes and an upper control tower.
func _build_clubhouse(row: Dictionary, profile: Dictionary) -> void:
	var footprint := _polygon(row)
	var floor_y := INF
	for p in footprint:
		floor_y = minf(floor_y, _point([p.x, p.y]).y)
	floor_y -= 0.10
	for part: Dictionary in profile.parts:
		var r: Array = part.clip_xz
		var clip := PackedVector2Array(
			[Vector2(r[0], r[1]), Vector2(r[2], r[1]), Vector2(r[2], r[3]), Vector2(r[0], r[3])]
		)
		var polygons := Geometry2D.intersect_polygons(footprint, clip)
		assert(not polygons.is_empty(), "Clubhouse section misses mapped footprint")
		for polygon: PackedVector2Array in polygons:
			# Keep the original footprint winding so facade offsets face outward.
			var area := 0.0
			for i in polygon.size():
				area += polygon[i].cross(polygon[(i + 1) % polygon.size()])
			if area > 0.0:
				polygon.reverse()
			var section := part.duplicate(true)
			section.points = []
			for p in polygon:
				section.points.append([p.x, p.y])
			section.floor_y = floor_y + float(part.base_m)
			_build_building(section)


func _window(a: Vector3, b: Vector3, low: float, top: float, band: bool) -> void:
	var along := (b - a).normalized()
	var outward := along.cross(Vector3.UP).normalized()
	var left := a + outward * 0.045
	var right := b + outward * 0.045
	_quad(
		_glass,
		left + Vector3.UP * low,
		right + Vector3.UP * low,
		right + Vector3.UP * top,
		left + Vector3.UP * top
	)
	# Slim projecting frames break the featureless dark rectangle and cast real
	# shadows. Profile dimensions are artistic, not measured window joinery.
	for lift in [low, top]:
		_beam(
			_roofs,
			left + outward * 0.025 + Vector3.UP * lift,
			right + outward * 0.025 + Vector3.UP * lift,
			0.075
		)
	var count := maxi(1, ceili(a.distance_to(b) / (1.2 if band else 1.4)))
	for i in count + 1:
		var p := left.lerp(right, float(i) / count) + outward * 0.025
		_beam(_roofs, p + Vector3.UP * low, p + Vector3.UP * top, 0.055)


func _build_building(row: Dictionary) -> void:
	var polygon := _polygon(row)
	var triangles := Geometry2D.triangulate_polygon(polygon)
	assert(not triangles.is_empty(), "Mapped building footprint must triangulate")
	var floor_y := INF
	for p in polygon:
		floor_y = minf(floor_y, _point([p.x, p.y]).y)
	floor_y = float(row.get("floor_y", floor_y - 0.10))
	var height: float = row.height_m
	var roof_y := floor_y + height
	for i in range(0, triangles.size(), 3):
		var a: Vector2 = polygon[triangles[i]]
		var b: Vector2 = polygon[triangles[i + 1]]
		var c: Vector2 = polygon[triangles[i + 2]]
		_tri(
			_roofs,
			Vector3(a.x, roof_y, a.y),
			Vector3(b.x, roof_y, b.y),
			Vector3(c.x, roof_y, c.y),
			true
		)
	for i in polygon.size():
		var a2 := polygon[i]
		var b2 := polygon[(i + 1) % polygon.size()]
		var a := Vector3(a2.x, floor_y, a2.y)
		var b := Vector3(b2.x, floor_y, b2.y)
		var length := a.distance_to(b)
		var along := (b - a).normalized()
		var outward := along.cross(Vector3.UP).normalized()
		var overhang: float = row.get("roof_overhang_m", 0.11)
		_beam(
			_roofs,
			a + Vector3.UP * (height - 0.1),
			b + Vector3.UP * (height - 0.1),
			overhang * 2.0,
			0.20
		)
		if row.kind == "carport":
			var count := maxi(1, ceili(length / 7.0))
			for j in count + 1:
				var p := a.lerp(b, float(j) / count)
				_beam(_metal, p, p + Vector3.UP * height, 0.18)
		else:
			_quad(_walls, a, b, b + Vector3.UP * height, a + Vector3.UP * height)
			# Unreviewed buildings retain the original estimated panel placement.
			# The clubhouse uses reference driven horizontal bands on the tower.
			if length > 5.0:
				var garage: bool = row.kind in ["garage", "garages"]
				var levels: Array = row.get(
					"window_rows", [[0.25, 2.7]] if garage else [[1.1, 2.25]]
				)
				var band: bool = row.get("band_windows", false)
				var count := 1 if band else maxi(1, floori(length / 4.5))
				for level: Array in levels:
					for j in count:
						var center := a.lerp(b, (float(j) + 0.5) / count)
						var half := length * 0.5 - 0.7 if band else minf(1.3, length / count * 0.28)
						_window(
							center - along * half, center + along * half, level[0], level[1], band
						)


func _pavement_triangle(a: Vector2, b: Vector2, c: Vector2) -> void:
	var lengths := [a.distance_squared_to(b), b.distance_squared_to(c), c.distance_squared_to(a)]
	var edge: int = lengths.find(lengths.max())
	if float(lengths[edge]) > 64.0:
		var triangle := [a, b, c]
		var first: Vector2 = triangle[edge]
		var second: Vector2 = triangle[(edge + 1) % 3]
		var third: Vector2 = triangle[(edge + 2) % 3]
		var middle := first.lerp(second, 0.5)
		_pavement_triangle(first, middle, third)
		_pavement_triangle(middle, second, third)
		return
	_tri(
		_paving,
		_point([a.x, a.y], 0.045),
		_point([b.x, b.y], 0.045),
		_point([c.x, c.y], 0.045),
		true
	)


func _build_paving(row: Dictionary) -> void:
	var polygon := _polygon(row)
	var triangles := Geometry2D.triangulate_polygon(polygon)
	assert(not triangles.is_empty(), "Mapped paving footprint must triangulate")
	for i in range(0, triangles.size(), 3):
		_pavement_triangle(
			polygon[triangles[i]], polygon[triangles[i + 1]], polygon[triangles[i + 2]]
		)


func _build_fence(row: Dictionary) -> void:
	for i in row.points.size() - 1:
		var a := _point(row.points[i])
		var b := _point(row.points[i + 1])
		var count := maxi(1, ceili(a.distance_to(b) / 4.0))
		for j in count:
			var p := a.lerp(b, float(j) / count)
			var q := a.lerp(b, float(j + 1) / count)
			p.y = _point([p.x, p.z]).y
			q.y = _point([q.x, q.z]).y
			var height: float = row.height_m
			if row.get("kind", "") == "pit_wall":
				_beam(
					_concrete,
					p + Vector3.UP * height * .5,
					q + Vector3.UP * height * .5,
					0.35,
					height
				)
			else:
				_beam(_metal, p, p + Vector3.UP * height, 0.06)
				for lift in [0.45, 1.1, height]:
					_beam(_metal, p + Vector3.UP * lift, q + Vector3.UP * lift, 0.025)


func _build_marshal(row: Dictionary) -> void:
	var base := _point(row.p)
	for x in [-0.9, 0.9]:
		for z in [-0.8, 0.8]:
			_beam(_metal, base + Vector3(x, 0, z), base + Vector3(x, 2.3, z), 0.10)
	_tri(
		_roofs,
		base + Vector3(-1.1, 2.3, -1),
		base + Vector3(1.1, 2.3, -1),
		base + Vector3(1.1, 2.3, 1),
		true
	)
	_tri(
		_roofs,
		base + Vector3(-1.1, 2.3, -1),
		base + Vector3(1.1, 2.3, 1),
		base + Vector3(-1.1, 2.3, 1),
		true
	)


# Original geometric foliage. Each small bent leaf has its own normal and color,
# leaving visible sky holes instead of an opaque crown shell. Only crown placement
# comes from observations; branch/leaf structure is deterministic artistic detail.
func _leaf_cluster() -> ArrayMesh:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	surface.set_smooth_group(-1)
	var rng := RandomNumberGenerator.new()
	rng.seed = 73021
	for i in 110:
		var direction := (
			Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1))
			. normalized()
		)
		var center := direction * pow(rng.randf(), 0.3333) * 0.78
		var axis := (
			Vector3(rng.randf_range(-1, 1), rng.randf_range(-0.6, 0.6), rng.randf_range(-1, 1))
			. normalized()
		)
		var side := axis.cross(Vector3.UP).normalized()
		var size := rng.randf_range(0.085, 0.17)
		var a := center - axis * size
		var b := center + side * size * .58
		var c := center + axis * size
		var d := center - side * size * .58
		var ridge := center + Vector3.UP * size * .22
		surface.set_color(Color("374b26").lerp(Color("697143"), rng.randf()))
		_tri(surface, a, b, ridge)
		_tri(surface, b, c, ridge)
		_tri(surface, c, d, ridge)
		_tri(surface, d, a, ridge)
	surface.generate_normals()
	return surface.commit()


func _build_trees(rows: Array) -> void:
	const CLUSTERS := 46
	var foliage := MultiMesh.new()
	foliage.transform_format = MultiMesh.TRANSFORM_3D
	foliage.use_colors = true
	foliage.mesh = _leaf_cluster()
	foliage.instance_count = rows.size() * CLUSTERS
	var wood := SurfaceTool.new()
	wood.begin(Mesh.PRIMITIVE_TRIANGLES)
	wood.set_smooth_group(-1)
	var rng := RandomNumberGenerator.new()
	rng.seed = 86330
	for i in rows.size():
		var row: Dictionary = rows[i]
		var base := _point(row.p)
		var height: float = row.height_m
		var radius: float = row.radius_m
		var fork := base + Vector3.UP * height * .43
		_beam(wood, base, fork, radius * .095)
		for branch in 7:
			var angle := branch * TAU / 7.0 + float(i) * .7
			var tip := (
				base
				+ Vector3(
					cos(angle) * radius * .67,
					height * (.67 + rng.randf() * .14),
					sin(angle) * radius * .67
				)
			)
			var elbow := fork.lerp(tip, .55) - Vector3.UP * .18
			_beam(wood, fork, elbow, radius * .049)
			_beam(wood, elbow, tip, radius * .022)
		for crown in CLUSTERS:
			var angle := crown * 2.399963 + float(i) * .4
			var radial := sqrt((float(crown) + .5) / CLUSTERS)
			var spread := radius * .83 * radial
			var center := (
				base
				+ Vector3(
					cos(angle) * spread,
					height * (.60 + .21 * (1 - radial) + rng.randf_range(-.06, .09)),
					sin(angle) * spread
				)
			)
			var scale := radius * rng.randf_range(.26, .38)
			foliage.set_instance_transform(
				i * CLUSTERS + crown,
				Transform3D(
					Basis(Vector3.UP, angle).scaled(Vector3(scale, scale * 1.1, scale)), center
				)
			)
			foliage.set_instance_color(
				i * CLUSTERS + crown, Color.WHITE.lerp(Color("b8b191"), rng.randf() * .3)
			)
	var foliage_mat := _material(Color.WHITE, 1.0)
	foliage_mat.metallic_specular = 0.0
	foliage_mat.vertex_color_use_as_albedo = true
	# Leaf palette and instance tint are authored as sRGB hexadecimal colors.
	foliage_mat.vertex_color_is_srgb = true
	var foliage_instance := MultiMeshInstance3D.new()
	foliage_instance.name = "ObservedTreeCrowns"
	foliage_instance.multimesh = foliage
	foliage_instance.material_override = foliage_mat
	add_child(foliage_instance)
	_commit(wood, _material(Color("665948")), "ObservedTreeBranches")
