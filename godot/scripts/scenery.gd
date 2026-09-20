class_name TrackScenery
extends Node3D
## Original decorative vegetation, not surveyed individual plants.
## Braking boards are provisional visual landmarks, with no collision geometry.
## Trees and paddock structures await georeferenced placement evidence.

const GRASS_COUNT := 24000
const PATCH_SIZE := 32.0
var _track: Node3D
var _random := RandomNumberGenerator.new()


func build(track: Node3D) -> void:
	_track = track
	_random.seed = 481927
	_build_grass()
	_build_boards()


func _ground_height(p: Vector3, _road: Dictionary) -> float:
	return _track.terrain_surface_height(p)


func _grass_mesh() -> ArrayMesh:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	# Bent tapered ribbons, no alpha textures, each clump contains six blades.
	for i in range(6):
		var angle := float(i) * 2.39996
		var width := 0.013 + float(i % 3) * 0.004
		var height := 0.16 + float(i % 4) * 0.042
		var offset := Vector3(cos(angle) * 0.05, 0, sin(angle) * 0.05)
		var side := Vector3(cos(angle), 0, sin(angle)) * width
		var bend := Vector3(sin(angle), 0, -cos(angle)) * height * 0.28
		var left := offset - side
		var right := offset + side
		var mid_left := offset + bend * 0.4 + Vector3.UP * height * 0.6 - side * 0.55
		var mid_right := offset + bend * 0.4 + Vector3.UP * height * 0.6 + side * 0.55
		var tip := offset + bend + Vector3.UP * height
		for vertex in [left, right, mid_right, left, mid_right, mid_left, mid_left, mid_right, tip]:
			surface.set_color(
				Color("877246").lerp(Color("c0ad75"), clampf(vertex.y / height, 0, 1))
			)
			surface.add_vertex(vertex)
	surface.generate_normals()
	var material := StandardMaterial3D.new()
	material.albedo_color = Color.WHITE
	material.vertex_color_use_as_albedo = true
	material.vertex_color_is_srgb = true
	material.roughness = 1
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	surface.set_material(material)
	return surface.commit()


func _build_grass() -> void:
	var patches: Dictionary = {}
	var positions: PackedVector3Array = _track.points
	# Sample around all sections rather than scattering uniformly over distant land.
	for attempt in range(GRASS_COUNT * 2):
		var index := _random.randi_range(0, positions.size() - 1)
		var next: int = (index + 1) % positions.size()
		var tangent := (positions[next] - positions[index]).normalized()
		var left := Vector3(tangent.z, 0, -tangent.x).normalized()
		var width: float = _track.samples[index].width
		var side := -1.0 if _random.randf() < 0.5 else 1.0
		var offset := width * 0.5 + 1.5 + pow(_random.randf(), 1.7) * 24.0
		var position := (
			positions[index].lerp(positions[next], _random.randf()) + left * offset * side
		)
		var road: Dictionary = _track.sample_world(position)
		if absf(road.distance) < road.width * 0.5 + 1.4:
			continue
		position.y = _ground_height(position, road) - 0.015
		var cell := Vector2i(floori(position.x / PATCH_SIZE), floori(position.z / PATCH_SIZE))
		if not patches.has(cell):
			patches[cell] = []
		patches[cell].append(position)
		if attempt >= GRASS_COUNT - 1:
			break
	var mesh := _grass_mesh()
	for cell: Vector2i in patches:
		var positions_in_patch: Array = patches[cell]
		var origin := Vector3(
			(cell.x + 0.5) * PATCH_SIZE, positions_in_patch[0].y, (cell.y + 0.5) * PATCH_SIZE
		)
		var multi := MultiMesh.new()
		multi.transform_format = MultiMesh.TRANSFORM_3D
		multi.use_colors = true
		multi.mesh = mesh
		multi.instance_count = positions_in_patch.size()
		for i in range(positions_in_patch.size()):
			var scale := _random.randf_range(0.65, 1.55)
			var basis := Basis(Vector3.UP, _random.randf() * TAU).scaled(
				Vector3(scale, scale * _random.randf_range(0.7, 1.2), scale)
			)
			multi.set_instance_transform(i, Transform3D(basis, positions_in_patch[i] - origin))
			multi.set_instance_color(i, Color.WHITE.lerp(Color("ab9d74"), _random.randf() * 0.25))
		var instance := MultiMeshInstance3D.new()
		instance.name = "DryGrass_%d_%d" % [cell.x, cell.y]
		instance.multimesh = multi
		instance.position = origin
		instance.visibility_range_end = 80
		instance.visibility_range_end_margin = 12
		instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		add_child(instance)


func _build_boards() -> void:
	var last_s := -200.0
	var points: PackedVector3Array = _track.points
	for i in range(0, points.size(), 3):
		var sample: Dictionary = _track.samples[i]
		var s: float = sample.s
		if absf(float(sample.curvature)) < 0.022 or s - last_s < 230:
			continue
		last_s = s
		# A single neutral marker, not an invented measured braking distance.
		var side := -1.0 if float(sample.curvature) > 0 else 1.0
		var position: Vector3 = _track.edge_point(i, side * (float(sample.width) * 0.5 + 3.0))
		var road: Dictionary = _track.sample_world(position)
		position.y = _ground_height(position, road)
		var board := Node3D.new()
		board.name = "ProvisionalVisualMarker_NoCollision"
		board.position = position
		var tangent: Vector3 = road.tangent
		board.rotation.y = atan2(tangent.x, tangent.z)
		add_child(board)
		var material := StandardMaterial3D.new()
		material.albedo_color = Color("e4e0d0")
		material.roughness = 0.9
		var panel := BoxMesh.new()
		panel.size = Vector3(0.56, 0.8, 0.045)
		var visual := MeshInstance3D.new()
		visual.mesh = panel
		visual.material_override = material
		visual.position.y = 0.61
		board.add_child(visual)
		var dark := StandardMaterial3D.new()
		dark.albedo_color = Color("343931")
		for x in [-0.16, 0.0, 0.16]:
			var stripe := BoxMesh.new()
			stripe.size = Vector3(0.065, 0.54, 0.05)
			var mark := MeshInstance3D.new()
			mark.mesh = stripe
			mark.material_override = dark
			mark.position = Vector3(x, 0.61, 0)
			board.add_child(mark)
