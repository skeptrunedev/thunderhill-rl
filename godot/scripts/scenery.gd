class_name TrackScenery
extends Node3D
## CC0 Poly Haven decorative vegetation, not surveyed individual plants.
## Braking boards are provisional visual landmarks, with no collision geometry.
## Mapped trees and buildings are provided separately by TrackLandmarks.

const GRASS_COUNT := 96000
const PATCH_SIZE := 32.0
var _track: Node3D
var _random := RandomNumberGenerator.new()
var _grass_exclusions: Array[PackedVector2Array] = []


func build(track: Node3D) -> void:
	_track = track
	_random.seed = 481927
	_grass_exclusions.clear()
	var landmarks: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/landmarks.json")
	)
	for row: Dictionary in landmarks.paving + landmarks.buildings:
		var polygon := PackedVector2Array()
		for point: Array in row.points:
			polygon.append(Vector2(point[0], point[1]))
		# Keep the whole clump clear, including its largest authored footprint.
		for expanded: PackedVector2Array in Geometry2D.offset_polygon(polygon, 0.4):
			_grass_exclusions.append(expanded)
	_build_grass()
	_build_boards()


func _ground_height(p: Vector3, _road: Dictionary) -> float:
	return _track.terrain_surface_height(p)


func _grass_meshes() -> Array[ArrayMesh]:
	var data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://assets/grass/grass.json")
	)
	var material := StandardMaterial3D.new()
	material.albedo_texture = load("res://assets/grass/dry_grass_rgba.png")
	material.vertex_color_use_as_albedo = true
	material.vertex_color_is_srgb = true
	material.roughness = 1.0
	material.metallic_specular = 0.05
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	material.alpha_scissor_threshold = 0.35
	# MSAA alone only covers mesh edges, not the thin atlas cutouts inside them.
	material.alpha_antialiasing_mode = BaseMaterial3D.ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE
	material.alpha_antialiasing_edge = 0.30
	material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	var meshes: Array[ArrayMesh] = []
	# The source assets are individual stems, not complete roadside tussocks.
	# Assemble several sizes in each clump, keeping the source UVs and normals.
	# A separate seed keeps decorative mesh edits independent of world placement.
	var clump_random := RandomNumberGenerator.new()
	clump_random.seed = 91317
	for variant in range(3):
		var surface := SurfaceTool.new()
		surface.begin(Mesh.PRIMITIVE_TRIANGLES)
		for source_index: int in [0, 1, 1, 2, 2, 2]:
			var row: Dictionary = data.meshes[source_index]
			var scale := clump_random.randf_range(0.75, 1.15)
			if source_index == 0:
				scale *= 0.7
			var basis := Basis(Vector3.UP, clump_random.randf() * TAU)
			var angle := clump_random.randf() * TAU
			var radius := sqrt(clump_random.randf()) * 0.16
			var offset := Vector3(cos(angle), 0, sin(angle)) * radius
			# Source glTF uses counterclockwise front faces; Godot uses clockwise.
			for face in range(0, row.indices.size(), 3):
				for corner in [0, 2, 1]:
					var index := int(row.indices[face + corner])
					var p: Array = row.positions[index]
					var n: Array = row.normals[index]
					var uv: Array = row.uv[index]
					surface.set_normal(basis * Vector3(n[0], n[1], n[2]))
					surface.set_uv(Vector2(uv[0], uv[1]))
					surface.add_vertex(basis * Vector3(p[0], p[1], p[2]) * scale + offset)
		surface.set_material(material)
		meshes.append(surface.commit())
	return meshes


func _excluded_grass(position: Vector3) -> bool:
	for polygon in _grass_exclusions:
		if Geometry2D.is_point_in_polygon(Vector2(position.x, position.z), polygon):
			return true
	return false


func _build_grass() -> void:
	var patches: Dictionary = {}
	var positions: PackedVector3Array = _track.points
	var accepted := 0
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
		if _excluded_grass(position):
			continue
		var road: Dictionary = _track.sample_world(position)
		if absf(road.distance) < road.width * 0.5 + 1.4:
			continue
		position.y = road.height - 0.015
		var cell := Vector2i(floori(position.x / PATCH_SIZE), floori(position.z / PATCH_SIZE))
		if not patches.has(cell):
			patches[cell] = []
		patches[cell].append({"position": position, "normal": road.normal})
		accepted += 1
		if accepted >= GRASS_COUNT:
			break
	var meshes := _grass_meshes()
	for cell: Vector2i in patches:
		var positions_in_patch: Array = patches[cell]
		var origin := Vector3(
			(cell.x + 0.5) * PATCH_SIZE,
			positions_in_patch[0].position.y,
			(cell.y + 0.5) * PATCH_SIZE
		)
		var multi := MultiMesh.new()
		multi.transform_format = MultiMesh.TRANSFORM_3D
		multi.use_colors = true
		var variant := absi(cell.x * 73 + cell.y * 131) % meshes.size()
		multi.mesh = meshes[variant]
		multi.instance_count = positions_in_patch.size()
		for i in range(positions_in_patch.size()):
			var scale := _random.randf_range(0.65, 1.55)
			var ground_basis := Basis(Quaternion(Vector3.UP, positions_in_patch[i].normal))
			var basis := (
				ground_basis
				* Basis(Vector3.UP, _random.randf() * TAU).scaled(
					Vector3(scale, scale * _random.randf_range(0.7, 1.2), scale)
				)
			)
			multi.set_instance_transform(
				i, Transform3D(basis, positions_in_patch[i].position - origin)
			)
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


static func validate_bake() -> String:
	var manifest: Variant = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/scenery-bake.json")
	)
	if not manifest is Dictionary or manifest.get("schema_version") != 1:
		return "Missing or invalid scenery bake manifest"
	for source: String in manifest.sources:
		# Export converts scripts and textures. Packaging verifies their source
		# hashes; runtime can still verify the original ground JSON documents.
		if OS.has_feature("editor") or source.begins_with("data/"):
			if FileAccess.get_sha256("res://" + source) != manifest.sources[source]:
				return (
					"Stale scenery bake: "
					+ source
					+ ". Run res://tools/bake_scenery.gd with a renderer."
				)
	return ""
