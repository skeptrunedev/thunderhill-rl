extends RefCounted
## Historical aerial footprint study. Tree heights and row layout are estimates.


static func apply(track: Node3D) -> Dictionary:
	# Fractions of the personally inspected 1391 by 1792 aerial view.
	# Source extent: data/reference/ortho-measurements.json, NAD83 UTM zone 10.
	var regions := [
		PackedVector2Array(
			[Vector2(477, 0), Vector2(1370, 0), Vector2(1370, 153), Vector2(474, 153)]
		),
		PackedVector2Array([Vector2(948, 160), Vector2(1374, 155), Vector2(1098, 340)]),
		PackedVector2Array([Vector2(1391, 188), Vector2(1391, 823), Vector2(1145, 383)])
	]
	var mesh := SphereMesh.new()
	mesh.radius = 1.0
	mesh.height = 2.0
	mesh.radial_segments = 12
	mesh.rings = 6
	var transforms: Array[Transform3D] = []
	var colors: Array[Color] = []
	var rng := RandomNumberGenerator.new()
	rng.seed = 20220715
	for region_index in regions.size():
		var polygon: PackedVector2Array = regions[region_index]
		for row in range(0, 830, 11):
			for column in range(470, 1392, 11):
				var pixel := Vector2(column + (5 if row % 22 == 0 else 0), row)
				if not Geometry2D.is_point_in_polygon(pixel, polygon):
					continue
				var p := Vector3(
					556880.0 + pixel.x / 1391.0 * 1128.0 - float(track.data.origin.easting),
					0.0,
					float(track.data.origin.northing) - (4377699.2 - pixel.y / 1792.0 * 1451.4)
				)
				p.y = track.terrain_surface_height(p)
				var radius := rng.randf_range(2.7, 3.3)
				var height := rng.randf_range(4.0, 5.2)
				transforms.append(
					Transform3D(
						Basis.IDENTITY.scaled(Vector3(radius, height * 0.42, radius)),
						p + Vector3.UP * height * 0.58
					)
				)
				colors.append(Color("293d20").lerp(Color("485332"), rng.randf()))
	var multimesh := MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh.use_colors = true
	multimesh.mesh = mesh
	multimesh.instance_count = transforms.size()
	for i in transforms.size():
		multimesh.set_instance_transform(i, transforms[i])
		multimesh.set_instance_color(i, colors[i])
	var material := StandardMaterial3D.new()
	material.vertex_color_use_as_albedo = true
	material.vertex_color_is_srgb = true
	material.roughness = 1.0
	material.metallic_specular = 0.0
	var instance := MultiMeshInstance3D.new()
	instance.name = "HistoricalOrchardStudy"
	instance.multimesh = multimesh
	instance.material_override = material
	track.add_child(instance)
	return {
		"trees": transforms.size(),
		"source_sha256": "e49486d2d4f9c40b8d83a00e9ff02dd86407e5f09454f44cc835e1cd9f96573a",
		"source_date": "2022-07-15",
		"limitation":
		"Manually interpreted footprint; estimated trees, not contemporary survey. NAD83 datum realization differences are not corrected in this preview.",
		"script_sha256": FileAccess.get_sha256("res://scripts/orchard_study.gd")
	}


static func audit(track: Node3D, camera: Camera3D) -> Dictionary:
	var instance: MultiMeshInstance3D = track.get_node("HistoricalOrchardStudy")
	var viewport_size := camera.get_viewport().get_visible_rect().size
	var projected := 0
	var blocked := 0
	for i in instance.multimesh.instance_count:
		var pose := instance.multimesh.get_instance_transform(i)
		var top := pose.origin + Vector3.UP * pose.basis.y.length()
		var depth := -camera.to_local(top).z
		if depth < camera.near or depth > camera.far:
			continue
		var pixel := camera.unproject_position(top)
		if pixel.x < 0 or pixel.y < 0 or pixel.x >= viewport_size.x or pixel.y >= viewport_size.y:
			continue
		projected += 1
		var samples := ceili(camera.global_position.distance_to(top) / 8.0)
		for step in range(1, samples):
			var point := camera.global_position.lerp(top, float(step) / samples)
			if track.terrain_height(point) > point.y:
				blocked += 1
				break
	return {
		"tops_projected_in_view": projected,
		"tops_blocked_by_sampled_terrain": blocked,
		"method":
		"Eight metre approximate sightline sampling on terrain grid; not a pixel visibility or full canopy test"
	}
