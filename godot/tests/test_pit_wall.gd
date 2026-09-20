extends SceneTree


class FlatTrack:
	extends Node3D
	var data := {"origin": {"easting": 1.0, "northing": 2.0, "elevation_m": 90.0}}

	func terrain_surface_height(p: Vector3) -> float:
		return p.z * 0.1


func _initialize() -> void:
	var track := FlatTrack.new()
	var wall := preload("res://scripts/pit_wall.gd").new()
	var profile := {
		"origin": track.data.origin,
		"visual_width_m": 0.35,
		"rows":
		[
			{"face_xz_m": [0, 0], "road_inward_xz": [1, 0], "height_m": 1.0},
			{"face_xz_m": [0, 10], "road_inward_xz": [1, 0], "height_m": 0.8}
		]
	}
	profile["track_sha256"] = FileAccess.get_sha256("res://data/track.json")
	wall.build(track, profile)
	assert(wall.initialization_error.is_empty(), wall.initialization_error)
	var mesh: Mesh = wall.get_child(0).mesh
	var arrays := mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
	assert(vertices.size() == 36, "Closed prism must have twelve triangles")
	for i in 6:
		assert(normals[i].x > 0.99, "Road facing wall normal must face the road")
		assert(normals[6 + i].y > 0.99, "Top must face upwards")
		assert(normals[12 + i].x < -0.99, "Back must face away from road")
		assert(normals[18 + i].y < -0.99, "Bottom must face downwards")
		assert(normals[24 + i].z < -0.99, "Start cap must face outwards")
		assert(normals[30 + i].z > 0.99, "End cap must face outwards")
	for p in vertices:
		assert(p.x >= -0.35001 and p.x <= 0.00001, "Wall must extrude away from road")
		assert(p.y >= track.terrain_surface_height(p) - 0.00001, "Wall penetrates support plane")
	assert(
		vertices.has(Vector3(0, 1.8, 10)), "Independent top elevation must follow changing height"
	)
	var folded := preload("res://scripts/pit_wall.gd").new()
	var folded_profile: Dictionary = profile.duplicate(true)
	folded_profile.rows[1].road_inward_xz = [-1, 0]
	folded.build(track, folded_profile)
	assert(not folded.initialization_error.is_empty(), "Folded solid must be rejected")
	assert(folded.get_child_count() == 0, "Invalid solid must not leave partial geometry")
	folded.free()
	var bad := preload("res://scripts/pit_wall.gd").new()
	profile.origin = {}
	bad.build(track, profile)
	assert(not bad.initialization_error.is_empty() and bad.get_child_count() == 0)
	bad.free()
	wall.free()
	track.free()
	print("PIT_WALL_CHECK passed")
	quit()
