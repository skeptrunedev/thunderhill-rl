extends SceneTree
## Actual physics queries against the same triangulated solid used by the wall.

const WALL := preload("res://scripts/pit_wall.gd")
var failures := 0
var checks := 0
var stage: Node3D


class FlatTrack:
	extends Node3D
	var data := {"origin": {"easting": 1.0, "northing": 2.0, "elevation_m": 90.0}}

	func terrain_surface_height(_p: Vector3) -> float:
		return 0.0


func _initialize() -> void:
	_run.call_deferred()


func _check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)


func _fixture(rows: Array, width: float, offset: Vector3) -> Node3D:
	var track := FlatTrack.new()
	var wall := WALL.new()
	(
		wall
		. build(
			track,
			{
				"origin": track.data.origin,
				"track_sha256": FileAccess.get_sha256("res://data/track.json"),
				"visual_width_m": width,
				"rows": rows,
			}
		)
	)
	track.free()
	_check(wall.initialization_error.is_empty(), wall.initialization_error)
	wall.position = offset
	stage.add_child(wall)
	return wall


func _query(p: Vector3, radius: float = 0.025) -> PhysicsShapeQueryParameters3D:
	var sphere := SphereShape3D.new()
	sphere.radius = radius
	sphere.margin = 0.0
	var query := PhysicsShapeQueryParameters3D.new()
	query.shape = sphere
	query.margin = 0.0
	query.collision_mask = WALL.COLLISION_LAYER
	query.transform.origin = p
	return query


func _overlap(p: Vector3, radius: float = 0.025) -> bool:
	return not (
		stage.get_world_3d().direct_space_state.intersect_shape(_query(p, radius), 1).is_empty()
	)


func _sweep(label: String, start: Vector3, finish: Vector3, expected: float) -> void:
	_check(not _overlap(start), label + " starts clear")
	_check(not _overlap(finish), label + " ends clear")
	var query := _query(start)
	query.motion = finish - start
	var fractions := stage.get_world_3d().direct_space_state.cast_motion(query)
	_check(fractions.size() == 2, label + " returns two fractions")
	if fractions.size() != 2:
		return
	_check(fractions[0] < 1.0 and fractions[1] < 1.0, label + " blocks full translation")
	# The engine reports a numerical bracket rather than an exact analytic root.
	_check(
		fractions[0] <= expected + 0.00001 and fractions[0] >= expected - 0.008,
		label + " safe fraction matches analytic sphere contact: " + str(fractions)
	)
	_check(
		fractions[1] >= expected - 0.00001 and fractions[1] <= expected + 0.008,
		label + " unsafe fraction matches analytic sphere contact: " + str(fractions)
	)


func _clear_sweep(label: String, start: Vector3, finish: Vector3) -> void:
	var query := _query(start)
	query.motion = finish - start
	var fractions := stage.get_world_3d().direct_space_state.cast_motion(query)
	_check(not _overlap(start) and not _overlap(finish), label + " endpoints clear")
	_check(fractions == PackedFloat32Array([1.0, 1.0]), label + " remains clear: " + str(fractions))


func _run() -> void:
	stage = Node3D.new()
	root.add_child(stage)
	var wall := _fixture(
		[
			{"face_xz_m": [0, 0], "road_inward_xz": [1, 0], "height_m": 1.0},
			{"face_xz_m": [0, 5], "road_inward_xz": [1, 0], "height_m": 1.0},
			{"face_xz_m": [0, 10], "road_inward_xz": [1, 0], "height_m": 1.0},
		],
		0.35,
		Vector3.ZERO
	)
	var warped := _fixture(
		[
			{"face_xz_m": [0, 0], "road_inward_xz": [1, 0], "height_m": 1.0},
			{"face_xz_m": [0, 4], "road_inward_xz": [1, -0.6], "height_m": 2.0},
		],
		1.0,
		Vector3(20, 0, 0)
	)
	if failures > 0:
		quit(1)
		return
	# A control hull deliberately bridges the concave roof. Its overlap proves
	# that the exterior probes distinguish exact shell collision from a hull.
	var mesh: Mesh = warped.get_child(0).mesh
	var vertices: PackedVector3Array = mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX]
	var unique_vertices := PackedVector3Array()
	for vertex in vertices:
		if not unique_vertices.has(vertex):
			unique_vertices.append(vertex)
	_check(unique_vertices.size() == 8, "Control hull uses exactly eight section vertices")
	var control := StaticBody3D.new()
	control.position = Vector3(40, 0, 0)
	control.collision_layer = WALL.COLLISION_LAYER
	control.collision_mask = 0
	var hull := ConvexPolygonShape3D.new()
	hull.points = unique_vertices
	hull.margin = 0.0
	var hull_instance := CollisionShape3D.new()
	hull_instance.shape = hull
	control.add_child(hull_instance)
	stage.add_child(control)
	await physics_frame
	await physics_frame
	print("PIT_WALL_COLLISION_ENGINE ", ProjectSettings.get_setting("physics/3d/physics_engine"))
	_check(wall.get_child(1).get_child_count() == 24, "Two sections contain 24 solid tetrahedra")
	_check(_overlap(Vector3(-0.175, 0.5, 2.5)), "Entirely enclosed sphere overlaps solid interior")
	_check(_overlap(Vector3(-0.175, 0.5, 5.0)), "Shared section cap has no interior gap")
	for station in [2.5, 5.0, 7.5]:
		_sweep(
			"Road face at " + str(station),
			Vector3(1, 0.5, station),
			Vector3(-1, 0.5, station),
			0.975 / 2.0
		)
		_sweep(
			"Back face at " + str(station),
			Vector3(-1, 0.5, station),
			Vector3(1, 0.5, station),
			0.625 / 2.0
		)
	_sweep("Top", Vector3(-0.175, 2, 2.5), Vector3(-0.175, -1, 2.5), 0.975 / 3.0)
	_sweep("Bottom", Vector3(-0.175, -1, 2.5), Vector3(-0.175, 2, 2.5), 0.975 / 3.0)
	_sweep("Start cap", Vector3(-0.175, 0.5, -1), Vector3(-0.175, 0.5, 11), 0.975 / 12.0)
	_sweep("End cap", Vector3(-0.175, 0.5, 11), Vector3(-0.175, 0.5, -1), 0.975 / 12.0)
	_clear_sweep("Above top", Vector3(1, 1.04, 5), Vector3(-1, 1.04, 5))
	_clear_sweep("Below bottom", Vector3(1, -0.04, 5), Vector3(-1, -0.04, 5))
	_clear_sweep("Before start", Vector3(1, 0.5, -0.04), Vector3(-1, 0.5, -0.04))
	_clear_sweep("After end", Vector3(1, 0.5, 10.04), Vector3(-1, 0.5, 10.04))
	_clear_sweep("Along road face", Vector3(0.04, 0.5, -1), Vector3(0.04, 0.5, 11))
	_clear_sweep("Along back face", Vector3(-0.39, 0.5, -1), Vector3(-0.39, 0.5, 11))
	var hull_only_overlaps := 0
	for first in [6, 9]:
		var center := (
			(vertices[first] + vertices[first + 1] + vertices[first + 2]) / 3.0 + warped.position
		)
		_check(
			not _overlap(center + Vector3.UP * 0.03, 0.01),
			"Warped top triangle " + str(first) + " exterior stays clear"
		)
		_check(
			_overlap(center - Vector3.UP * 0.03, 0.01),
			"Warped top triangle " + str(first) + " interior remains solid"
		)
		if _overlap(center + Vector3.UP * 0.03 + control.position - warped.position, 0.01):
			hull_only_overlaps += 1
	_check(
		hull_only_overlaps > 0,
		"Control hull overlaps at least one probe outside the triangulated roof"
	)
	await _real_profile()
	print("PIT_WALL_COLLISION_CHECK checks=", checks, " failures=", failures)
	quit(1 if failures else 0)


func _real_profile() -> void:
	# Remove synthetic geometry before querying the historical wall.
	for child in stage.get_children():
		child.free()
	var track := preload("res://scripts/track.gd").new()
	stage.add_child(track)
	_check(track.initialization_error.is_empty(), "Actual terrain initializes")
	if not track.initialization_error.is_empty():
		return
	var profile: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/pit-wall.json")
	)
	var wall := WALL.new()
	var started := Time.get_ticks_usec()
	wall.build(track, profile)
	_check(wall.initialization_error.is_empty(), wall.initialization_error)
	if not wall.initialization_error.is_empty():
		wall.free()
		return
	stage.add_child(wall)
	var construction_usec := Time.get_ticks_usec() - started
	_check(
		wall.get_child(1).get_child_count() == 12 * (profile.rows.size() - 1),
		"Actual profile includes every solid section"
	)
	await physics_frame
	await physics_frame
	var vertices: PackedVector3Array = wall.get_child(0).mesh.surface_get_arrays(0)[
		Mesh.ARRAY_VERTEX
	]
	started = Time.get_ticks_usec()
	for section in range(0, profile.rows.size() - 1, 7):
		# Each road face uses [a,d,c], [a,c,b]. Sample inside one triangle,
		# independently of the tetrahedron centers used by construction.
		var index := section * 24
		var a := vertices[index]
		var b := vertices[index + 1]
		var c := vertices[index + 2]
		var center := (a + b + c) / 3.0
		var outward := (c - a).cross(b - a).normalized()
		_check(
			_overlap(center - outward * 0.05, 0.01),
			"Actual section " + str(section) + " interior solid"
		)
		_check(
			not _overlap(center + outward * 0.05, 0.01),
			"Actual section " + str(section) + " road side clear"
		)
	var query_usec := Time.get_ticks_usec() - started
	print(
		"PIT_WALL_COLLISION_PROFILE construction_usec=",
		construction_usec,
		" query_usec=",
		query_usec
	)
