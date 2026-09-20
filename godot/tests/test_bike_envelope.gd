extends SceneTree
## Compare the authored mesh tree against independently posed collision shapes.

const BIKE := preload("res://scripts/bike_visual.gd")
const ENVELOPE := preload("res://scripts/bike_collision_envelope.gd")
var checks := 0
var failures := 0


func _initialize() -> void:
	_run.call_deferred()


func _check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)


func _meshes(node: Node) -> Array[MeshInstance3D]:
	var result: Array[MeshInstance3D] = []
	if node is MeshInstance3D:
		result.append(node)
	for child in node.get_children():
		result.append_array(_meshes(child))
	return result


func _vertices(mesh: Mesh) -> PackedVector3Array:
	var result := PackedVector3Array()
	for surface in mesh.get_surface_count():
		result.append_array(mesh.surface_get_arrays(surface)[Mesh.ARRAY_VERTEX])
	return result


func _support(points: PackedVector3Array, direction: Vector3) -> float:
	var support := -INF
	for point in points:
		support = maxf(support, point.dot(direction))
	return support


func _directions() -> Array[Vector3]:
	var result: Array[Vector3] = []
	for x in range(-1, 2):
		for y in range(-1, 2):
			for z in range(-1, 2):
				if x != 0 or y != 0 or z != 0:
					result.append(Vector3(x, y, z).normalized())
	# Deterministic sphere samples catch oblique support loss beyond the AABB.
	for i in 48:
		var y := 1.0 - 2.0 * (float(i) + 0.5) / 48.0
		var radius := sqrt(1.0 - y * y)
		var angle := float(i) * PI * (3.0 - sqrt(5.0))
		result.append(Vector3(cos(angle) * radius, y, sin(angle) * radius))
	return result


func _run() -> void:
	var stage := Node3D.new()
	root.add_child(stage)
	var bike := BIKE.new()
	stage.add_child(bike)
	bike.set_rider_visible(false)
	bike.update_instruments(0.0, 0.0, 0)
	# Capture with all joints already moved, so a build cannot bake their pose twice.
	stage.transform = Transform3D(Basis.from_euler(Vector3(0.12, 0.73, 0.0)), Vector3(7, 2, -4))
	bike.update_pose(0.47, -0.31, 1.83)
	var envelope := ENVELOPE.new()
	var build_started := Time.get_ticks_usec()
	var error: String = envelope.build(bike)
	var build_ms := (Time.get_ticks_usec() - build_started) / 1000.0
	_check(error.is_empty(), "authored bike envelope builds: " + error)
	if not error.is_empty():
		stage.free()
		quit(1)
		return
	var input_points := 0
	for component: Dictionary in envelope.components:
		input_points += component.shape.points.size()
	print(
		"Envelope geometry: ",
		JSON.stringify(
			{
				"components": envelope.components.size(),
				"shape_input_points": input_points,
				"build_ms": build_ms
			}
		)
	)
	var meshes := _meshes(bike)
	_check(
		envelope.components.size() == meshes.size(), "every authored mesh has a collision component"
	)
	var paths := {}
	var ids := {}
	var joints := {}
	var hidden_rider_count := 0
	var directions := _directions()
	for component: Dictionary in envelope.components:
		var id: String = component.id
		_check(not ids.has(id) and not id.is_empty(), "component ID is unique and nonempty")
		ids[id] = true
		paths[component.source_path] = true
		joints[component.joint] = true
		var mesh_node := bike.get_node(component.source_path) as MeshInstance3D
		_check(mesh_node != null, "source path resolves to an authored mesh")
		if mesh_node == null:
			continue
		if bike.rider.is_ancestor_of(mesh_node):
			hidden_rider_count += 1
		var source := _vertices(mesh_node.mesh)
		var recorded: PackedVector3Array = component.source_vertices
		var source_set := {}
		var recorded_set := {}
		for point in source:
			source_set[point] = true
		for point in recorded:
			recorded_set[point] = true
		_check(source_set == recorded_set, "snapshot contains every source vertex: " + id)
		var shape := component.shape as ConvexPolygonShape3D
		_check(
			shape != null and shape.points.size() >= 4, "component has a solid convex shape: " + id
		)
		if shape == null:
			continue
		var shape_set := {}
		for point in shape.points:
			shape_set[point] = true
		_check(shape_set == source_set, "convex input contains every exact authored vertex: " + id)
		var worst_support_loss := 0.0
		for direction in directions:
			worst_support_loss = maxf(
				worst_support_loss, _support(source, direction) - _support(shape.points, direction)
			)
		_check(
			worst_support_loss <= 0.000002,
			(
				"convex shape covers mesh support in 74 directions: "
				+ id
				+ " loss="
				+ str(worst_support_loss)
			)
		)
	for mesh_node in meshes:
		_check(paths.has(bike.get_path_to(mesh_node)), "mesh was not silently excluded")
	_check(hidden_rider_count > 0, "hidden cockpit rider still has collision coverage")
	for joint in [
		"body",
		"front",
		"rear_wheel",
		"front_wheel",
		"left_upper",
		"left_lower",
		"left_glove",
		"right_upper",
		"right_lower",
		"right_glove"
	]:
		_check(joints.has(joint), "joint represented: " + joint)
	var poses := [
		[Vector3.ZERO, Vector3.ZERO, 0.0, 0.0, 0.0],
		[Vector3(13, 4, -9), Vector3(0.22, 1.8, 0), 0.95, -0.48, 2.1],
		[Vector3(-12, -2, 21), Vector3(-0.31, -2.4, 0), -1.05, 0.5, -4.7],
		[Vector3(0.1, 0.2, 0.3), Vector3(0.03, 3.13, 0), 0.02, -0.02, 7.2],
	]
	for pose in poses:
		var root_pose := Transform3D(Basis.from_euler(pose[1]), pose[0])
		# Ask before changing the tree; no dependence on last render pose is allowed.
		var transforms: Array = envelope.transforms(root_pose, pose[2], pose[3], pose[4])
		_check(transforms.size() == envelope.components.size(), "pose returns every component")
		stage.transform = root_pose
		bike.update_pose(pose[2], pose[3], pose[4])
		var maximum_error := 0.0
		for index in mini(transforms.size(), envelope.components.size()):
			var component: Dictionary = envelope.components[index]
			var mesh_node := bike.get_node(component.source_path) as MeshInstance3D
			for vertex in _vertices(mesh_node.mesh):
				maximum_error = maxf(
					maximum_error,
					(transforms[index] * vertex).distance_to(mesh_node.global_transform * vertex)
				)
		_check(
			maximum_error < 0.00002,
			"independent joint pose matches every visual vertex: " + str(maximum_error)
		)
	await _check_overlaps(stage, bike, envelope)
	var second_bike := BIKE.new()
	stage.add_child(second_bike)
	var second := ENVELOPE.new()
	_check(second.build(second_bike).is_empty(), "second bike builds")
	var first_ids: Array = []
	var second_ids: Array = []
	for component: Dictionary in envelope.components:
		first_ids.append(component.id)
	for component: Dictionary in second.components:
		second_ids.append(component.id)
	_check(
		first_ids == second_ids, "IDs survive regenerated node names and different capture poses"
	)
	var invalid_mesh := MeshInstance3D.new()
	second_bike.add_child(invalid_mesh)
	_check(not second.build(second_bike).is_empty(), "null mesh fails explicitly")
	_check(
		second.components.is_empty(),
		"failed rebuild leaves no stale or partial collision components"
	)
	invalid_mesh.mesh = ArrayMesh.new()
	_check(not second.build(second_bike).is_empty(), "empty mesh fails explicitly")
	_check(second.components.is_empty(), "empty mesh failure stays atomic")
	invalid_mesh.mesh = QuadMesh.new()
	_check(not second.build(second_bike).is_empty(), "planar mesh fails explicitly")
	_check(second.components.is_empty(), "planar mesh failure stays atomic")
	invalid_mesh.mesh = BoxMesh.new()
	invalid_mesh.transform = Transform3D(
		Basis(Vector3.ZERO, Vector3.UP, Vector3.BACK), Vector3.ZERO
	)
	_check(not second.build(second_bike).is_empty(), "singular mesh transform fails explicitly")
	_check(second.components.is_empty(), "singular transform failure stays atomic")
	stage.free()
	print("Bike collision envelope: %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)


func _check_overlaps(stage: Node3D, bike: Node3D, envelope: RefCounted) -> void:
	var obstacle := StaticBody3D.new()
	obstacle.collision_layer = 1 << 4
	obstacle.collision_mask = 0
	var box := BoxShape3D.new()
	box.size = Vector3.ONE * 0.025
	box.margin = 0.0
	var collision := CollisionShape3D.new()
	collision.shape = box
	obstacle.add_child(collision)
	root.add_child(obstacle)
	var space := stage.get_world_3d().direct_space_state
	var root_pose := Transform3D(Basis.from_euler(Vector3(0.16, 0.8, 0)), Vector3(5, 3, 8))
	stage.transform = root_pose
	bike.update_pose(0.65, 0.4, 1.7)
	var points := [
		[bike.global_transform * Vector3(0, 1.485, -0.29), "body", "hidden helmet"],
		[
			bike._front.global_transform * Vector3(0.35, 0.708, 0.257),
			"front",
			"steered handlebar grip"
		],
	]
	for fixture in points:
		obstacle.position = fixture[0]
		await physics_frame
		await physics_frame
		var contacts: Array = envelope.overlaps(space, root_pose, 0.65, 0.4, 1.7, 1 << 4)
		var expected_joint_hit := false
		for contact: Dictionary in contacts:
			if contact.collider_id == obstacle.get_instance_id() and contact.joint == fixture[1]:
				expected_joint_hit = true
		_check(expected_joint_hit, fixture[2] + " hits actual physics obstacle")
		var point_query := PhysicsPointQueryParameters3D.new()
		point_query.collision_mask = 1 << 4
		for point in [
			root_pose.origin, bike._front_wheel.global_position, bike._rear_wheel.global_position
		]:
			point_query.position = point
			_check(
				space.intersect_point(point_query).is_empty(),
				fixture[2] + " contact is missed by wheel/root points"
			)
		_check(
			envelope.overlaps(space, root_pose, 0.65, 0.4, 1.7, 1 << 3).is_empty(),
			"collision mask excludes obstacle"
		)
	obstacle.position = Vector3(-10, 20, 30)
	await physics_frame
	await physics_frame
	_check(
		envelope.overlaps(space, root_pose, 0.65, 0.4, 1.7, 1 << 4).is_empty(),
		"separated obstacle remains clear"
	)
	obstacle.free()
