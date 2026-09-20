extends RefCounted
## Convex parts of the authored mesh, including the rider regardless of camera
## visibility. This is an envelope of the artwork, not measured motorcycle CAD.
## Per mesh hulls fill local concavities (wheel centres, coils, panel recesses).

const MODEL_VERSION := "authored-convex-parts-v1"

var components: Array[Dictionary] = []
var _front_origin := Vector3.ZERO
var _rear_origin := Vector3.ZERO
var _front_wheel_origin := Vector3.ZERO


func build(bike: Node3D) -> String:
	components.clear()
	if bike == null or not is_instance_valid(bike):
		return "A constructed BikeVisual is required"
	if not bike is BikeVisual:
		return "Expected BikeVisual geometry"
	if bike.position != Vector3.ZERO or not bike.scale.is_equal_approx(Vector3.ONE):
		return "BikeVisual root must have unit scale and zero local position"
	if absf(bike.rotation.x) > 1e-7 or absf(bike.rotation.y) > 1e-7:
		return "BikeVisual root supports only the authored lean joint"
	for joint in [bike._front, bike._rear_wheel, bike._front_wheel]:
		if not is_instance_valid(joint):
			return "BikeVisual joints have not been constructed"
		if not joint.position.is_finite() or not joint.scale.is_equal_approx(Vector3.ONE):
			return "Invalid BikeVisual joint translation or scale"
	_front_origin = bike._front.position
	_rear_origin = bike._rear_wheel.position
	_front_wheel_origin = bike._front_wheel.position
	var pending: Array[Dictionary] = []
	var error := _collect(bike, bike, "body", Transform3D.IDENTITY, "root", pending)
	if not error.is_empty():
		return error
	if pending.is_empty():
		return "BikeVisual has no mesh geometry"
	components = pending
	return ""


func _collect(
	bike: Node3D,
	node: Node,
	joint: String,
	relative: Transform3D,
	id: String,
	pending: Array[Dictionary]
) -> String:
	if node is Node3D and node.is_set_as_top_level():
		return "Independent world transform is unsupported at " + id
	if node != bike:
		if node == bike._front:
			joint = "front"
			relative = Transform3D.IDENTITY
		elif node == bike._rear_wheel:
			joint = "rear_wheel"
			relative = Transform3D.IDENTITY
		elif node == bike._front_wheel:
			joint = "front_wheel"
			relative = Transform3D.IDENTITY
		elif node is Node3D:
			relative = relative * node.transform
	if not relative.is_finite() or absf(relative.basis.determinant()) < 1e-12:
		return "Invalid component transform at " + id
	if node is MeshInstance3D:
		if node.mesh == null:
			return "Missing mesh at " + id
		var vertices := PackedVector3Array()
		for surface in node.mesh.get_surface_count():
			var arrays: Array = node.mesh.surface_get_arrays(surface)
			if arrays.is_empty() or arrays[Mesh.ARRAY_VERTEX] == null:
				return "Missing vertices at " + id
			vertices.append_array(arrays[Mesh.ARRAY_VERTEX])
		if vertices.size() < 4:
			return "Insufficient volume vertices at " + id
		for vertex in vertices:
			if not vertex.is_finite():
				return "Nonfinite mesh vertex at " + id
		if not _has_volume(vertices):
			return "Mesh has no three dimensional volume at " + id
		# Engine mesh cleaning contracts some thin authored parts. Remove only
		# exact duplicates here and retain every distinct support vertex.
		var unique: Dictionary = {}
		for vertex in vertices:
			unique[vertex] = true
		var shape := ConvexPolygonShape3D.new()
		shape.points = PackedVector3Array(unique.keys())
		shape.margin = 0.0
		pending.append(
			{
				"id": id,
				"joint": joint,
				"shape": shape,
				"local_transform": relative,
				"source_vertices": vertices,
				"source_path": bike.get_path_to(node)
			}
		)
	for child in node.get_children():
		# Child indices are stable across scene instances, unlike generated names.
		var error := _collect(
			bike, child, joint, relative, id + "/" + str(child.get_index()), pending
		)
		if not error.is_empty():
			return error
	return ""


## root_pose is the bike root's world transform (heading and terrain pitch).
## Angles use BikeVisual.update_pose conventions, not simulation sign conventions.
## No live scene transforms or visibility are read during this calculation.
func transforms(
	root_pose: Transform3D, lean: float, steering: float, wheel_rotation: float
) -> Array[Transform3D]:
	var body := root_pose * Transform3D(Basis(Vector3.BACK, lean), Vector3.ZERO)
	var spin := Basis(Vector3.RIGHT, wheel_rotation)
	var front := body * Transform3D(Basis(Vector3.UP, steering), _front_origin)
	var joints := {
		"body": body,
		"front": front,
		"rear_wheel": body * Transform3D(spin, _rear_origin),
		"front_wheel": front * Transform3D(spin, _front_wheel_origin),
	}
	var result: Array[Transform3D] = []
	for component in components:
		result.append(joints[component.joint] * component.local_transform)
	return result


## Reports existence of contact for each component. One obstacle per component
## suffices here; this is neither an exhaustive manifold nor a swept query.
func overlaps(
	space: PhysicsDirectSpaceState3D,
	root_pose: Transform3D,
	lean: float,
	steering: float,
	wheel_rotation: float,
	collision_mask: int
) -> Array[Dictionary]:
	var poses := transforms(root_pose, lean, steering, wheel_rotation)
	var contacts: Array[Dictionary] = []
	var query := PhysicsShapeQueryParameters3D.new()
	query.margin = 0.0
	query.collision_mask = collision_mask
	query.collide_with_areas = false
	query.collide_with_bodies = true
	for i in components.size():
		query.shape = components[i].shape
		query.transform = poses[i]
		var hits := space.intersect_shape(query, 1)
		if not hits.is_empty():
			var contact: Dictionary = hits[0].duplicate()
			contact["component_id"] = components[i].id
			contact["joint"] = components[i].joint
			contacts.append(contact)
	return contacts


## Reject planar or collapsed parts before asking the physics engine for a hull.
## Tolerance scales with mesh extent, preserving the small instrument meshes.
static func _has_volume(vertices: PackedVector3Array) -> bool:
	var origin := vertices[0]
	var axis := Vector3.ZERO
	for vertex in vertices:
		if (vertex - origin).length_squared() > axis.length_squared():
			axis = vertex - origin
	var extent := axis.length()
	if extent == 0.0:
		return false
	axis /= extent
	var normal := Vector3.ZERO
	for vertex in vertices:
		var candidate := axis.cross(vertex - origin)
		if candidate.length_squared() > normal.length_squared():
			normal = candidate
	if normal.length() <= extent * 1e-7:
		return false
	normal = normal.normalized()
	for vertex in vertices:
		if absf(normal.dot(vertex - origin)) > extent * 1e-7:
			return true
	return false
