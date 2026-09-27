extends RefCounted
## Convex parts of the authored car mesh for the articulated sweep
## (bike_sweep.gd). Every part rides the rigid body joint: body panels as one
## hull per mesh, each rear tire as its bounding box and each front tire as the
## hull of its bounding box across the full steering range, so steering and
## wheel spin never leave the envelope. An envelope of the artwork, not CAD.

const MODEL_VERSION := "car-authored-convex-parts-v1"
# Referenced by script, not class_name: a fresh checkout's class cache may not
# list new classes until the editor rescans.
const CarVisualScript = preload("res://scripts/car_visual.gd")
const STEER_SAMPLES := 9

# Interface read by bike_sweep.gd: only the body joint is used, so the
# motorcycle's articulated joint origins are zero and the rider arms absent.
var arms_enabled := false
var components: Array[Dictionary] = []
var _front_origin := Vector3.ZERO
var _rear_origin := Vector3.ZERO
var _front_wheel_origin := Vector3.ZERO


func build(car: Node3D, steering_limit_rad: float) -> String:
	components.clear()
	if car == null or not is_instance_valid(car):
		return "A constructed CarVisual is required"
	if car.get_script() != CarVisualScript:
		return "Expected CarVisual geometry"
	if car.position != Vector3.ZERO or not car.scale.is_equal_approx(Vector3.ONE):
		return "CarVisual root must have unit scale and zero local position"
	if car.wheels.size() != 4:
		return "CarVisual wheels have not been constructed"
	if not is_finite(steering_limit_rad) or steering_limit_rad < 0.0:
		return "Steering limit must be finite and nonnegative"
	var wheel_nodes: Dictionary = {}
	for wheel in car.wheels:
		wheel_nodes[wheel] = true
	var pending: Array[Dictionary] = []
	var error := _collect(car, car, Transform3D.IDENTITY, "root", wheel_nodes, pending)
	if not error.is_empty():
		return error
	var half := Vector3(
		CarVisualScript.TIRE_WIDTH_M * 0.5, CarVisualScript.TIRE_RADIUS_M, CarVisualScript.TIRE_RADIUS_M
	)
	for index in 4:
		var wheel: Node3D = car.wheels[index]
		var steered := index < 2
		var center: Vector3 = (wheel.get_parent() as Node3D).position if steered else wheel.position
		var points := PackedVector3Array()
		for sample in (STEER_SAMPLES if steered else 1):
			var angle := 0.0
			if steered:
				angle = lerpf(-steering_limit_rad, steering_limit_rad, float(sample) / (STEER_SAMPLES - 1))
			var turn := Basis(Vector3.UP, angle)
			for corner in 8:
				var offset := Vector3(
					half.x * (1.0 if corner & 4 else -1.0),
					half.y * (1.0 if corner & 2 else -1.0),
					half.z * (1.0 if corner & 1 else -1.0)
				)
				points.append(turn * offset)
		var shape := ConvexPolygonShape3D.new()
		shape.points = points
		shape.margin = 0.0
		pending.append({
			"id": "wheel/" + str(index),
			"joint": "body",
			"shape": shape,
			"local_transform": Transform3D(Basis.IDENTITY, center),
			"source_vertices": points,
			"source_path": car.get_path_to(wheel),
		})
	if pending.is_empty():
		return "CarVisual has no mesh geometry"
	components = pending
	return ""


func _collect(
	car: Node3D,
	node: Node,
	relative: Transform3D,
	id: String,
	wheel_nodes: Dictionary,
	pending: Array[Dictionary]
) -> String:
	if wheel_nodes.has(node):
		return ""
	if node is Node3D and node.is_set_as_top_level():
		return "Independent world transform is unsupported at " + id
	if node != car and node is Node3D:
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
		var unique: Dictionary = {}
		for vertex in vertices:
			if not vertex.is_finite():
				return "Nonfinite mesh vertex at " + id
			unique[vertex] = true
		# Thin decals (race number discs) are covered by the panels behind them.
		if unique.size() >= 4 and _has_volume(PackedVector3Array(unique.keys())):
			var shape := ConvexPolygonShape3D.new()
			shape.points = PackedVector3Array(unique.keys())
			shape.margin = 0.0
			pending.append({
				"id": id,
				"joint": "body",
				"shape": shape,
				"local_transform": relative,
				"source_vertices": vertices,
				"source_path": car.get_path_to(node),
			})
	for child in node.get_children():
		var error := _collect(car, child, relative, id + "/" + str(child.get_index()), wheel_nodes, pending)
		if not error.is_empty():
			return error
	return ""


## Same transform contract as the motorcycle envelope: root_pose is the car
## root's world transform and lean rolls the body about it (BikeVisual angle
## conventions). Steering and wheel rotation are inside the wheel hulls.
func transforms(
	root_pose: Transform3D, lean: float, _steering: float, _wheel_rotation: float
) -> Array[Transform3D]:
	var body := root_pose * Transform3D(Basis(Vector3.BACK, lean), Vector3.ZERO)
	var result: Array[Transform3D] = []
	for component in components:
		result.append(body * component.local_transform)
	return result


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


static func _has_volume(vertices: PackedVector3Array) -> bool:
	return preload("res://scripts/bike_collision_envelope.gd")._has_volume(vertices)
