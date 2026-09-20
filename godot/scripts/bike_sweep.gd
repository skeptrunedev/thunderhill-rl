extends RefCounted
## Conservative continuous query for the articulated authored collision envelope.
## Clear intervals are rejected by enclosing volumes. Possible contact is refined
## to a declared spatial bound, never presented as an exact physical impact.

const MODEL_VERSION := "articulated-conservative-sweep-v1"
const SPATIAL_TOLERANCE_M := 0.002
const NUMERICAL_PADDING_M := 0.0005
const MAX_ROOT_COORDINATE_M := 4096.0
const MAX_INTERVALS := 4096
const MAX_DEPTH := 32

var envelope: RefCounted
var _parts: Array[Dictionary] = []
var _radius := 0.0
var _space: PhysicsDirectSpaceState3D
var _mask := 0
var _start: Dictionary
var _finish: Dictionary
var _rotation_start := Quaternion.IDENTITY
var _rotation_axis := Vector3.RIGHT
var _rotation_angle := 0.0
var _motion_bounds: Array[Vector2] = []
var _poses: Dictionary = {}
var _query := PhysicsShapeQueryParameters3D.new()
var _intervals := 0
var query_count := 0


func build(source: RefCounted) -> String:
	_parts.clear()
	_radius = 0.0
	envelope = null
	if source == null or source.components.is_empty():
		return "A populated motorcycle collision envelope is required"
	for component: Dictionary in source.components:
		var points := PackedVector3Array()
		var radius := 0.0
		for vertex in component.shape.points:
			var p: Vector3 = component.local_transform * vertex
			points.append(p)
			radius = maxf(radius, p.length())
		var bounds := AABB(points[0], Vector3.ZERO)
		for p in points:
			bounds = bounds.expand(p)
		_parts.append(
			{
				"points": points,
				"radius": radius,
				"bounds": bounds,
				"id": component.id,
				"joint": component.joint
			}
		)
		var reach := radius
		if component.joint == "front" or component.joint == "front_wheel":
			reach += source._front_origin.length()
		if component.joint == "front_wheel":
			reach += source._front_wheel_origin.length()
		if component.joint == "rear_wheel":
			reach += source._rear_origin.length()
		_radius = maxf(_radius, reach)
	envelope = source
	return ""


## Pose fields: root (proper rigid Transform3D), lean, steering, wheel_rotation.
## Root translation and all joint angles interpolate linearly. Root rotation uses
## constant angular speed along the shortest quaternion arc. Wheel angle is not
## wrapped: a full revolution must remain a full revolution.
func sweep(
	space: PhysicsDirectSpaceState3D, start: Dictionary, finish: Dictionary, collision_mask: int
) -> Dictionary:
	query_count = 0
	_intervals = 0
	_poses.clear()
	_motion_bounds.clear()
	if envelope == null:
		return {"error": "Motorcycle sweep has not been built"}
	for pose in [start, finish]:
		var error := _validate_pose(pose)
		if not error.is_empty():
			return {"error": error}
	_space = space
	_mask = collision_mask
	_start = start
	_finish = finish
	_query.margin = 0.0
	_query.collision_mask = _mask
	_query.collide_with_areas = false
	_query.collide_with_bodies = true
	var broad := AABB(start.root.origin, Vector3.ZERO).expand(finish.root.origin).grow(
		_radius + NUMERICAL_PADDING_M
	)
	if _box_hits(broad).is_empty():
		return {"status": "clear", "safe_fraction": 1.0, "queries": query_count}
	var initial: Array = envelope.overlaps(
		space, start.root, start.lean, start.steering, start.wheel_rotation, collision_mask
	)
	query_count += envelope.components.size()
	if not initial.is_empty():
		return {
			"status": "initial_overlap",
			"safe_fraction": 0.0,
			"interval_end_fraction": 0.0,
			"component_id": initial[0].component_id,
			"collider_id": initial[0].collider_id,
			"queries": query_count
		}
	_rotation_start = start.root.basis.get_rotation_quaternion().normalized()
	var delta: Quaternion = (
		(_rotation_start.inverse() * finish.root.basis.get_rotation_quaternion().normalized())
		. normalized()
	)
	if delta.w < 0.0:
		delta = -delta
	var vector := Vector3(delta.x, delta.y, delta.z)
	_rotation_angle = 2.0 * atan2(vector.length(), delta.w)
	_rotation_axis = vector.normalized() if vector.length_squared() > 0.0 else Vector3.RIGHT
	var w0 := _rotation_angle + absf(finish.lean - start.lean)
	var ws := w0 + absf(finish.steering - start.steering)
	var wheel := absf(finish.wheel_rotation - start.wheel_rotation)
	var translation: float = start.root.origin.distance_to(finish.root.origin)
	for part in _parts:
		var speed := translation
		var acceleration := 0.0
		var rate := w0
		if part.joint == "front" or part.joint == "front_wheel":
			speed += envelope._front_origin.length() * w0
			acceleration += envelope._front_origin.length() * w0 * w0
			rate = ws
		if part.joint == "front_wheel":
			speed += envelope._front_wheel_origin.length() * ws
			acceleration += envelope._front_wheel_origin.length() * ws * ws
			rate += wheel
		if part.joint == "rear_wheel":
			speed += envelope._rear_origin.length() * w0
			acceleration += envelope._rear_origin.length() * w0 * w0
			rate += wheel
		speed += part.radius * rate
		acceleration += part.radius * rate * rate
		var finest_h := pow(0.5, MAX_DEPTH)
		var finest_uncertainty := (
			speed * finest_h
			+ sqrt(3.0) * (acceleration * finest_h * finest_h / 8.0 + NUMERICAL_PADDING_M)
		)
		if not is_finite(finest_uncertainty) or finest_uncertainty > SPATIAL_TOLERANCE_M:
			return {"error": "Motorcycle sweep motion exceeds attainable precision"}
		_motion_bounds.append(Vector2(speed, acceleration))
	var candidates: Array[int] = []
	for i in _parts.size():
		candidates.append(i)
	var result := _interval(0.0, 1.0, candidates, 0)
	result["queries"] = query_count
	result["intervals"] = _intervals
	return result


func _interval(a: float, b: float, candidates: Array[int], depth: int) -> Dictionary:
	_intervals += 1
	if _intervals > MAX_INTERVALS:
		return {"error": "Motorcycle sweep interval budget exceeded"}
	var from := _joint_poses(a)
	var to := _joint_poses(b)
	var possible: Array[int] = []
	var first_hit := {}
	var maximum_uncertainty := 0.0
	for i in candidates:
		var part := _parts[i]
		var h := b - a
		var padding := _motion_bounds[i].y * h * h / 8.0 + NUMERICAL_PADDING_M
		# Very large angle intervals need subdivision before asking the engine
		# about huge hulls, which can exceed its numeric broadphase range.
		if padding > _radius:
			possible.append(i)
			maximum_uncertainty = INF
			continue
		var bounds: AABB = (from[part.joint] * part.bounds).merge(to[part.joint] * part.bounds)
		bounds = bounds.grow(padding)
		if _box_hits(bounds).is_empty():
			continue
		var center := bounds.get_center()
		var vertices := PackedVector3Array()
		for pose: Transform3D in [from[part.joint], to[part.joint]]:
			for p: Vector3 in part.points:
				var point: Vector3 = pose * p - center
				for x in [-1.0, 1.0]:
					for y in [-1.0, 1.0]:
						for z in [-1.0, 1.0]:
							vertices.append(point + Vector3(x, y, z) * padding)
		var shape := ConvexPolygonShape3D.new()
		shape.margin = 0.0
		shape.points = vertices
		_query.shape = shape
		_query.transform = Transform3D(Basis.IDENTITY, center)
		query_count += 1
		var hits := _space.intersect_shape(_query, 1)
		if hits.is_empty():
			continue
		possible.append(i)
		if first_hit.is_empty():
			first_hit = {"component_id": part.id, "collider_id": hits[0].collider_id}
		maximum_uncertainty = maxf(
			maximum_uncertainty, _motion_bounds[i].x * h + sqrt(3.0) * padding
		)
	if possible.is_empty():
		return {"status": "clear", "safe_fraction": b}
	if maximum_uncertainty <= SPATIAL_TOLERANCE_M:
		first_hit.merge(
			{
				"status": "possible_contact",
				"safe_fraction": a,
				"interval_end_fraction": b,
				"spatial_uncertainty_m": maximum_uncertainty
			}
		)
		return first_hit
	if depth >= MAX_DEPTH:
		return {"error": "Motorcycle sweep precision limit exceeded"}
	var mid := (a + b) * 0.5
	var left := _interval(a, mid, possible, depth + 1)
	if left.get("status", "") != "clear":
		return left
	return _interval(mid, b, possible, depth + 1)


func _joint_poses(fraction: float) -> Dictionary:
	if _poses.has(fraction):
		return _poses[fraction]
	var pose := pose_at(fraction)
	var body: Transform3D = pose.root * Transform3D(Basis(Vector3.BACK, pose.lean), Vector3.ZERO)
	var steer := Basis(Vector3.UP, pose.steering)
	var spin := Basis(Vector3.RIGHT, pose.wheel_rotation)
	var front: Transform3D = body * Transform3D(steer, envelope._front_origin)
	var result := {
		"body": body,
		"front": front,
		"rear_wheel": body * Transform3D(spin, envelope._rear_origin),
		"front_wheel": front * Transform3D(spin, envelope._front_wheel_origin)
	}
	_poses[fraction] = result
	return result


func _box_hits(bounds: AABB) -> Array[Dictionary]:
	var shape := BoxShape3D.new()
	shape.size = bounds.size
	shape.margin = 0.0
	_query.shape = shape
	_query.transform = Transform3D(Basis.IDENTITY, bounds.get_center())
	query_count += 1
	return _space.intersect_shape(_query, 1)


static func _validate_pose(pose: Dictionary) -> String:
	if not pose.get("root") is Transform3D:
		return "Sweep pose requires a rigid root transform"
	var transform: Transform3D = pose.root
	if (
		not transform.is_finite()
		or transform.basis.determinant() <= 0.0
		or not transform.basis.is_equal_approx(transform.basis.orthonormalized())
	):
		return "Sweep root must be a finite proper unit rotation"
	if (
		maxf(absf(transform.origin.x), maxf(absf(transform.origin.y), absf(transform.origin.z)))
		> MAX_ROOT_COORDINATE_M
	):
		return "Sweep pose exceeds supported local coordinate range"
	for key in ["lean", "steering", "wheel_rotation"]:
		if not pose.get(key) is float and not pose.get(key) is int:
			return "Sweep pose requires numeric " + key
		if not is_finite(float(pose[key])):
			return "Nonfinite sweep joint " + key
	return ""


## Only called after a possible_contact result from the most recent sweep.
func pose_at(fraction: float) -> Dictionary:
	var rotation := _rotation_start * Quaternion(_rotation_axis, _rotation_angle * fraction)
	return {
		"root":
		Transform3D(Basis(rotation), _start.root.origin.lerp(_finish.root.origin, fraction)),
		"lean": lerpf(_start.lean, _finish.lean, fraction),
		"steering": lerpf(_start.steering, _finish.steering, fraction),
		"wheel_rotation": lerpf(_start.wheel_rotation, _finish.wheel_rotation, fraction),
	}
