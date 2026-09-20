extends SceneTree
## Engine queries independently sample articulated poses to audit sweep bounds.
const ENVELOPE := preload("res://scripts/bike_collision_envelope.gd")
const SWEEP := preload("res://scripts/bike_sweep.gd")
const BIKE := preload("res://scripts/bike_visual.gd")
const MASK := 1 << 4
var checks := 0
var failures := 0
var obstacle: StaticBody3D
var box: BoxShape3D
var space: PhysicsDirectSpaceState3D


func _initialize() -> void:
	_run.call_deferred()


func _check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)


func _pose(
	at := Vector3.ZERO, axis := Vector3.UP, angle := 0.0, lean := 0.0, steer := 0.0, wheel := 0.0
) -> Dictionary:
	return {
		"root": Transform3D(Basis(axis, angle), at),
		"lean": lean,
		"steering": steer,
		"wheel_rotation": wheel
	}


func _interpolate(a: Dictionary, b: Dictionary, t: float) -> Dictionary:
	var qa: Quaternion = a.root.basis.get_rotation_quaternion()
	var qb: Quaternion = b.root.basis.get_rotation_quaternion()
	return {
		"root": Transform3D(Basis(qa.slerp(qb, t)), a.root.origin.lerp(b.root.origin, t)),
		"lean": lerpf(a.lean, b.lean, t),
		"steering": lerpf(a.steering, b.steering, t),
		"wheel_rotation": lerpf(a.wheel_rotation, b.wheel_rotation, t)
	}


func _hits(envelope: RefCounted, pose: Dictionary) -> bool:
	return not (
		envelope
		. overlaps(space, pose.root, pose.lean, pose.steering, pose.wheel_rotation, MASK)
		. is_empty()
	)


func _synthetic(joint: String, center: Vector3, size := Vector3.ONE * 0.1) -> RefCounted:
	var result := ENVELOPE.new()
	var shape := ConvexPolygonShape3D.new()
	var vertices := PackedVector3Array()
	for x in [-0.5, 0.5]:
		for y in [-0.5, 0.5]:
			for z in [-0.5, 0.5]:
				vertices.append(Vector3(x, y, z) * size)
	shape.points = vertices
	shape.margin = 0.0
	result.components.append(
		{
			"id": "test/box",
			"joint": joint,
			"shape": shape,
			"local_transform": Transform3D(Basis.IDENTITY, center)
		}
	)
	return result


func _place(at: Vector3, size: Vector3) -> void:
	obstacle.position = at
	box.size = size
	await physics_frame
	await physics_frame


func _contact_case(
	label: String,
	envelope: RefCounted,
	a: Dictionary,
	b: Dictionary,
	speed_bound: float,
	samples := 1024
) -> void:
	_check(not _hits(envelope, a), label + " starts clear")
	_check(not _hits(envelope, b), label + " ends clear")
	var solver := SWEEP.new()
	solver.profiling_enabled = "--profile-sweep" in OS.get_cmdline_user_args()
	_check(solver.build(envelope).is_empty(), label + " builds")
	var started := Time.get_ticks_usec()
	var result: Dictionary = solver.sweep(space, a, b, MASK)
	var elapsed_ms := (Time.get_ticks_usec() - started) / 1000.0
	print(label, " sweep ", JSON.stringify(result), " ms=", elapsed_ms)
	if solver.profiling_enabled:
		print(label, " profile ", JSON.stringify(solver.profile))
	_check(result.get("status", "") == "possible_contact", label + " catches intermediate contact")
	if result.get("status", "") != "possible_contact":
		return
	var first := -1.0
	for index in range(1, samples + 1):
		var t := float(index) / samples
		if _hits(envelope, _interpolate(a, b, t)):
			first = t
			break
	_check(first >= 0.0, label + " independent actual overlap exists")
	if first < 0.0:
		return
	var lo := maxf(0.0, first - 1.0 / samples)
	var hi := first
	for iteration in 20:
		var mid := (lo + hi) * 0.5
		if _hits(envelope, _interpolate(a, b, mid)):
			hi = mid
		else:
			lo = mid
	_check(result.safe_fraction <= hi, label + " safe bound precedes actual overlap")
	_check(
		not _hits(envelope, _interpolate(a, b, result.safe_fraction)),
		label + " safe pose is actually clear"
	)
	_check(result.interval_end_fraction >= result.safe_fraction, label + " ordered interval")
	_check(
		result.spatial_uncertainty_m <= SWEEP.SPATIAL_TOLERANCE_M, label + " declared spatial bound"
	)
	# Conservative padding may stop before true contact. Independent first overlap
	# must nevertheless lie within the declared distance uncertainty for fixtures.
	_check(
		(
			maxf(0.0, hi - result.safe_fraction) * speed_bound
			<= result.spatial_uncertainty_m + 0.00005
		),
		label + " first actual overlap agrees with uncertainty"
	)


func _run() -> void:
	obstacle = StaticBody3D.new()
	obstacle.collision_layer = MASK
	obstacle.collision_mask = 0
	box = BoxShape3D.new()
	box.margin = 0.0
	var collision := CollisionShape3D.new()
	collision.shape = box
	obstacle.add_child(collision)
	root.add_child(obstacle)
	space = obstacle.get_world_3d().direct_space_state
	var cube := _synthetic("body", Vector3.ZERO)
	await _place(Vector3.ZERO, Vector3(0.02, 3, 3))
	_contact_case(
		"fast translation", cube, _pose(Vector3(-10, 0, 0)), _pose(Vector3(10, 0, 0)), 20.0
	)
	var solver := SWEEP.new()
	solver.profiling_enabled = "--profile-sweep" in OS.get_cmdline_user_args()
	_check(solver.build(cube).is_empty(), "cube builds")
	var initial: Dictionary = solver.sweep(space, _pose(), _pose(Vector3(10, 0, 0)), MASK)
	_check(initial.get("status", "") == "initial_overlap", "initial overlap explicit")
	_check(initial.get("safe_fraction", -1.0) == 0.0, "initial overlap has no safe motion")
	var started := Time.get_ticks_usec()
	var clear: Dictionary = {}
	for i in 100:
		clear = solver.sweep(space, _pose(Vector3(-10, 3, 0)), _pose(Vector3(10, 3, 0)), MASK)
	_check(clear.get("status", "") == "clear", "translation near miss")
	print(
		"clear broadphase mean ms=",
		(Time.get_ticks_usec() - started) / 100000.0,
		" queries=",
		clear.get("queries")
	)
	started = Time.get_ticks_usec()
	for i in 100:
		clear = solver.sweep(
			space, _pose(Vector3(-10, 1.551, 0)), _pose(Vector3(10, 1.551, 0)), MASK
		)
	_check(clear.get("status", "") == "clear", "one millimetre near miss remains clear")
	_check(clear.get("queries", 0) > 1, "near miss exercises narrowphase")
	print(
		"near miss mean ms=",
		(Time.get_ticks_usec() - started) / 100000.0,
		" queries=",
		clear.get("queries")
	)
	_check(
		solver.sweep(space, _pose(), _pose(), 1 << 3).get("status", "") == "clear",
		"mask excludes initial overlap"
	)
	await _place(Vector3(1, 0, 0), Vector3.ONE * 0.12)
	var arm := _synthetic("body", Vector3(1, 0, 0))
	_contact_case(
		"root yaw",
		arm,
		_pose(Vector3.ZERO, Vector3.UP, -1.2),
		_pose(Vector3.ZERO, Vector3.UP, 1.2),
		2.6
	)
	_contact_case(
		"body lean",
		arm,
		_pose(Vector3.ZERO, Vector3.UP, 0, -1.2),
		_pose(Vector3.ZERO, Vector3.UP, 0, 1.2),
		2.6
	)
	var front := _synthetic("front", Vector3(1, 0, 0))
	front._front_origin = Vector3(0, 0, -0.7)
	await _place(Vector3(1, 0, -0.7), Vector3.ONE * 0.12)
	_contact_case(
		"front steering",
		front,
		_pose(Vector3.ZERO, Vector3.UP, 0, 0, -1.2),
		_pose(Vector3.ZERO, Vector3.UP, 0, 0, 1.2),
		2.6
	)
	var wheel := _synthetic("rear_wheel", Vector3(0, 1, 0))
	wheel._rear_origin = Vector3(0, 0, 0.7)
	await _place(Vector3(0, 0, 1.7), Vector3.ONE * 0.12)
	_contact_case(
		"unwrapped wheel revolution",
		wheel,
		_pose(),
		_pose(Vector3.ZERO, Vector3.UP, 0, 0, 0, TAU),
		6.8
	)
	var invalid := _pose()
	invalid.lean = NAN
	_check(solver.sweep(space, invalid, _pose(), MASK).has("error"), "nonfinite joint rejected")
	invalid = _pose()
	invalid.root = Transform3D(Basis.IDENTITY.scaled(Vector3(2, 1, 1)), Vector3.ZERO)
	_check(solver.sweep(space, invalid, _pose(), MASK).has("error"), "scaled root rejected")
	_check(SWEEP.new().sweep(space, _pose(), _pose(), MASK).has("error"), "unbuilt solver rejected")
	_check(
		solver.sweep(space, _pose(Vector3(1e20, 0, 0)), _pose(), MASK).has("error"),
		"unsupported coordinate range rejected"
	)
	invalid.root = Transform3D(Basis(Vector3.ZERO, Vector3.ZERO, Vector3.ZERO), Vector3.ZERO)
	_check(solver.sweep(space, invalid, _pose(), MASK).has("error"), "singular root rejected")
	var wheel_solver := SWEEP.new()
	_check(wheel_solver.build(wheel).is_empty(), "budget fixture builds")
	var enormous := _pose(Vector3.ZERO, Vector3.UP, 0, 0, 0, 1e12)
	var exhausted: Dictionary = wheel_solver.sweep(space, _pose(), enormous, MASK)
	_check(exhausted.has("error"), "unresolvable angular motion errors explicitly")
	print("precision exhaustion: ", JSON.stringify(exhausted))
	var bike := BIKE.new()
	root.add_child(bike)
	bike.set_rider_visible(false)
	var authored := ENVELOPE.new()
	_check(authored.build(bike).is_empty(), "authored bike builds")
	await _place(Vector3(0, 1.485, -0.29), Vector3.ONE * 0.025)
	_contact_case(
		"authored hidden helmet lean",
		authored,
		_pose(Vector3.ZERO, Vector3.UP, 0, -0.8),
		_pose(Vector3.ZERO, Vector3.UP, 0, 0.8),
		3.6,
		64
	)
	await _place(Vector3(1.0, 1.0, 0.0), Vector3(0.1, 4.0, 40.0))
	var authored_solver := SWEEP.new()
	_check(authored_solver.build(authored).is_empty(), "authored near wall solver builds")
	var durations: Array[float] = []
	var near_queries := 0
	for i in 10:
		var a := _pose(Vector3(0, 0, -float(i) * 0.3), Vector3.UP, 0, 0, 0, float(i) * 0.9)
		var b := _pose(Vector3(0, 0, -float(i + 1) * 0.3), Vector3.UP, 0, 0, 0, float(i + 1) * 0.9)
		started = Time.get_ticks_usec()
		var near: Dictionary = authored_solver.sweep(space, a, b, MASK)
		durations.append((Time.get_ticks_usec() - started) / 1000.0)
		near_queries += near.get("queries", 0)
		_check(
			near.get("status", "") == "clear",
			"authored parallel wall step stays clear: " + JSON.stringify(near)
		)
		_check(near.get("queries", 0) > 1, "authored wall step exercises component bounds")
	durations.sort()
	print(
		"authored parallel wall 10 steps median ms=",
		durations[5],
		" max ms=",
		durations[9],
		" mean queries=",
		near_queries / 10.0
	)
	bike.free()
	obstacle.free()
	print("Bike sweep: %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)
