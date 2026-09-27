class_name CarVisual
extends Node3D
## Original MX-5 Cup inspired visual study, not manufacturer CAD. Overall
## length, width, wheelbase, track and tire size follow docs/car-reference.md;
## every body profile is an artistic approximation. The root sits on the ground
## midway between the axles (the static 50/50 centre of gravity), forward -Z.

const LENGTH_M := 3.915
const WIDTH_M := 1.735
const WHEELBASE_M := 2.310
const FRONT_TRACK_M := 1.495
const REAR_TRACK_M := 1.505
const TIRE_RADIUS_M := 0.305
const TIRE_WIDTH_M := 0.215
## Estimated left hand drive eye point and roof camera, not measured.
const RIDER_EYE_LOCAL := Vector3(-0.36, 1.12, 0.22)
const ONBOARD_CAMERA_LOCAL := Vector3(0.0, 1.16, 0.45)
const ONBOARD_LOOK_DOWN := 0.10
const RIDER_LOOK_DOWN := 0.08
const ONBOARD_FOV_DEG := 80.0
## The camera is fixed to the body, so it rolls fully with it.
const ONBOARD_ROLL_SCALE := 1.0
const RIDER_ROLL_SCALE := 1.0
## Layer hidden in the driver's eye view (main.gd culls layer 20 off the chase
## camera), so the cabin roof and pillars do not block the driver's view.
const CABIN_LAYER := 1 << 19

var _front_left: Node3D
var _front_right: Node3D
var wheels: Array[Node3D] = []
var cabin: Node3D
var _paint: StandardMaterial3D
var _white: StandardMaterial3D
var _black: StandardMaterial3D
var _glass: StandardMaterial3D
var _rubber: StandardMaterial3D
var _metal: StandardMaterial3D
var _amber: StandardMaterial3D
var _red_lamp: StandardMaterial3D


func _ready() -> void:
	_paint = _material(Color("a3111c"), 0.0, 0.22)
	_paint.clearcoat_enabled = true
	_paint.clearcoat = 0.9
	_paint.clearcoat_roughness = 0.08
	_white = _material(Color("e9ecee"), 0.0, 0.35)
	_black = _material(Color("15181a"), 0.3, 0.45)
	_glass = _material(Color("1c2630"), 0.2, 0.05)
	_rubber = _material(Color("121416"), 0.0, 0.9)
	_metal = _material(Color("9aa0a6"), 0.85, 0.3)
	_amber = _material(Color("f2e6c8"), 0.0, 0.1)
	_amber.emission_enabled = true
	_amber.emission = Color("fff2d0")
	_amber.emission_energy_multiplier = 0.4
	_red_lamp = _material(Color("8a0b10"), 0.0, 0.1)
	_red_lamp.emission_enabled = true
	_red_lamp.emission = Color("ff1010")
	_red_lamp.emission_energy_multiplier = 0.3
	_build_body()
	_build_cabin()
	var front_z := -WHEELBASE_M * 0.5
	var rear_z := WHEELBASE_M * 0.5
	_front_left = _steer_node(Vector3(-FRONT_TRACK_M * 0.5, TIRE_RADIUS_M, front_z))
	_front_right = _steer_node(Vector3(FRONT_TRACK_M * 0.5, TIRE_RADIUS_M, front_z))
	wheels = [
		_wheel(Vector3.ZERO, _front_left, -1.0),
		_wheel(Vector3.ZERO, _front_right, 1.0),
		_wheel(Vector3(-REAR_TRACK_M * 0.5, TIRE_RADIUS_M, rear_z), self, -1.0),
		_wheel(Vector3(REAR_TRACK_M * 0.5, TIRE_RADIUS_M, rear_z), self, 1.0),
	]


## Same signature and conventions as BikeVisual.update_pose: lean rolls the
## whole car about its ground reference, steering turns the front wheels.
func update_pose(lean: float, steering: float, wheel_rotation: float) -> void:
	rotation.z = lean
	if is_instance_valid(_front_left):
		_front_left.rotation.y = steering
		_front_right.rotation.y = steering
		for wheel in wheels:
			wheel.rotation.x = wheel_rotation


## No modeled dashboard; kept for interface parity with BikeVisual.
func update_instruments(_speed_mps: float, _rpm: float, _gear: int, _lap_seconds := 0.0) -> void:
	pass


func set_rider_visible(value: bool) -> void:
	if is_instance_valid(cabin):
		cabin.visible = value


func _material(color: Color, metallic: float, roughness: float) -> StandardMaterial3D:
	var result := StandardMaterial3D.new()
	result.albedo_color = color
	result.metallic = metallic
	result.roughness = roughness
	result.cull_mode = BaseMaterial3D.CULL_DISABLED
	return result


func _mesh(mesh: Mesh, material: Material, parent: Node3D, at := Vector3.ZERO) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.material_override = material
	instance.position = at
	parent.add_child(instance)
	return instance


func _rounded_box(
	size: Vector3, at: Vector3, radius: float, material: Material, parent: Node3D
) -> MeshInstance3D:
	return _mesh(preload("res://scripts/rounded_box.gd").build(size, radius), material, parent, at)


func _chamfered_box(
	size: Vector3, at: Vector3, bevel: float, material: Material, parent: Node3D
) -> MeshInstance3D:
	return _mesh(preload("res://scripts/chamfered_box.gd").build(size, bevel), material, parent, at)


## A closed convex slab from a side profile (z, y) extruded across x.
func _profile_slab(
	profile: PackedVector2Array, half_width: float, material: Material, parent: Node3D
) -> MeshInstance3D:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	var count := profile.size()
	var center := Vector2.ZERO
	for p in profile:
		center += p / count
	for side in [-1.0, 1.0]:
		for i in range(1, count - 1):
			var tri := [profile[0], profile[i], profile[i + 1]]
			if side > 0:
				tri.reverse()
			for p: Vector2 in tri:
				surface.set_normal(Vector3(side, 0, 0))
				surface.add_vertex(Vector3(side * half_width, p.y, p.x))
	for i in count:
		var p := profile[i]
		var q := profile[(i + 1) % count]
		var edge := q - p
		var outward := Vector2(edge.y, -edge.x).normalized()
		if outward.dot((p + q) * 0.5 - center) < 0.0:
			outward = -outward
		var normal := Vector3(0, outward.y, outward.x)
		var quad: Array[Vector3] = [
			Vector3(-half_width, p.y, p.x), Vector3(half_width, p.y, p.x),
			Vector3(half_width, q.y, q.x), Vector3(-half_width, q.y, q.x),
		]
		var face: Vector3 = (quad[1] - quad[0]).cross(quad[2] - quad[0])
		if face.dot(normal) > 0.0:
			quad.reverse()
		for index in [0, 1, 2, 0, 2, 3]:
			surface.set_normal(normal)
			surface.add_vertex(quad[index])
	return _mesh(surface.commit(), material, parent)


func _build_body() -> void:
	var half := WIDTH_M * 0.5
	var nose := -LENGTH_M * 0.5
	var tail := LENGTH_M * 0.5
	# Side profile of the body top (z forward negative, y up): low nose, long
	# hood, short high rear deck, as the ND silhouette. Artistic estimate.
	var top := PackedVector2Array([
		Vector2(nose, 0.34), Vector2(nose + 0.12, 0.58), Vector2(nose + 0.75, 0.70),
		Vector2(-0.35, 0.78), Vector2(0.95, 0.80), Vector2(tail - 0.12, 0.80),
		Vector2(tail, 0.62),
	])
	# Sections between wheel openings: bumpers, fenders over the wheels and
	# sills between them, so the tires show through their openings.
	var opening := TIRE_RADIUS_M + 0.04
	var front_axle := -WHEELBASE_M * 0.5
	var rear_axle := WHEELBASE_M * 0.5
	for section in [
		[nose, front_axle - opening, 0.16],
		[front_axle - opening, front_axle + opening, 2.0 * TIRE_RADIUS_M + 0.03],
		[front_axle + opening, rear_axle - opening, 0.17],
		[rear_axle - opening, rear_axle + opening, 2.0 * TIRE_RADIUS_M + 0.03],
		[rear_axle + opening, tail, 0.28],
	]:
		_body_section(top, float(section[0]), float(section[1]), float(section[2]), half - 0.06)
	# Front and rear fender bulges over the wheels.
	for z in [-WHEELBASE_M * 0.5, WHEELBASE_M * 0.5]:
		for side in [-1.0, 1.0]:
			_rounded_box(
				Vector3(0.12, 0.14, 0.80),
				Vector3(side * (half - 0.07), _top_at(top, z) - 0.075, z),
				0.05,
				_paint,
				self
			)
	# Side skirts, front splitter and rear diffuser in black composite.
	for side in [-1.0, 1.0]:
		_chamfered_box(Vector3(0.06, 0.10, 1.30), Vector3(side * (half - 0.05), 0.20, 0.0), 0.02, _black, self)
	_chamfered_box(Vector3(WIDTH_M - 0.20, 0.035, 0.18), Vector3(0, 0.11, nose + 0.12), 0.012, _black, self)
	_chamfered_box(Vector3(WIDTH_M - 0.40, 0.10, 0.14), Vector3(0, 0.22, tail - 0.08), 0.02, _black, self)
	# Grille, lamps and a white race number roundel on each door.
	_chamfered_box(Vector3(0.78, 0.13, 0.05), Vector3(0, 0.36, nose + 0.02), 0.02, _black, self)
	for side in [-1.0, 1.0]:
		_chamfered_box(Vector3(0.30, 0.07, 0.20), Vector3(side * 0.56, 0.60, nose + 0.20), 0.02, _amber, self)
		_chamfered_box(Vector3(0.26, 0.08, 0.05), Vector3(side * 0.58, 0.68, tail - 0.07), 0.02, _red_lamp, self)
		var roundel := CylinderMesh.new()
		roundel.top_radius = 0.20
		roundel.bottom_radius = 0.20
		roundel.height = 0.012
		var disc := _mesh(roundel, _white, self, Vector3(side * (half - 0.055), 0.52, 0.05))
		disc.rotation.z = PI * 0.5
	# Rear wing on two uprights (Cup cars carry a small wing).
	for side in [-1.0, 1.0]:
		_chamfered_box(Vector3(0.03, 0.18, 0.08), Vector3(side * 0.55, 0.89, tail - 0.18), 0.01, _black, self)
	_chamfered_box(Vector3(1.40, 0.025, 0.22), Vector3(0, 0.99, tail - 0.18), 0.01, _black, self)


## One closed slab of the body between two stations, from a flat bottom up to
## the top profile, sampled at the profile's own vertices inside the range.
func _body_section(
	top: PackedVector2Array, start: float, end: float, bottom: float, half_width: float
) -> void:
	var outline := PackedVector2Array([Vector2(start, bottom), Vector2(start, _top_at(top, start))])
	for point in top:
		if point.x > start and point.x < end:
			outline.append(point)
	outline.append(Vector2(end, _top_at(top, end)))
	outline.append(Vector2(end, bottom))
	_profile_slab(outline, half_width, _paint, self)


func _top_at(top: PackedVector2Array, z: float) -> float:
	for i in range(1, top.size()):
		if z <= top[i].x:
			var t := clampf((z - top[i - 1].x) / (top[i].x - top[i - 1].x), 0.0, 1.0)
			return lerpf(top[i - 1].y, top[i].y, t)
	return top[-1].y


func _build_cabin() -> void:
	# Hard top greenhouse, on its own node so the driver's eye view can hide it.
	cabin = Node3D.new()
	cabin.name = "Cabin"
	add_child(cabin)
	_profile_slab(PackedVector2Array([
		Vector2(-0.40, 0.79), Vector2(0.12, 1.19), Vector2(0.62, 1.19),
		Vector2(0.98, 0.81),
	]), 0.62, _glass, cabin)
	# Roof panel and pillars in body colour over the glass.
	_chamfered_box(Vector3(1.22, 0.04, 0.54), Vector3(0, 1.20, 0.37), 0.015, _paint, cabin)
	for side in [-1.0, 1.0]:
		_chamfered_box(Vector3(0.05, 0.05, 0.60), Vector3(side * 0.60, 0.99, -0.13), 0.015, _black, cabin)
	_set_layer(cabin, CABIN_LAYER)


func _set_layer(node: Node, layer: int) -> void:
	if node is VisualInstance3D:
		node.layers = layer
	for child in node.get_children():
		_set_layer(child, layer)


func _steer_node(at: Vector3) -> Node3D:
	var node := Node3D.new()
	node.position = at
	add_child(node)
	return node


## Revolve an axial (x, radius) profile about the wheel axle.
func _lathe(profile: Array[Vector2], material: Material, parent: Node3D) -> void:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	const SEGMENTS := 40
	for segment in range(SEGMENTS):
		var angle_a := TAU * float(segment) / SEGMENTS
		var angle_b := TAU * float(segment + 1) / SEGMENTS
		for ring in range(profile.size() - 1):
			var p := profile[ring]
			var q := profile[ring + 1]
			var a := Vector3(p.x, sin(angle_a) * p.y, cos(angle_a) * p.y)
			var b := Vector3(q.x, sin(angle_a) * q.y, cos(angle_a) * q.y)
			var c := Vector3(q.x, sin(angle_b) * q.y, cos(angle_b) * q.y)
			var d := Vector3(p.x, sin(angle_b) * p.y, cos(angle_b) * p.y)
			surface.add_vertex(a)
			surface.add_vertex(c)
			surface.add_vertex(b)
			surface.add_vertex(a)
			surface.add_vertex(d)
			surface.add_vertex(c)
	surface.generate_normals()
	_mesh(surface.commit(), material, parent)


func _wheel(at: Vector3, parent: Node3D, side: float) -> Node3D:
	var wheel := Node3D.new()
	wheel.position = at
	parent.add_child(wheel)
	var w := TIRE_WIDTH_M * 0.5
	var rim := 0.19  # 15 inch wheel
	_lathe([
		Vector2(-w * 0.9, rim), Vector2(-w, TIRE_RADIUS_M - 0.03),
		Vector2(-w * 0.8, TIRE_RADIUS_M), Vector2(w * 0.8, TIRE_RADIUS_M),
		Vector2(w, TIRE_RADIUS_M - 0.03), Vector2(w * 0.9, rim),
	], _rubber, wheel)
	# Outer rim face with five spokes, on the outboard side.
	var face := side * w * 0.7
	_lathe([
		Vector2(face - 0.01, rim - 0.02), Vector2(face - 0.01, rim),
		Vector2(face + 0.01, rim), Vector2(face + 0.01, rim - 0.02),
	], _metal, wheel)
	for i in 5:
		var spoke := BoxMesh.new()
		spoke.size = Vector3(0.02, rim * 2.0 - 0.04, 0.045)
		var instance := _mesh(spoke, _black, wheel, Vector3(face, 0, 0))
		instance.rotation.x = TAU * i / 5.0
	var hub := CylinderMesh.new()
	hub.top_radius = 0.05
	hub.bottom_radius = 0.05
	hub.height = 0.03
	var cap := _mesh(hub, _metal, wheel, Vector3(face, 0, 0))
	cap.rotation.z = PI * 0.5
	return wheel
