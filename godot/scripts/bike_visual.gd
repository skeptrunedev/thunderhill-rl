class_name BikeVisual
extends Node3D
## Original Streetfighter inspired visual study, not manufacturer CAD.
## All bodywork profiles are artistic approximations. Axle spacing and nominal
## unloaded tire envelopes follow docs/motorcycle-reference.md.

## Approximate eye location inside the original helmet mesh, not measured rider data.
const RIDER_EYE_LOCAL := Vector3(0.0, 1.51, -0.30)
## Approximate onboard framing anchor, not a measured eye or camera mount.
const ONBOARD_CAMERA_LOCAL := Vector3(0.0, 1.14, -0.30)
const ONBOARD_LOOK_DOWN := 0.40

var _front: Node3D
var _rear_wheel: Node3D
var _front_wheel: Node3D
var _red: StandardMaterial3D
var _black: StandardMaterial3D
var _metal: StandardMaterial3D
var _gold: StandardMaterial3D
var _rubber: StandardMaterial3D
var rider: Node3D
var _arms: Node3D
var _rider_visible := true
var _display: Control
var _display_viewport: SubViewport


func _ready() -> void:
	# Solid red paint is a dielectric coating, not partially exposed metal.
	_red = _material(Color("bc101b"), 0.0, 0.20)
	_red.clearcoat_enabled = true
	_red.clearcoat = 0.9
	_red.clearcoat_roughness = 0.08
	_black = _material(Color("14191b"), 0.5, 0.36)
	_metal = _material(Color("a5abb0"), 0.9, 0.25)
	_gold = _material(Color("ab8650"), 0.8, 0.28)
	_rubber = _material(Color("121416"), 0.0, 0.87)
	_rear_wheel = _wheel(Vector3(0, 0.3359, 0.75), 0.3359, 0.200, self, false)
	_front = Node3D.new()
	_front.position = Vector3(0, 0.2999, -0.746)
	add_child(_front)
	_front_wheel = _wheel(Vector3.ZERO, 0.2999, 0.120, _front, true)
	_build_body()
	_build_front()
	_build_chassis()
	_build_rider()


func set_rider_visible(value: bool) -> void:
	_rider_visible = value
	if is_instance_valid(rider):
		rider.visible = value
	if is_instance_valid(_arms):
		_arms.visible = value


## Simulation speed is metres per second; dashboard displays kilometres per hour.
func update_instruments(speed_mps: float, rpm: float, gear: int, lap_seconds := 0.0) -> void:
	if not is_instance_valid(_display):
		return
	if _display.set_readings(speed_mps, rpm, gear, lap_seconds):
		_display_viewport.render_target_update_mode = SubViewport.UPDATE_ONCE


func update_pose(lean: float, steering: float, wheel_rotation: float) -> void:
	if is_instance_valid(_arms):
		var error: String = _arms.set_steering(steering)
		if not error.is_empty():
			push_error(error)
			return
	rotation.z = lean
	if is_instance_valid(_front):
		_front.rotation.y = steering
		_front_wheel.rotation.x = wheel_rotation
		_rear_wheel.rotation.x = wheel_rotation


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


func _bar(a: Vector3, b: Vector3, radius: float, material: Material, parent: Node3D) -> void:
	var cylinder := CylinderMesh.new()
	cylinder.top_radius = radius
	cylinder.bottom_radius = radius
	cylinder.height = a.distance_to(b)
	cylinder.radial_segments = 24
	var instance := _mesh(cylinder, material, parent, (a + b) * 0.5)
	var axis := (b - a).normalized()
	var tangent := Vector3.RIGHT
	if absf(axis.dot(tangent)) > 0.95:
		tangent = Vector3.FORWARD
	var side := axis.cross(tangent).normalized()
	instance.basis = Basis(side, axis, side.cross(axis).normalized())


func _box(size: Vector3, at: Vector3, material: Material, parent: Node3D) -> MeshInstance3D:
	var box := BoxMesh.new()
	box.size = size
	return _mesh(box, material, parent, at)


func _chamfered_box(
	size: Vector3, at: Vector3, bevel: float, material: Material, parent: Node3D
) -> MeshInstance3D:
	return _mesh(preload("res://scripts/chamfered_box.gd").build(size, bevel), material, parent, at)


func _triangle(surface: SurfaceTool, a: Vector3, b: Vector3, c: Vector3) -> void:
	surface.add_vertex(a)
	surface.add_vertex(b)
	surface.add_vertex(c)


## Revolve an axial (x, radius) profile about the wheel axle.
func _lathe(
	profile: Array[Vector2], material: Material, parent: Node3D, at := Vector3.ZERO
) -> void:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	const SEGMENTS := 64
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
			_triangle(surface, a, b, c)
			_triangle(surface, a, c, d)
	surface.generate_normals()
	_mesh(surface.commit(), material, parent, at)


func _wheel(at: Vector3, radius: float, width: float, parent: Node3D, front: bool) -> Node3D:
	var wheel := Node3D.new()
	wheel.position = at
	parent.add_child(wheel)
	var half_width := width * 0.5
	var rim := 0.2159
	_lathe(
		[
			Vector2(-half_width * 0.72, rim),
			Vector2(-half_width, radius - 0.035),
			Vector2(-half_width * 0.64, radius - 0.008),
			Vector2(0, radius),
			Vector2(half_width * 0.64, radius - 0.008),
			Vector2(half_width, radius - 0.035),
			Vector2(half_width * 0.72, rim)
		],
		_rubber,
		wheel
	)
	_lathe(
		[
			Vector2(-half_width * 0.73, rim - 0.015),
			Vector2(-half_width * 0.73, rim),
			Vector2(half_width * 0.73, rim),
			Vector2(half_width * 0.73, rim - 0.015)
		],
		_black,
		wheel
	)
	_bar(Vector3(-half_width, 0, 0), Vector3(half_width, 0, 0), 0.043, _metal, wheel)
	for i in range(5):
		var angle := TAU * float(i) / 5
		for branch in [-1.0, 1.0]:
			var end: float = angle + branch * 0.14
			_bar(
				Vector3(0, sin(angle) * 0.046, cos(angle) * 0.046),
				Vector3(0, sin(end) * (rim - 0.013), cos(end) * (rim - 0.013)),
				0.011,
				_black,
				wheel
			)
	var disc_radius := 0.165 if front else 0.1225
	for side in [-1.0, 1.0] if front else [1.0]:
		var offset: float = side * (half_width + 0.008)
		_lathe(
			[
				Vector2(offset - 0.003, disc_radius * 0.60),
				Vector2(offset - 0.003, disc_radius),
				Vector2(offset + 0.003, disc_radius),
				Vector2(offset + 0.003, disc_radius * 0.60)
			],
			_metal,
			wheel
		)
		for bolt in range(12):
			var angle := TAU * float(bolt) / 12
			_bar(
				Vector3(
					offset - 0.004, sin(angle) * disc_radius * 0.85, cos(angle) * disc_radius * 0.85
				),
				Vector3(
					offset + 0.004, sin(angle) * disc_radius * 0.85, cos(angle) * disc_radius * 0.85
				),
				0.004,
				_black,
				wheel
			)
	return wheel


## Shape preserving Hermite interpolation. Harmonic slopes stop at extrema,
## keeping every interpolated dimension inside its authored segment bounds.
func _bounded_hermite(
	previous: float,
	a: float,
	b: float,
	following: float,
	t: float,
	before_span := 1.0,
	span := 1.0,
	after_span := 1.0
) -> float:
	var slope := (b - a) / span
	var before := (a - previous) / before_span
	var after := (following - b) / after_span
	var start_tangent := 0.0
	var end_tangent := 0.0
	if before * slope > 0:
		start_tangent = 2.0 * before * slope / (before + slope)
	if after * slope > 0:
		end_tangent = 2.0 * slope * after / (slope + after)
	var t2 := t * t
	var t3 := t2 * t
	var value := (
		(2 * t3 - 3 * t2 + 1) * a
		+ (t3 - 2 * t2 + t) * span * start_tangent
		+ (-2 * t3 + 3 * t2) * b
		+ (t3 - t2) * span * end_tangent
	)
	return clampf(value, minf(a, b), maxf(a, b))


## Each ring is (longitudinal z, center height, half width, half height).
## Authored creases become bounded smooth shoulders, not inflated primitives.
func _body_panel(rings: Array[Vector4], material: Material, parent: Node3D) -> void:
	var profile: Array[Vector2] = [
		Vector2(-0.35, 0.99),
		Vector2(0, 1.015),
		Vector2(0.35, 0.99),
		Vector2(0.70, 0.87),
		Vector2(0.93, 0.60),
		Vector2(1, 0.24),
		Vector2(0.94, -0.22),
		Vector2(0.77, -0.68),
		Vector2(0.44, -0.96),
		Vector2(0, -1),
		Vector2(-0.44, -0.96),
		Vector2(-0.77, -0.68),
		Vector2(-0.94, -0.22),
		Vector2(-1, 0.24),
		Vector2(-0.93, 0.60),
		Vector2(-0.70, 0.87)
	]
	var perimeter := PackedVector2Array()
	for i in range(profile.size()):
		var previous := profile[posmod(i - 1, profile.size())]
		var a := profile[i]
		var b := profile[(i + 1) % profile.size()]
		var following := profile[(i + 2) % profile.size()]
		for subdivision in range(4):
			var t := float(subdivision) / 4
			perimeter.append(
				Vector2(
					_bounded_hermite(previous.x, a.x, b.x, following.x, t),
					_bounded_hermite(previous.y, a.y, b.y, following.y, t)
				)
			)
	var interpolated: Array[Vector4] = []
	for i in range(rings.size() - 1):
		var a := rings[i]
		var b := rings[i + 1]
		var previous := rings[i - 1] if i > 0 else a * 2 - b
		var following := rings[i + 2] if i + 2 < rings.size() else b * 2 - a
		for subdivision in range(6):
			var t := float(subdivision) / 6
			var ring := Vector4(lerpf(a.x, b.x, t), 0, 0, 0)
			for component in range(1, 4):
				ring[component] = _bounded_hermite(
					previous[component],
					a[component],
					b[component],
					following[component],
					t,
					a.x - previous.x,
					b.x - a.x,
					following.x - b.x
				)
			interpolated.append(ring)
	interpolated.append(rings[-1])
	var sections: Array[PackedVector3Array] = []
	for ring in interpolated:
		var section := PackedVector3Array()
		for point in perimeter:
			section.append(Vector3(point.x * ring.z, ring.y + point.y * ring.w, ring.x))
		sections.append(section)
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	surface.set_smooth_group(0)
	for i in range(sections.size() - 1):
		for j in range(perimeter.size()):
			var next := (j + 1) % perimeter.size()
			_triangle(surface, sections[i][j], sections[i + 1][j], sections[i + 1][next])
			_triangle(surface, sections[i][j], sections[i + 1][next], sections[i][next])
	# Cap seams intentionally stay sharp instead of blending into shoulder normals.
	surface.set_smooth_group(-1)
	for end in [0, sections.size() - 1]:
		for j in range(1, perimeter.size() - 1):
			if end == 0:
				_triangle(surface, sections[end][0], sections[end][j], sections[end][j + 1])
			else:
				_triangle(surface, sections[end][0], sections[end][j + 1], sections[end][j])
	surface.index()
	surface.generate_normals()
	_mesh(surface.commit(), material, parent)


func _build_body() -> void:
	# Tank shoulder, pinched knee pocket and raised narrow tail.
	_body_panel(
		[
			Vector4(-0.48, 0.85, 0.10, 0.07),
			Vector4(-0.29, 0.89, 0.235, 0.14),
			Vector4(-0.01, 0.88, 0.22, 0.135),
			Vector4(0.23, 0.79, 0.13, 0.055)
		],
		_red,
		self
	)
	_body_panel(
		[
			Vector4(0.11, 0.822, 0.12, 0.025),
			Vector4(0.40, 0.836, 0.15, 0.022),
			Vector4(0.61, 0.90, 0.113, 0.028)
		],
		_rubber,
		self
	)
	_body_panel(
		[
			Vector4(0.39, 0.80, 0.125, 0.055),
			Vector4(0.71, 0.915, 0.125, 0.044),
			Vector4(0.91, 0.94, 0.054, 0.023)
		],
		_red,
		self
	)
	_body_panel(
		[
			Vector4(-0.46, 0.45, 0.17, 0.15),
			Vector4(-0.16, 0.42, 0.225, 0.18),
			Vector4(0.21, 0.47, 0.19, 0.14)
		],
		_black,
		self
	)
	_body_panel(
		[
			Vector4(-0.48, 0.285, 0.09, 0.045),
			Vector4(-0.05, 0.235, 0.17, 0.05),
			Vector4(0.29, 0.29, 0.12, 0.03)
		],
		_metal,
		self
	)
	# Broad radiator with visible closely spaced cooling fins.
	_box(Vector3(0.35, 0.28, 0.075), Vector3(0, 0.58, -0.40), _black, self)
	for i in range(15):
		_box(Vector3(0.32, 0.006, 0.007), Vector3(0, 0.46 + i * 0.017, -0.442), _metal, self)
	for side in [-1.0, 1.0]:
		var panel := Node3D.new()
		panel.position.x = side * 0.16
		add_child(panel)
		_body_panel(
			[
				Vector4(-0.41, 0.69, 0.026, 0.115),
				Vector4(-0.20, 0.71, 0.055, 0.105),
				Vector4(0.02, 0.70, 0.018, 0.025)
			],
			_red,
			panel
		)
		# Two small aerodynamic planes on each side.
		for height in [0.66, 0.72]:
			var wing := _box(
				Vector3(0.14, 0.014, 0.18), Vector3(side * 0.265, height, -0.34), _black, self
			)
			wing.rotation.z = side * -0.10
		# V4 cylinder banks and circular engine covers.
		_lathe(
			[
				Vector2(-0.015, 0.0),
				Vector2(-0.015, 0.098),
				Vector2(0.015, 0.098),
				Vector2(0.021, 0.081),
				Vector2(0.021, 0)
			],
			_gold,
			self,
			Vector3(side * 0.218, 0.43, 0.05)
		)
		for z in [-0.18, 0.08]:
			for i in range(5):
				_box(
					Vector3(0.09, 0.009, 0.12),
					Vector3(side * 0.19, 0.57 + i * 0.016, z),
					_metal,
					self
				)
	# Flush fuel cap.
	_bar(Vector3(0, 1.031, -0.15), Vector3(0, 1.035, -0.15), 0.046, _metal, self)
	_bar(Vector3(0, 1.035, -0.15), Vector3(0, 1.036, -0.15), 0.033, _black, self)
	_box(Vector3(0.017, 0.002, 0.028), Vector3(0, 1.038, -0.15), _metal, self)
	for i in range(6):
		var angle := float(i) * TAU / 6
		var center := Vector3(sin(angle) * 0.039, 1.036, -0.15 + cos(angle) * 0.039)
		_bar(center, center + Vector3.UP * 0.001, 0.0028, _black, self)
	var tail_light := _material(Color("ff3030"), 0, 0.25)
	tail_light.emission_enabled = true
	tail_light.emission = Color("e71922")
	_box(Vector3(0.09, 0.013, 0.014), Vector3(0, 0.944, 0.925), tail_light, self)


func _build_front() -> void:
	for side in [-1.0, 1.0]:
		_bar(Vector3(side * 0.094, 0, 0), Vector3(side * 0.094, 0.35, 0.15), 0.022, _metal, _front)
		_bar(
			Vector3(side * 0.094, 0.24, 0.104),
			Vector3(side * 0.094, 0.60, 0.258),
			0.029,
			_gold,
			_front
		)
		_box(Vector3(0.045, 0.10, 0.045), Vector3(side * 0.086, 0.035, 0.13), _gold, _front)
		# Dark handlebar finish observed in the supplied onboard footage.
		_bar(Vector3(0, 0.69, 0.27), Vector3(side * 0.30, 0.715, 0.25), 0.014, _black, _front)
		_bar(
			Vector3(side * 0.30, 0.715, 0.25),
			Vector3(side * 0.40, 0.70, 0.27),
			0.021,
			_rubber,
			_front
		)
		_bar(
			Vector3(side * 0.27, 0.705, 0.21),
			Vector3(side * 0.40, 0.69, 0.20),
			0.007,
			_metal,
			_front
		)
	_body_panel(
		[
			Vector4(-0.18, 0.29, 0.05, 0.014),
			Vector4(-0.04, 0.335, 0.075, 0.025),
			Vector4(0.18, 0.29, 0.05, 0.017)
		],
		_red,
		_front
	)
	_body_panel(
		[
			Vector4(-0.08, 0.53, 0.08, 0.025),
			Vector4(0.02, 0.58, 0.16, 0.07),
			Vector4(0.13, 0.59, 0.12, 0.08)
		],
		_black,
		_front
	)
	var light := _material(Color("c4ebef"), 0.2, 0.15)
	light.emission_enabled = true
	light.emission = Color("83b9c5")
	for side in [-1.0, 1.0]:
		var lamp := _box(
			Vector3(0.104, 0.015, 0.008), Vector3(side * 0.072, 0.58, -0.057), light, _front
		)
		lamp.rotation.z = side * -0.22
	_build_cockpit()


func _beveled_panel(
	size: Vector2, depth: float, bevel: float, material: Material, parent: Node3D
) -> void:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	# A reflective flat face must not inherit normals from its housing edges.
	surface.set_smooth_group(-1)
	var half_size := size * 0.5
	var outline: Array[Vector2] = [
		Vector2(-half_size.x + bevel, -half_size.y),
		Vector2(half_size.x - bevel, -half_size.y),
		Vector2(half_size.x, -half_size.y + bevel),
		Vector2(half_size.x, half_size.y - bevel),
		Vector2(half_size.x - bevel, half_size.y),
		Vector2(-half_size.x + bevel, half_size.y),
		Vector2(-half_size.x, half_size.y - bevel),
		Vector2(-half_size.x, -half_size.y + bevel)
	]
	# Clockwise front winding gives the glass face a normal toward the rider (+Z).
	outline.reverse()
	for i in range(outline.size()):
		var a := outline[i]
		var b := outline[(i + 1) % outline.size()]
		_triangle(
			surface,
			Vector3(0, 0, depth * 0.5),
			Vector3(a.x, a.y, depth * 0.5),
			Vector3(b.x, b.y, depth * 0.5)
		)
		_triangle(
			surface,
			Vector3(a.x, a.y, -depth * 0.5),
			Vector3(b.x, b.y, -depth * 0.5),
			Vector3(b.x, b.y, depth * 0.5)
		)
		_triangle(
			surface,
			Vector3(a.x, a.y, -depth * 0.5),
			Vector3(b.x, b.y, depth * 0.5),
			Vector3(a.x, a.y, depth * 0.5)
		)
	surface.generate_normals()
	_mesh(surface.commit(), material, parent)


func _hose(points: Array[Vector3], radius: float, material: Material) -> void:
	var curve := Curve3D.new()
	for i in range(points.size()):
		var previous := points[maxi(0, i - 1)]
		var next := points[mini(points.size() - 1, i + 1)]
		var tangent := (next - previous) * 0.16
		curve.add_point(points[i], -tangent, tangent)
	var vertices := curve.tessellate(2, 12)
	for i in range(vertices.size() - 1):
		_bar(vertices[i], vertices[i + 1], radius, material, _front)


## Original molded vessel and screw cap, within the previous cylinder bounds.
## Profile entries are height, outer radius and radial grip recess depth.
## These proportions are visual estimates from footage, not measured parts.
func _reservoir_shell(cap: bool) -> ArrayMesh:
	var profile: Array[Vector3]
	if cap:
		profile = [
			Vector3(-0.004, 0.028, 0),
			Vector3(-0.003, 0.030, 0.0012),
			Vector3(0.0025, 0.030, 0.0012),
			Vector3(0.004, 0.0285, 0)
		]
	else:
		profile = [
			Vector3(-0.0225, 0.024, 0),
			Vector3(-0.020, 0.027, 0),
			Vector3(-0.017, 0.028, 0),
			Vector3(0.017, 0.027, 0),
			Vector3(0.021, 0.024, 0),
			Vector3(0.0225, 0.024, 0)
		]
	var segments := 96 if cap else 64
	var rings: Array[PackedVector3Array] = []
	for section in profile:
		var ring := PackedVector3Array()
		for i in segments:
			var angle := TAU * float(i) / segments
			# Twenty four vertical grip recesses, formed in the cap itself.
			var radius := section.y - section.z * (0.5 + 0.5 * cos(24 * angle))
			ring.append(Vector3(cos(angle) * radius, section.x, sin(angle) * radius))
		rings.append(ring)
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	surface.set_smooth_group(0)
	for i in range(rings.size() - 1):
		for j in segments:
			var next := (j + 1) % segments
			_triangle(surface, rings[i][j], rings[i][next], rings[i + 1][next])
			_triangle(surface, rings[i][j], rings[i + 1][next], rings[i + 1][j])
	# Flat sealed ends share positions with the shell, but not smoothed normals.
	surface.set_smooth_group(-1)
	for j in segments:
		var next := (j + 1) % segments
		_triangle(surface, Vector3(0, profile[0].x, 0), rings[0][next], rings[0][j])
		_triangle(surface, Vector3(0, profile[-1].x, 0), rings[-1][j], rings[-1][next])
	surface.index()
	surface.generate_normals()
	return surface.commit()


func _build_cockpit() -> void:
	# Proportions and silhouette are guided by the provided Ken Moto cockpit
	# footage. This is original geometry and an original functional display,
	# not a reproduction of Ducati graphics or proprietary instrument firmware.
	var polymer := ShaderMaterial.new()
	polymer.shader = preload("res://shaders/cockpit_finish.gdshader")
	var dark_metal := ShaderMaterial.new()
	dark_metal.shader = preload("res://shaders/cockpit_finish.gdshader")
	dark_metal.set_shader_parameter("base_color", Color("353b3f"))
	dark_metal.set_shader_parameter("metalness", 0.85)
	dark_metal.set_shader_parameter("finish_roughness", 0.28)
	dark_metal.set_shader_parameter("grain_pitch_m", 0.00035)
	dark_metal.set_shader_parameter("relief_m", 0.000004)
	var markings := _material(Color("c6c9bd"), 0.1, 0.5)
	var accent := _material(Color("9f1822"), 0.05, 0.47)
	_display_viewport = SubViewport.new()
	_display_viewport.name = "InstrumentTexture"
	_display_viewport.size = Vector2i(1024, 560)
	_display_viewport.disable_3d = true
	_display_viewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	add_child(_display_viewport)
	_display = preload("res://scripts/instrument_display.gd").new()
	_display_viewport.add_child(_display)
	var display_material := ShaderMaterial.new()
	display_material.shader = preload("res://shaders/instrument_screen.gdshader")
	display_material.set_shader_parameter("display_texture", _display_viewport.get_texture())
	# Forged top yoke and risers, with visible fasteners and fork adjustment caps.
	_chamfered_box(Vector3(0.245, 0.028, 0.084), Vector3(0, 0.59, 0.245), 0.005, dark_metal, _front)
	for side in [-1.0, 1.0]:
		_bar(
			Vector3(side * 0.094, 0.595, 0.25),
			Vector3(side * 0.094, 0.611, 0.25),
			0.026,
			_gold,
			_front
		)
		_bar(
			Vector3(side * 0.094, 0.611, 0.25),
			Vector3(side * 0.094, 0.615, 0.25),
			0.012,
			_metal,
			_front
		)
		_chamfered_box(
			Vector3(0.036, 0.072, 0.043),
			Vector3(side * 0.045, 0.647, 0.27),
			0.004,
			dark_metal,
			_front
		)
		_chamfered_box(
			Vector3(0.036, 0.026, 0.05),
			Vector3(side * 0.045, 0.70, 0.27),
			0.004,
			dark_metal,
			_front
		)
		for z in [0.253, 0.286]:
			_bar(
				Vector3(side * 0.045, 0.713, z),
				Vector3(side * 0.045, 0.717, z),
				0.004,
				_metal,
				_front
			)
			var socket := CylinderMesh.new()
			socket.top_radius = 0.0022
			socket.bottom_radius = 0.0022
			socket.height = 0.0002
			socket.radial_segments = 6
			_mesh(socket, polymer, _front, Vector3(side * 0.045, 0.7171, z))
		# Circumferential grip grooves catch light without noisy normal textures.
		for i in range(13):
			var x: float = side * (0.305 + i * 0.0071)
			var y: float = 0.715 - float(i) * 0.00105
			var z: float = 0.25 + float(i) * 0.0014
			_lathe(
				[
					Vector2(-0.0009, 0.020),
					Vector2(-0.0009, 0.0218),
					Vector2(0.0009, 0.0218),
					Vector2(0.0009, 0.020)
				],
				polymer,
				_front,
				Vector3(x, y, z)
			)
		_bar(
			Vector3(side * 0.40, 0.70, 0.27),
			Vector3(side * 0.425, 0.696, 0.275),
			0.019,
			dark_metal,
			_front
		)
		var switchgear := Node3D.new()
		switchgear.position = Vector3(side * 0.278, 0.717, 0.25)
		switchgear.rotation.x = -0.3
		_front.add_child(switchgear)
		_beveled_panel(Vector2(0.047, 0.055), 0.047, 0.01, polymer, switchgear)
		_box(
			Vector3(0.023, 0.010, 0.006),
			Vector3(0, 0.009, 0.027),
			accent if side > 0 else markings,
			switchgear
		)
		_box(Vector3(0.014, 0.008, 0.006), Vector3(0, -0.009, 0.027), dark_metal, switchgear)
		# Amber reservoir tint and transmitted background follow the footage.
		# Screen transmission and analytic absorption omit refraction and slosh.
		var fluid := ShaderMaterial.new()
		fluid.shader = preload("res://shaders/reservoir.gdshader")
		# The footage shows raised pots on stalks, above the bar and display sides.
		# Mount height is an artistic estimate, not a manufacturer measurement.
		var reservoir_base := 0.775
		_bar(
			Vector3(side * 0.219, 0.70, 0.205),
			Vector3(side * 0.219, reservoir_base, 0.205),
			0.005,
			dark_metal,
			_front
		)
		# Outlet and feed hose connect the vessel to the master cylinder.
		_bar(
			Vector3(side * 0.235, reservoir_base, 0.205),
			Vector3(side * 0.235, reservoir_base - 0.012, 0.205),
			0.007,
			polymer,
			_front
		)
		_hose(
			[
				Vector3(side * 0.235, reservoir_base - 0.01, 0.205),
				Vector3(side * 0.242, 0.743, 0.214),
				Vector3(side * 0.253, 0.720, 0.218)
			],
			0.004,
			polymer
		)
		_mesh(
			_reservoir_shell(false),
			fluid,
			_front,
			Vector3(side * 0.219, reservoir_base + 0.0225, 0.205)
		)
		_mesh(
			_reservoir_shell(true),
			polymer,
			_front,
			Vector3(side * 0.219, reservoir_base + 0.049, 0.205)
		)
		_bar(
			Vector3(side * 0.252, 0.713, 0.211),
			Vector3(side * 0.288, 0.704, 0.186),
			0.009,
			dark_metal,
			_front
		)
		_bar(
			Vector3(side * 0.288, 0.704, 0.186),
			Vector3(side * 0.372, 0.687, 0.167),
			0.006,
			dark_metal,
			_front
		)
		_bar(
			Vector3(side * 0.372, 0.687, 0.167),
			Vector3(side * 0.393, 0.69, 0.18),
			0.007,
			dark_metal,
			_front
		)
		_hose(
			[
				Vector3(side * 0.23, 0.704, 0.215),
				Vector3(side * 0.20, 0.64, 0.10),
				Vector3(side * 0.12, 0.53, 0.12),
				Vector3(side * 0.11, 0.20, 0.06)
			],
			0.0035,
			polymer
		)
	# Display face normal points upward and toward the rider, not skyward.
	var instruments := Node3D.new()
	instruments.name = "LiveInstrumentCluster"
	instruments.position = Vector3(0, 0.745, 0.12)
	instruments.rotation.x = -0.55
	_front.add_child(instruments)
	_beveled_panel(Vector2(0.196, 0.119), 0.025, 0.015, polymer, instruments)
	var face := Node3D.new()
	face.position.z = 0.013
	instruments.add_child(face)
	_beveled_panel(Vector2(0.177, 0.097), 0.001, 0.008, display_material, face)
	update_instruments(0, 1500, 1)


func _build_chassis() -> void:
	for side in [-1.0, 1.0]:
		# Symmetrical swingarm follows the selected current generation reference.
		_bar(
			Vector3(side * 0.14, 0.46, 0.17),
			Vector3(side * 0.14, 0.3359, 0.75),
			0.039,
			_black,
			self
		)
		_bar(
			Vector3(side * 0.15, 0.56, 0.21),
			Vector3(side * 0.14, 0.3359, 0.75),
			0.021,
			_black,
			self
		)
		_bar(
			Vector3(side * 0.11, 0.75, -0.40), Vector3(side * 0.19, 0.55, 0.23), 0.024, _metal, self
		)
		_bar(
			Vector3(side * 0.15, 0.66, 0.16), Vector3(side * 0.085, 0.87, 0.76), 0.019, _black, self
		)
		_bar(
			Vector3(side * 0.17, 0.39, 0.24), Vector3(side * 0.28, 0.39, 0.24), 0.016, _metal, self
		)
		# Headers descend from the exposed engine into short underbody exhausts.
		_bar(
			Vector3(side * 0.10, 0.52, -0.29), Vector3(side * 0.12, 0.28, -0.35), 0.022, _gold, self
		)
		_bar(
			Vector3(side * 0.12, 0.28, -0.35), Vector3(side * 0.18, 0.22, 0.10), 0.022, _gold, self
		)
		_bar(
			Vector3(side * 0.18, 0.22, 0.10), Vector3(side * 0.23, 0.27, 0.43), 0.045, _metal, self
		)
		_bar(
			Vector3(side * 0.23, 0.27, 0.43),
			Vector3(side * 0.235, 0.275, 0.451),
			0.032,
			_black,
			self
		)
	# Rear shock and exposed spring coils.
	_bar(Vector3(0.02, 0.43, 0.42), Vector3(0.02, 0.69, 0.24), 0.025, _gold, self)
	for i in range(9):
		var t := float(i) / 8
		var center := Vector3(0.02, 0.44, 0.41).lerp(Vector3(0.02, 0.67, 0.25), t)
		var coil := TorusMesh.new()
		coil.inner_radius = 0.025
		coil.outer_radius = 0.033
		coil.rings = 12
		coil.ring_segments = 6
		var instance := _mesh(coil, _gold, self, center)
		instance.rotation.x = -0.60


func _rider_limb(
	a: Vector3, b: Vector3, radius: float, material: Material, width_scale := 1.0
) -> MeshInstance3D:
	var capsule := CapsuleMesh.new()
	capsule.radius = radius
	capsule.height = a.distance_to(b) + radius * 1.1
	capsule.radial_segments = 16
	capsule.rings = 6
	var instance := _mesh(capsule, material, rider, (a + b) * 0.5)
	var axis := (b - a).normalized()
	var tangent := Vector3.RIGHT
	if absf(axis.dot(tangent)) > 0.95:
		tangent = Vector3.FORWARD
	var side := axis.cross(tangent).normalized()
	instance.basis = Basis(side * width_scale, axis, side.cross(axis).normalized())
	return instance


func _rider_ellipsoid(at: Vector3, dimensions: Vector3, material: Material) -> MeshInstance3D:
	var sphere := SphereMesh.new()
	sphere.radius = 0.5
	sphere.height = 1
	sphere.radial_segments = 24
	sphere.rings = 12
	var instance := _mesh(sphere, material, rider, at)
	instance.scale = dimensions
	return instance


func _build_rider() -> void:
	# Original neutral adult silhouette. Pose dimensions are artistic estimates,
	# not a measured rider anthropometry or a change to simulation mass.
	rider = Node3D.new()
	rider.name = "OriginalPosedRider"
	rider.visible = _rider_visible
	add_child(rider)
	var leather := _material(Color("24282a"), 0.05, 0.72)
	var panels := _material(Color("b7b5aa"), 0.03, 0.66)
	var armor := _material(Color("101518"), 0.12, 0.48)
	var helmet := _material(Color("d9d7cb"), 0.18, 0.24)
	var visor := _material(Color("162a32"), 0.65, 0.14)
	_rider_ellipsoid(Vector3(0, 0.965, 0.32), Vector3(0.33, 0.25, 0.30), leather)
	# The jacket leans forward from the saddle, with a compact back protector.
	_rider_limb(Vector3(0, 1.01, 0.30), Vector3(0, 1.29, -0.11), 0.15, leather, 1.15)
	var back := _rider_ellipsoid(Vector3(0, 1.205, 0.18), Vector3(0.27, 0.12, 0.37), armor)
	back.rotation.x = -0.60
	_rider_limb(Vector3(0, 1.30, -0.13), Vector3(0, 1.40, -0.24), 0.059, armor)
	# Full face helmet and original curved visor, all unbranded.
	var head := _rider_ellipsoid(Vector3(0, 1.485, -0.29), Vector3(0.255, 0.29, 0.295), helmet)
	head.rotation.x = 0.10
	_rider_ellipsoid(Vector3(0, 1.416, -0.345), Vector3(0.225, 0.12, 0.25), helmet)
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(16):
		var angle_a := lerpf(-1.18, 1.18, float(i) / 16)
		var angle_b := lerpf(-1.18, 1.18, float(i + 1) / 16)
		var a := Vector3(sin(angle_a) * 0.129, 1.49, -0.30 - cos(angle_a) * 0.148)
		var b := Vector3(sin(angle_b) * 0.129, 1.49, -0.30 - cos(angle_b) * 0.148)
		var c := Vector3(sin(angle_b) * 0.122, 1.553, -0.30 - cos(angle_b) * 0.134)
		var d := Vector3(sin(angle_a) * 0.122, 1.553, -0.30 - cos(angle_a) * 0.134)
		_triangle(surface, a, b, c)
		_triangle(surface, a, c, d)
	surface.generate_normals()
	_mesh(surface.commit(), visor, rider)
	for side in [-1.0, 1.0]:
		var shoulder := Vector3(side * 0.185, 1.285, -0.12)
		_rider_ellipsoid(shoulder, Vector3(0.135, 0.105, 0.13), panels)
		var hip := Vector3(side * 0.125, 0.948, 0.31)
		var knee := Vector3(side * 0.247, 0.675, -0.075)
		var ankle := Vector3(side * 0.253, 0.43, 0.28)
		_rider_limb(hip, knee, 0.083, leather)
		_rider_limb(knee, ankle, 0.059, leather)
		_rider_ellipsoid(knee + Vector3(side * 0.025, 0, 0), Vector3(0.095, 0.14, 0.12), panels)
		_rider_limb(ankle, Vector3(side * 0.256, 0.405, 0.11), 0.055, armor)
		_rider_ellipsoid(Vector3(side * 0.265, 0.47, 0.235), Vector3(0.095, 0.14, 0.11), armor)

	_arms = preload("res://scripts/rider_arm_visual.gd").new()
	_arms.name = "ArticulatedRiderArms"
	add_child(_arms)
	var arm_error: String = _arms.build(_front.position)
	assert(arm_error.is_empty(), arm_error)
	_arms.visible = _rider_visible
