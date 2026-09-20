class_name BikeVisual
extends Node3D
## Original Streetfighter inspired visual study, not manufacturer CAD.
## All bodywork profiles are artistic approximations. Axle spacing and nominal
## unloaded tire envelopes follow docs/motorcycle-reference.md.

var _front: Node3D
var _rear_wheel: Node3D
var _front_wheel: Node3D
var _red: StandardMaterial3D
var _black: StandardMaterial3D
var _metal: StandardMaterial3D
var _gold: StandardMaterial3D
var _rubber: StandardMaterial3D
var rider: Node3D
var _rider_visible := true


func _ready() -> void:
	_red = _material(Color("bc101b"), 0.48, 0.25)
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


func update_pose(lean: float, steering: float, wheel_rotation: float) -> void:
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
	cylinder.radial_segments = 12
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


## Each ring is (longitudinal z, center height, half width, half height).
## Eight facets create creased, sculpted panels rather than stacked primitives.
func _body_panel(rings: Array[Vector4], material: Material, parent: Node3D) -> void:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	var sections: Array[PackedVector3Array] = []
	for ring in rings:
		var section := PackedVector3Array()
		for point in [
			Vector2(-0.7, 1),
			Vector2(0.7, 1),
			Vector2(1, 0.35),
			Vector2(0.83, -0.65),
			Vector2(0.45, -1),
			Vector2(-0.45, -1),
			Vector2(-0.83, -0.65),
			Vector2(-1, 0.35)
		]:
			section.append(Vector3(point.x * ring.z, ring.y + point.y * ring.w, ring.x))
		sections.append(section)
	for i in range(sections.size() - 1):
		for j in range(8):
			var next := (j + 1) % 8
			_triangle(surface, sections[i][j], sections[i + 1][j], sections[i + 1][next])
			_triangle(surface, sections[i][j], sections[i + 1][next], sections[i][next])
	for end in [0, sections.size() - 1]:
		for j in range(1, 7):
			_triangle(surface, sections[end][0], sections[end][j], sections[end][j + 1])
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
	_bar(Vector3(0, 1.018, -0.15), Vector3(0, 1.022, -0.15), 0.046, _metal, self)
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
		_bar(Vector3(0, 0.69, 0.27), Vector3(side * 0.30, 0.715, 0.25), 0.014, _metal, _front)
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
	var screen := _box(Vector3(0.12, 0.015, 0.075), Vector3(0, 0.695, 0.16), _black, _front)
	screen.rotation.x = -0.30
	var display := _box(Vector3(0.095, 0.003, 0.052), Vector3(0, 0.704, 0.16), light, _front)
	display.rotation.x = -0.30


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
		var elbow := Vector3(side * 0.31, 1.075, -0.18)
		var wrist := Vector3(side * 0.335, 1.01, -0.48)
		_rider_limb(shoulder, elbow, 0.064, leather)
		_rider_limb(elbow, wrist, 0.052, leather)
		_rider_ellipsoid(shoulder, Vector3(0.135, 0.105, 0.13), panels)
		_rider_ellipsoid(elbow, Vector3(0.12, 0.12, 0.12), armor)
		_rider_limb(wrist, Vector3(side * 0.365, 0.995, -0.492), 0.039, armor)
		var hip := Vector3(side * 0.125, 0.948, 0.31)
		var knee := Vector3(side * 0.247, 0.675, -0.075)
		var ankle := Vector3(side * 0.253, 0.43, 0.28)
		_rider_limb(hip, knee, 0.083, leather)
		_rider_limb(knee, ankle, 0.059, leather)
		_rider_ellipsoid(knee + Vector3(side * 0.025, 0, 0), Vector3(0.095, 0.14, 0.12), panels)
		_rider_limb(ankle, Vector3(side * 0.256, 0.405, 0.11), 0.055, armor)
		_rider_ellipsoid(Vector3(side * 0.265, 0.47, 0.235), Vector3(0.095, 0.14, 0.11), armor)
