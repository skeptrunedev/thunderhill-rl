extends RefCounted
## Original unbranded glove study fitted to the existing 21 mm grip radius.
## All anatomy, tailoring and armor dimensions are artistic estimates.
## Local X follows the grip; Y is up; positive Z points toward the wrist.


static func build(side: float) -> ArrayMesh:
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	var leather := Color("24282a")
	var armor := Color("121719")
	var panel := Color("b84324")
	# A tailored palm and cuff shell, rather than intersecting spherical pads.
	_palm(surface, leather)
	_ellipsoid(
		surface, Vector3(0, 0.047, 0.014), Vector3(0.068, 0.012, 0.036), armor, Basis.IDENTITY, 1.0
	)
	for finger in range(4):
		var x := side * (-0.028 + finger * 0.018)
		_finger(surface, x, 0.0075 if finger == 3 else 0.0085, leather)
	# Thumb approaches from the inside of the grip and closes below the palm.
	_segment(
		surface,
		Vector3(-side * 0.036, 0.012, 0.045),
		Vector3(-side * 0.050, -0.015, 0.021),
		0.012,
		leather
	)
	_segment(
		surface,
		Vector3(-side * 0.050, -0.015, 0.021),
		Vector3(-side * 0.025, -0.029, 0.006),
		0.010,
		leather
	)
	# Restrained original contrasting cuff panel; no copied logos or branding.
	_ellipsoid(surface, Vector3(0, 0.040, 0.086), Vector3(0.051, 0.003, 0.018), panel)
	surface.generate_normals()
	return surface.commit()


static func _segment(
	surface: SurfaceTool, a: Vector3, b: Vector3, radius: float, color: Color
) -> void:
	var axis := (b - a).normalized()
	var across := axis.cross(Vector3.RIGHT).normalized()
	if across.length_squared() < 0.5:
		across = axis.cross(Vector3.FORWARD).normalized()
	var basis := Basis(across, axis, across.cross(axis).normalized())
	_ellipsoid(
		surface,
		(a + b) * 0.5,
		Vector3(radius * 2, a.distance_to(b) + radius * 1.8, radius * 2),
		color,
		basis
	)


static func _ellipsoid(
	surface: SurfaceTool,
	at: Vector3,
	size: Vector3,
	color: Color,
	rotation := Basis.IDENTITY,
	armor := 0.0
) -> void:
	var sphere := SphereMesh.new()
	sphere.radius = 0.5
	sphere.height = 1.0
	sphere.radial_segments = 16
	sphere.rings = 8
	var arrays := sphere.get_mesh_arrays()
	var points: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
	var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
	var basis := rotation * Basis.from_scale(size)
	var normal_basis := basis.inverse().transposed()
	for index in indices:
		surface.set_color(color)
		surface.set_uv2(Vector2(armor, 0.0))
		surface.set_normal((normal_basis * normals[index]).normalized())
		surface.add_vertex(at + basis * points[index])


static func _palm(surface: SurfaceTool, color: Color) -> void:
	# Each row: z, half width, bottom and top. The front clears the grip;
	# the wrist tapers before the broader gauntlet opening.
	var sections := [
		Vector4(-0.025, 0.028, 0.025, 0.038),
		Vector4(-0.010, 0.039, 0.024, 0.048),
		Vector4(0.018, 0.043, 0.019, 0.050),
		Vector4(0.044, 0.038, 0.006, 0.045),
		Vector4(0.066, 0.032, -0.006, 0.035),
		Vector4(0.086, 0.035, -0.012, 0.040),
		Vector4(0.115, 0.039, -0.014, 0.043),
	]
	var rings: Array[PackedVector3Array] = []
	for section in sections:
		var ring := PackedVector3Array()
		for i in range(32):
			var angle := TAU * i / 32.0
			var sx := sin(angle)
			var sy := cos(angle)
			ring.append(
				Vector3(
					signf(sx) * pow(absf(sx), 0.65) * section.y,
					(
						(section.z + section.w) * 0.5
						+ signf(sy) * pow(absf(sy), 0.65) * (section.w - section.z) * 0.5
					),
					section.x
				)
			)
		rings.append(ring)
	for j in range(rings.size() - 1):
		for i in range(32):
			var k := (i + 1) % 32
			_quad(surface, rings[j][i], rings[j][k], rings[j + 1][k], rings[j + 1][i], color, 1.0)
	# Rounded glove nose closes the palm; cuff stays open for the future sleeve.
	var center := Vector3(0, 0.0315, -0.025)
	for i in range(32):
		_triangle(surface, center, rings[0][(i + 1) % 32], rings[0][i], color, 1.0)


static func _finger(surface: SurfaceTool, x: float, radius: float, color: Color) -> void:
	var rings: Array[PackedVector3Array] = []
	for step in range(25):
		var angle := lerpf(deg_to_rad(-30.0), deg_to_rad(205.0), step / 24.0)
		var radial := Vector3(0, cos(angle), -sin(angle))
		var center := Vector3(x, 0, 0) + radial * 0.030
		var ring := PackedVector3Array()
		for i in range(12):
			var around := TAU * i / 12.0
			ring.append(center + radius * (Vector3.RIGHT * cos(around) + radial * sin(around)))
		rings.append(ring)
	for j in range(rings.size() - 1):
		for i in range(12):
			var k := (i + 1) % 12
			_quad(surface, rings[j][i], rings[j][k], rings[j + 1][k], rings[j + 1][i], color)
	for step in [0, 24]:
		var angle := lerpf(deg_to_rad(-30.0), deg_to_rad(205.0), step / 24.0)
		_ellipsoid(
			surface,
			Vector3(x, cos(angle) * 0.030, -sin(angle) * 0.030),
			Vector3.ONE * radius * 2,
			color
		)


static func _quad(
	surface: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, d: Vector3, color: Color, palm := 0.0
) -> void:
	_triangle(surface, a, b, c, color, palm)
	_triangle(surface, a, c, d, color, palm)


static func _triangle(
	surface: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, color: Color, palm := 0.0
) -> void:
	var normal := (c - a).cross(b - a).normalized()
	for point in [a, b, c]:
		surface.set_color(color)
		surface.set_uv2(Vector2(0.0, palm))
		surface.set_normal(normal)
		surface.add_vertex(point)
