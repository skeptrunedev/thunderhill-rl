extends Node3D
## Original angular lamp reconstruction from manufacturer front photographs.
## Dimensions, rear enclosure and mounting details are artistic estimates.


func _ready() -> void:
	var polymer := ShaderMaterial.new()
	polymer.shader = preload("res://shaders/cockpit_finish.gdshader")
	var red := StandardMaterial3D.new()
	red.albedo_color = Color("bc101b")
	red.roughness = 0.20
	red.clearcoat_enabled = true
	red.clearcoat = 0.9
	red.clearcoat_roughness = 0.08
	var lens := StandardMaterial3D.new()
	lens.albedo_color = Color("131a20")
	lens.roughness = 0.12
	lens.clearcoat_enabled = true
	lens.clearcoat = 1.0
	var reflector := StandardMaterial3D.new()
	reflector.albedo_color = Color("657079")
	reflector.metallic = 1.0
	reflector.roughness = 0.20
	var lamp := StandardMaterial3D.new()
	lamp.albedo_color = Color("dce8e9")
	lamp.emission_enabled = true
	lamp.emission = Color("cbdde0")
	lamp.roughness = 0.24
	_add_mesh(_shell(), polymer, "HeadlightShell")
	_add_mesh(
		_panel(
			[
				Vector2(-0.145, 0.617),
				Vector2(-0.110, 0.648),
				Vector2(-0.065, 0.641),
				Vector2(-0.032, 0.610),
				Vector2(0.032, 0.610),
				Vector2(0.065, 0.641),
				Vector2(0.110, 0.648),
				Vector2(0.145, 0.617),
				Vector2(0.125, 0.585),
				Vector2(0.082, 0.557),
				Vector2(0, 0.519),
				Vector2(-0.082, 0.557),
				Vector2(-0.125, 0.585)
			],
			-0.094,
			0.015
		),
		red,
		"PaintedLampSurround"
	)
	for side in [-1.0, 1.0]:
		var outline: Array[Vector2] = []
		for point in [
			Vector2(0.145, 0.583),
			Vector2(0.018, 0.518),
			Vector2(0.029, 0.475),
			Vector2(0.084, 0.473),
			Vector2(0.133, 0.513)
		]:
			outline.append(Vector2(point.x * side, point.y))
		_add_mesh(_panel(outline, -0.092, 0.013), lens, "LampLens")
		var insert: Array[Vector2] = []
		for point in [
			Vector2(0.111, 0.548),
			Vector2(0.049, 0.518),
			Vector2(0.056, 0.491),
			Vector2(0.087, 0.492),
			Vector2(0.111, 0.515)
		]:
			insert.append(Vector2(point.x * side, point.y))
		_add_mesh(_panel(insert, -0.095, 0.004), reflector, "OpticReflector")
		var optic := SphereMesh.new()
		optic.radius = 0.018
		optic.height = 0.036
		optic.radial_segments = 24
		optic.rings = 12
		var glass := _add_mesh(optic, lens, "ProjectorLens")
		glass.scale.z = 0.28
		glass.position = Vector3(side * 0.078, 0.515, -0.096)
		var strip: Array[Vector2] = []
		for point in [
			Vector2(0.144, 0.590),
			Vector2(0.014, 0.524),
			Vector2(0.017, 0.518),
			Vector2(0.146, 0.583)
		]:
			strip.append(Vector2(point.x * side, point.y))
		_add_mesh(_panel(strip, -0.099, 0.007), lamp, "RunningLight")
		# Short enclosed mounting arms connect the housing to the fork bridge.
		var support := BoxMesh.new()
		support.size = Vector3(0.014, 0.025, 0.11)
		_add_mesh(support, polymer, "LampMount").position = Vector3(side * 0.08, 0.555, 0.175)


func _add_mesh(mesh: Mesh, material: Material, label: String) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	node.name = label
	node.mesh = mesh
	node.material_override = material
	add_child(node)
	return node


func _shell() -> ArrayMesh:
	var outline: Array[Vector2] = [
		Vector2(-0.75, 1),
		Vector2(0.75, 1),
		Vector2(1, 0.55),
		Vector2(0.93, -0.35),
		Vector2(0.55, -0.92),
		Vector2(0, -1),
		Vector2(-0.55, -0.92),
		Vector2(-0.93, -0.35),
		Vector2(-1, 0.55)
	]
	var profiles: Array[Vector4] = [
		Vector4(-0.080, 0.560, 0.150, 0.095),
		Vector4(-0.072, 0.560, 0.157, 0.100),
		Vector4(0.020, 0.560, 0.155, 0.092),
		Vector4(0.085, 0.540, 0.105, 0.055),
		Vector4(0.126, 0.540, 0.075, 0.038),
		Vector4(0.130, 0.540, 0.070, 0.033)
	]
	var rings: Array[PackedVector3Array] = []
	for profile in profiles:
		var ring := PackedVector3Array()
		for point in outline:
			ring.append(Vector3(point.x * profile.z, profile.y + point.y * profile.w, profile.x))
		rings.append(ring)
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	for ring in range(rings.size() - 1):
		for i in outline.size():
			var next := (i + 1) % outline.size()
			var radial := (outline[i] + outline[next]) * 0.5
			_face(
				surface,
				[rings[ring][i], rings[ring][next], rings[ring + 1][next], rings[ring + 1][i]],
				Vector3(radial.x, radial.y, 0)
			)
	for end in [0, rings.size() - 1]:
		var center := Vector3(0, profiles[end].y, profiles[end].x)
		var normal := Vector3.FORWARD if end == 0 else Vector3.BACK
		for i in outline.size():
			_face(surface, [center, rings[end][i], rings[end][(i + 1) % outline.size()]], normal)
	surface.index()
	return surface.commit()


func _panel(points: Array[Vector2], front: float, depth: float) -> ArrayMesh:
	var outline := PackedVector2Array(points)
	var area := 0.0
	for i in outline.size():
		area += outline[i].cross(outline[(i + 1) % outline.size()])
	if area < 0:
		outline.reverse()
	var triangles := Geometry2D.triangulate_polygon(outline)
	assert(not triangles.is_empty(), "Invalid lamp panel outline")
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	for side in [0, 1]:
		var z: float = front + depth * float(side)
		for i in range(0, triangles.size(), 3):
			var face: Array[Vector3] = []
			for j in range(3):
				var point := outline[triangles[i + j]]
				face.append(Vector3(point.x, point.y, z))
			_face(surface, face, Vector3.FORWARD if side == 0 else Vector3.BACK)
	for i in outline.size():
		var a := outline[i]
		var b := outline[(i + 1) % outline.size()]
		var delta := b - a
		_face(
			surface,
			[
				Vector3(a.x, a.y, front),
				Vector3(b.x, b.y, front),
				Vector3(b.x, b.y, front + depth),
				Vector3(a.x, a.y, front + depth)
			],
			Vector3(delta.y, -delta.x, 0)
		)
	surface.index()
	return surface.commit()


func _face(surface: SurfaceTool, points: Array[Vector3], outward: Vector3) -> void:
	# Match the existing chamfered panel builder's clockwise front convention.
	if (points[1] - points[0]).cross(points[2] - points[0]).dot(outward) > 0:
		points.reverse()
	for i in range(1, points.size() - 1):
		var normal := -(points[i] - points[0]).cross(points[i + 1] - points[0]).normalized()
		for point in [points[0], points[i], points[i + 1]]:
			surface.set_normal(normal)
			surface.add_vertex(point)
