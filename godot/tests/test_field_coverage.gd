extends SceneTree
const Coverage = preload("res://scripts/field_coverage.gd")
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var coverage := Coverage.new()
	var row := {"points_local_xz": [[0, 0], [10, 0], [10, 10]], "width_m": 4.0, "feather_m": 1.0}
	var data := {"schema_version": 1, "corridors": [row]}
	check(coverage.configure(data).is_empty(), "Valid corridor rejected")
	check(is_equal_approx(coverage.signed_distance(Vector2(5, 0), 0), -2.0), "Wrong interior width")
	check(is_equal_approx(coverage.signed_distance(Vector2(-3, 0), 0), 1.0), "Wrong endpoint cap")
	check(coverage.excludes_stubble(Vector2(5, 2.8), 0.9), "Clump footprint can enter cleared core")
	check(not coverage.excludes_stubble(Vector2(5, 3), 0.9), "Grass outside footprint excluded")
	check(coverage.excludes_stubble(Vector2(10, 5), 0.9), "Second segment ignored")
	var material := ShaderMaterial.new()
	material.shader = load("res://shaders/terrain.gdshader")
	coverage.apply_material(material)
	check(
		material.get_shader_parameter("field_segment_count") == 2, "Shader segment count mismatch"
	)
	var segments: PackedVector4Array = material.get_shader_parameter("field_segments")
	var sizes: PackedVector2Array = material.get_shader_parameter("field_sizes")
	check(
		segments.size() == Coverage.MAX_SEGMENTS and segments[1] == Vector4(10, 0, 10, 10),
		"Shader endpoints mismatch"
	)
	check(sizes[0] == Vector2(2, 1), "Shader width and feather mismatch")
	check(
		coverage.tints[0] == Vector3.ONE and coverage.grass_retentions[0] == 0.0,
		"Legacy cleared material changed"
	)
	row.tint_linear = [0.55, 0.5, 0.45]
	row.grass_retention = 1.0
	check(coverage.configure(data).is_empty(), "Flattened straw material rejected")
	coverage.apply_material(material)
	var tints: PackedVector3Array = material.get_shader_parameter("field_tints")
	var retentions: PackedFloat32Array = material.get_shader_parameter("field_grass_retentions")
	check(
		tints[1].is_equal_approx(Vector3(0.55, 0.5, 0.45)) and retentions[1] == 1.0,
		"Flattened material not passed to shader"
	)
	row.tint_linear = [NAN, 1.0, 1.0]
	check(
		not coverage.configure(data).is_empty() and coverage.tints.is_empty(),
		"Nonfinite tint retained stale material"
	)
	row.tint_linear = [1.0, 1.0, 1.0]
	row.grass_retention = 1.1
	check(not coverage.configure(data).is_empty(), "Invalid retention accepted")
	row.grass_retention = 0.0
	row.point_widths_m = [0.0, 4.0, 0.0]
	check(coverage.configure(data).is_empty(), "Tapered corridor rejected")
	check(
		is_equal_approx(coverage.signed_distance(Vector2(5, 0), 0), -1.0), "Taper not interpolated"
	)
	check(
		is_equal_approx(coverage.signed_distance(Vector2(-2, 0), 0), 2.0),
		"Taper endpoint has full width cap"
	)
	coverage.apply_material(material)
	var radii: PackedFloat32Array = material.get_shader_parameter("field_end_radii")
	check(radii[0] == 2.0 and radii[1] == 0.0, "Taper endpoints not passed to shader")
	row.point_widths_m = [0.0, INF, 0.0]
	check(not coverage.configure(data).is_empty(), "Nonfinite taper accepted")
	row.erase("point_widths_m")
	row.point_tint_strengths = [0.0, 1.0, 0.25]
	check(coverage.configure(data).is_empty(), "Variable tint rejected")
	coverage.apply_material(material)
	var strengths: PackedVector2Array = material.get_shader_parameter("field_tint_strengths")
	check(
		strengths[0] == Vector2(0, 1) and strengths[1] == Vector2(1, 0.25),
		"Tint breaks not passed to shader"
	)
	row.point_tint_strengths = [0.0]
	check(not coverage.configure(data).is_empty(), "Mismatched tint points accepted")
	row.erase("point_tint_strengths")
	row.width_m = NAN
	check(
		not coverage.configure(data).is_empty() and coverage.segments.is_empty(),
		"Nonfinite width retained stale coverage"
	)
	row.width_m = 4
	row.points_local_xz = [[0, 0], [0, 0]]
	check(not coverage.configure(data).is_empty(), "Degenerate segment accepted")
	row.points_local_xz = [[0, 0], [INF, 2]]
	check(not coverage.configure(data).is_empty(), "Nonfinite coordinate accepted")
	var actual: Variant = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/field-coverage.json")
	)
	check(coverage.configure(actual).is_empty(), "Runtime trace invalid")
	for segment in coverage.segments:
		check(
			coverage.excludes_stubble(Vector2(segment.x, segment.y), 0.9),
			"Traced core contains stubble"
		)
	if DisplayServer.get_name() != "headless":
		var scene: Node = load("res://assets/generated/scenery.scn").instantiate()
		var count := 0
		var intrusions := 0
		for node in scene.get_children():
			if node is MultiMeshInstance3D and str(node.name).begins_with("DryGrass_"):
				for index in node.multimesh.instance_count:
					var position: Vector3 = (
						node.transform * node.multimesh.get_instance_transform(index).origin
					)
					count += 1
					if coverage.excludes_stubble(Vector2(position.x, position.z), 0.899):
						intrusions += 1
		check(count == 96000, "Did not inspect all baked grass instances")
		check(intrusions == 0, "Baked stubble intrudes into cleared corridor")
		print("FIELD_COVERAGE_BAKED instances=", count, " intrusions=", intrusions)
		scene.free()
	print("FIELD_COVERAGE_CHECK failures=", failures)
	quit(failures)
