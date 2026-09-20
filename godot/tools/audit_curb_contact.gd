extends SceneTree
## Measure the current visual/contact discrepancy. This is not a passing physics test.


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var output := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--report="):
			output = arg.trim_prefix("--report=")
	if output.is_empty():
		push_error("Supply --report=/absolute/path/report.json")
		quit(2)
		return
	var track := preload("res://scripts/track.gd").new()
	root.add_child(track)
	if not track.initialization_error.is_empty():
		push_error(track.initialization_error)
		quit(2)
		return
	var mesh_node := track.get_node_or_null("ProvisionalCurbs") as MeshInstance3D
	if mesh_node == null:
		push_error("No rendered curb mesh found")
		quit(2)
		return
	var arrays := mesh_node.mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var indices := PackedInt32Array()
	if arrays[Mesh.ARRAY_INDEX] != null:
		indices = arrays[Mesh.ARRAY_INDEX]
	if indices.is_empty():
		for index in vertices.size():
			indices.append(index)
	var rows: Array[Dictionary] = []
	var errors: Array[float] = []
	var ground_above := 0
	var classified_road := 0
	var max_normal_angle := 0.0
	for index in range(0, indices.size(), 3):
		var a := mesh_node.transform * vertices[indices[index]]
		var b := mesh_node.transform * vertices[indices[index + 1]]
		var c := mesh_node.transform * vertices[indices[index + 2]]
		var normal := (b - a).cross(c - a).normalized()
		if normal.y < 0:
			normal = -normal
		for weight in [
			Vector3.ONE / 3.0,
			Vector3(0.8, 0.1, 0.1),
			Vector3(0.1, 0.8, 0.1),
			Vector3(0.1, 0.1, 0.8)
		]:
			var point: Vector3 = a * weight.x + b * weight.y + c * weight.z
			var contact: Dictionary = track.sample_world(point)
			var ground: Dictionary = track.offroad_surface.sample(point)
			if ground.has("error"):
				push_error(str(ground))
				quit(2)
				return
			var height: float = contact.height
			if not is_finite(height) or not contact.normal.is_finite():
				push_error("Nonfinite curb contact")
				quit(2)
				return
			var error := absf(height - point.y)
			var normal_angle := rad_to_deg(normal.angle_to(contact.normal))
			max_normal_angle = maxf(max_normal_angle, normal_angle)
			var penetration := float(ground.height) - point.y if ground.has("height") else 0.0
			ground_above += int(penetration > 0.0002)
			classified_road += int(contact.on_track)
			errors.append(error)
			rows.append(
				{
					"triangle": index / 3,
					"position": [point.x, point.y, point.z],
					"render_height": point.y,
					"contact_height": height,
					"height_error_m": error,
					"normal_angle_deg": normal_angle,
					"ground_above_curb_m": penetration,
					"on_track": contact.on_track
				}
			)
	errors.sort()
	rows.sort_custom(
		func(a: Dictionary, b: Dictionary) -> bool: return a.height_error_m > b.height_error_m
	)
	var report := {
		"purpose": "Measure known curb discrepancy, not physics acceptance",
		"track_sha256": FileAccess.get_sha256("res://data/track.json"),
		"track_script_sha256": FileAccess.get_sha256("res://scripts/track.gd"),
		"samples": errors.size(),
		"max_height_error_m": errors[-1],
		"p95_height_error_m": errors[int(errors.size() * 0.95)],
		"max_normal_angle_deg": max_normal_angle,
		"ground_above_samples": ground_above,
		"classified_road_samples": classified_road,
		"worst_samples": rows.slice(0, 24)
	}
	var file := FileAccess.open(output, FileAccess.WRITE)
	if file == null:
		push_error("Cannot write curb audit")
		quit(2)
		return
	file.store_string(JSON.stringify(report, "  ") + "\n")
	file.flush()
	var result := file.get_error()
	file.close()
	print("CURB_CONTACT_AUDIT ", JSON.stringify(report))
	quit(0 if result == OK else 2)
