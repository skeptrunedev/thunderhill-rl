extends SceneTree
## Run before and after startup optimizations to compare actual authored geometry.


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var track := preload("res://scripts/track.gd").new()
	root.add_child(track)
	assert(track.initialization_error.is_empty(), track.initialization_error)
	var landmarks := preload("res://scripts/landmarks.gd").new()
	root.add_child(landmarks)
	var started := Time.get_ticks_usec()
	landmarks.build(track)
	assert(landmarks.initialization_error.is_empty(), landmarks.initialization_error)
	var elapsed := (Time.get_ticks_usec() - started) / 1000.0
	var digest := HashingContext.new()
	digest.start(HashingContext.HASH_SHA256)
	_hash_node(landmarks, digest)
	print(
		"LANDMARK_GEOMETRY ",
		JSON.stringify(
			{
				"sha256": digest.finish().hex_encode(),
				"build_ms": elapsed,
				"track_sha256": FileAccess.get_sha256("res://data/track.json"),
				"surface_sha256": FileAccess.get_sha256("res://data/surface.json")
			}
		)
	)
	quit()


func _hash_node(node: Node, digest: HashingContext) -> void:
	if node is Node3D:
		digest.update(var_to_bytes(node.transform))
	if node is MeshInstance3D:
		for surface in node.mesh.get_surface_count():
			digest.update(var_to_bytes(node.mesh.surface_get_arrays(surface)))
	if node is MultiMeshInstance3D:
		for surface in node.multimesh.mesh.get_surface_count():
			digest.update(var_to_bytes(node.multimesh.mesh.surface_get_arrays(surface)))
		for index in node.multimesh.instance_count:
			digest.update(var_to_bytes(node.multimesh.get_instance_transform(index)))
			if node.multimesh.use_colors:
				digest.update(var_to_bytes(node.multimesh.get_instance_color(index)))
	for child in node.get_children():
		_hash_node(child, digest)
