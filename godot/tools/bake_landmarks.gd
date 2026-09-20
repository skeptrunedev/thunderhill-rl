extends SceneTree
## Prepare decorative landmarks; the pit wall remains generated and validated at runtime.
const OUTPUT := "res://assets/generated/landmarks.scn"
const Verifier = preload("res://tools/bake_scenery.gd")
const SOURCES := [
	"scripts/landmarks.gd",
	"scripts/track.gd",
	"scripts/offroad_surface.gd",
	"data/track.json",
	"data/terrain.json",
	"data/surface.json",
	"data/landmarks.json",
	"data/pit-wall.json",
	"tools/bake_landmarks.gd",
	"tools/bake_scenery.gd"
]


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("Landmark baking requires a real renderer to preserve tree instance buffers")
		quit(2)
		return
	var track := preload("res://scripts/track.gd").new()
	root.add_child(track)
	assert(track.initialization_error.is_empty(), track.initialization_error)
	var landmarks := preload("res://scripts/landmarks.gd").new()
	landmarks.name = "StaticLandmarks"
	root.add_child(landmarks)
	var started := Time.get_ticks_usec()
	landmarks.build(track, false)
	assert(landmarks.initialization_error.is_empty(), landmarks.initialization_error)
	var build_ms := (Time.get_ticks_usec() - started) / 1000.0
	var expected := fingerprint(landmarks)
	_prepare(landmarks, landmarks)
	landmarks.set_script(null)
	var packed := PackedScene.new()
	assert(packed.pack(landmarks) == OK, "Cannot pack landmarks")
	assert(
		ResourceSaver.save(packed, OUTPUT, ResourceSaver.FLAG_COMPRESS) == OK,
		"Cannot save landmarks"
	)
	started = Time.get_ticks_usec()
	var restored := (
		(
			ResourceLoader.load(OUTPUT, "PackedScene", ResourceLoader.CACHE_MODE_IGNORE)
			as PackedScene
		)
		. instantiate()
	)
	var load_ms := (Time.get_ticks_usec() - started) / 1000.0
	assert(
		fingerprint(restored) == expected,
		"Landmark geometry or materials changed during serialization"
	)
	var hashes := {}
	for source in SOURCES:
		hashes[source] = FileAccess.get_sha256("res://" + source)
	var report := {
		"schema_version": 1,
		"sources": hashes,
		"scene_sha256": FileAccess.get_sha256(OUTPUT),
		"geometry_sha256": expected,
		"build_ms": build_ms,
		"load_ms": load_ms,
		"godot_version": Engine.get_version_info().string,
		"collision": "Runtime pit wall construction unchanged; no collision in this scene"
	}
	var file := FileAccess.open("res://data/landmarks-bake.json", FileAccess.WRITE)
	assert(file != null, "Cannot write landmark provenance")
	file.store_string(JSON.stringify(report, "  ") + "\n")
	file.close()
	print("LANDMARK_BAKE ", JSON.stringify(report))
	restored.free()
	quit()


func _prepare(node: Node, owner_root: Node) -> void:
	assert(
		not node is CollisionObject3D and not node is CollisionShape3D,
		"Decorative landmark bake must not contain collision"
	)
	for child in node.get_children():
		child.owner = owner_root
		_prepare(child, owner_root)


static func fingerprint(node: Node) -> String:
	var digest := HashingContext.new()
	digest.start(HashingContext.HASH_SHA256)
	digest.update(Verifier.fingerprint(node).to_utf8_buffer())
	_hash_visibility_and_instances(node, digest)
	return digest.finish().hex_encode()


static func _hash_visibility_and_instances(node: Node, digest: HashingContext) -> void:
	if node is Node3D:
		digest.update(var_to_bytes(node.visible))
	if node is VisualInstance3D:
		digest.update(var_to_bytes(node.layers))
	if node is MultiMeshInstance3D:
		Verifier._hash_material(node.material_override, digest)
		digest.update(
			var_to_bytes(
				[
					node.multimesh.instance_count,
					node.multimesh.visible_instance_count,
					node.multimesh.transform_format,
					node.multimesh.use_colors,
					node.multimesh.use_custom_data
				]
			)
		)
		for index in node.multimesh.instance_count:
			assert(
				node.multimesh.get_instance_transform(index).basis.determinant() > 0.0,
				"Missing tree instance transform"
			)
	for child in node.get_children():
		_hash_visibility_and_instances(child, digest)
