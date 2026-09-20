extends SceneTree
## Regenerate static scenery before packaging. Never executes during gameplay.
const OUTPUT := "res://assets/generated/scenery.scn"
const SOURCES := [
	"scripts/scenery.gd",
	"scripts/track.gd",
	"scripts/offroad_surface.gd",
	"scripts/curb_surface.gd",
	"data/track.json",
	"data/terrain.json",
	"data/surface.json",
	"data/landmarks.json",
	"assets/grass/grass.json",
	"assets/grass/dry_grass_rgba.png",
	"tools/bake_scenery.gd"
]


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("Scenery baking requires a real renderer to preserve MultiMesh buffers")
		quit(2)
		return
	var track := preload("res://scripts/track.gd").new()
	root.add_child(track)
	assert(track.initialization_error.is_empty(), track.initialization_error)
	var scenery := preload("res://scripts/scenery.gd").new()
	scenery.name = "StaticScenery"
	root.add_child(scenery)
	var started := Time.get_ticks_usec()
	scenery.build(track)
	var build_ms := (Time.get_ticks_usec() - started) / 1000.0
	var instance_count := 0
	for child in scenery.get_children():
		if child is MultiMeshInstance3D:
			instance_count += child.multimesh.instance_count
			assert(
				child.multimesh.get_instance_transform(0).basis.determinant() > 0.0,
				"Missing instance transform"
			)
	assert(instance_count == 96000, "Expected all authored grass instances")
	var expected := fingerprint(scenery)
	_set_owners(scenery, scenery)
	scenery.set_script(null)
	var packed := PackedScene.new()
	assert(packed.pack(scenery) == OK, "Cannot pack scenery")
	assert(
		ResourceSaver.save(packed, OUTPUT, ResourceSaver.FLAG_COMPRESS) == OK, "Cannot save scenery"
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
	assert(fingerprint(restored) == expected, "Scenery changed during serialization")
	var hashes := {}
	for source in SOURCES:
		hashes[source] = FileAccess.get_sha256("res://" + source)
	var report := {
		"schema_version": 1,
		"grass_instances": instance_count,
		"sources": hashes,
		"scene_sha256": FileAccess.get_sha256(OUTPUT),
		"geometry_sha256": expected,
		"build_ms": build_ms,
		"load_ms": load_ms,
		"godot_version": Engine.get_version_info().string
	}
	var file := FileAccess.open("res://data/scenery-bake.json", FileAccess.WRITE)
	assert(file != null, "Cannot write scenery provenance")
	file.store_string(JSON.stringify(report, "  ") + "\n")
	file.close()
	print("SCENERY_BAKE ", JSON.stringify(report))
	restored.free()
	quit()


func _set_owners(node: Node, owner_root: Node) -> void:
	for child in node.get_children():
		child.owner = owner_root
		_set_owners(child, owner_root)


static func fingerprint(node: Node) -> String:
	var digest := HashingContext.new()
	digest.start(HashingContext.HASH_SHA256)
	_hash_node(node, digest)
	return digest.finish().hex_encode()


static func _hash_node(node: Node, digest: HashingContext) -> void:
	digest.update(var_to_bytes(str(node.name)))
	if node is Node3D:
		digest.update(var_to_bytes(node.transform))
	if node is GeometryInstance3D:
		digest.update(
			var_to_bytes(
				[node.visibility_range_end, node.visibility_range_end_margin, node.cast_shadow]
			)
		)
	if node is MeshInstance3D:
		_hash_material(node.material_override, digest)
		for surface in node.mesh.get_surface_count():
			digest.update(var_to_bytes(node.mesh.surface_get_arrays(surface)))
			_hash_material(node.mesh.surface_get_material(surface), digest)
	if node is MultiMeshInstance3D:
		for surface in node.multimesh.mesh.get_surface_count():
			digest.update(var_to_bytes(node.multimesh.mesh.surface_get_arrays(surface)))
			_hash_material(node.multimesh.mesh.surface_get_material(surface), digest)
		for index in node.multimesh.instance_count:
			digest.update(var_to_bytes(node.multimesh.get_instance_transform(index)))
			digest.update(var_to_bytes(node.multimesh.get_instance_color(index)))
	for child in node.get_children():
		_hash_node(child, digest)


static func _hash_material(material: Material, digest: HashingContext) -> void:
	if material == null:
		digest.update(var_to_bytes(null))
		return
	assert(material is StandardMaterial3D, "Extend scenery verifier for new material type")
	var values := []
	for property in material.get_property_list():
		if (
			property.usage & PROPERTY_USAGE_STORAGE
			and (
				property.name
				not in ["resource_name", "resource_local_to_scene", "resource_path", "script"]
			)
		):
			var value: Variant = material.get(property.name)
			if value is Resource:
				assert(not value.resource_path.is_empty(), "Unverified embedded material resource")
				value = value.resource_path
			values.append([property.name, value])
	digest.update(var_to_bytes(values))
