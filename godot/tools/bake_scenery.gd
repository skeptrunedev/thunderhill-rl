extends SceneTree
## Regenerate static scenery before packaging. Never executes during gameplay.
const OUTPUT := "res://assets/generated/scenery.scn"
const SOURCES := [
	"scripts/scenery.gd",
	"shaders/dry_stubble.gdshader",
	"scripts/track.gd",
	"scripts/field_coverage.gd",
	"data/field-coverage.json",
	"scripts/offroad_surface.gd",
	"scripts/curb_surface.gd",
	"scripts/curb_placement.gd",
	"data/curb-placement.json",
	"scripts/triangle_ribbon.gd",
	"data/track.json",
	"data/pavement.json",
	"data/terrain.json",
	"data/surface.json",
	"data/landmarks.json",
	"assets/materials/dense_straw_v7.png",
	"assets/materials/dense_straw_v7.res",
	"assets/materials/dense_straw_v7.json",
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
	if expected.is_empty():
		quit(2)
		return
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
	var restored_fingerprint := fingerprint(restored)
	if restored_fingerprint.is_empty() or restored_fingerprint != expected:
		push_error("Scenery changed during serialization or material verification failed")
		restored.free()
		quit(2)
		return
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
	if not _hash_node(node, digest):
		return ""
	return digest.finish().hex_encode()


static func _hash_node(node: Node, digest: HashingContext) -> bool:
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
		if not _hash_material(node.material_override, digest):
			return false
		for surface in node.mesh.get_surface_count():
			digest.update(var_to_bytes(node.mesh.surface_get_arrays(surface)))
			if not _hash_material(node.mesh.surface_get_material(surface), digest):
				return false
	if node is MultiMeshInstance3D:
		for surface in node.multimesh.mesh.get_surface_count():
			digest.update(var_to_bytes(node.multimesh.mesh.surface_get_arrays(surface)))
			if not _hash_material(node.multimesh.mesh.surface_get_material(surface), digest):
				return false
		for index in node.multimesh.instance_count:
			digest.update(var_to_bytes(node.multimesh.get_instance_transform(index)))
			digest.update(var_to_bytes(node.multimesh.get_instance_color(index)))
	for child in node.get_children():
		if not _hash_node(child, digest):
			return false
	return true


static func _hash_material(material: Material, digest: HashingContext) -> bool:
	if material == null:
		digest.update(var_to_bytes(null))
		return true
	if material is ShaderMaterial:
		return _hash_shader_material(material, digest)
	if not material is StandardMaterial3D:
		push_error("Unsupported scenery material type: " + material.get_class())
		return false
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
				if value.resource_path.is_empty():
					push_error("Unverified embedded material resource")
					return false
				value = value.resource_path
			values.append([property.name, value])
	digest.update(var_to_bytes(values))
	return true


static func _hash_shader_material(material: ShaderMaterial, digest: HashingContext) -> bool:
	var shader := material.shader
	if shader == null or shader.code.is_empty():
		push_error("Scenery ShaderMaterial requires shader source")
		return false
	# Includes and global/instance uniforms require additional dependency tracking.
	# Refuse them rather than certify incomplete material provenance.
	var unsupported := RegEx.new()
	unsupported.compile("(#\\s*include\\b|\\b(global|instance)\\s+uniform\\b)")
	if unsupported.search(shader.code) != null:
		push_error("Scenery shader includes or nonmaterial uniforms are not supported")
		return false
	var names: Array[String] = []
	for uniform in shader.get_shader_uniform_list():
		if uniform.usage & (PROPERTY_USAGE_GROUP | PROPERTY_USAGE_SUBGROUP):
			continue
		names.append(str(uniform.name))
	names.sort()
	var values := []
	for name in names:
		var value: Variant = material.get_shader_parameter(name)
		if value == null:
			value = RenderingServer.shader_get_parameter_default(shader.get_rid(), name)
		if value == null:
			value = shader.get_default_texture_parameter(name)
		var encoded := _shader_value(value, name)
		if not encoded.ok:
			return false
		values.append([name, encoded.value])
	if material.next_pass != null:
		push_error("Scenery shader next_pass is not supported")
		return false
	digest.update(
		var_to_bytes(
			["ShaderMaterial", shader.code.sha256_text(), material.render_priority, values]
		)
	)
	return true


static func _shader_value(value: Variant, name: String) -> Dictionary:
	if value == null:
		return {"ok": true, "value": null}
	if value is Array:
		var encoded := []
		for element in value:
			var entry := _shader_value(element, name)
			if not entry.ok:
				return {"ok": false}
			encoded.append(entry.value)
		return {"ok": true, "value": encoded}
	if value is Texture2D:
		var path: String = value.resource_path
		if path.is_empty() or "::" in path or not FileAccess.file_exists(path):
			push_error("Unverified embedded scenery shader texture: " + name)
			return {"ok": false}
		var checksum := FileAccess.get_sha256(path)
		if checksum.is_empty():
			push_error("Cannot hash scenery shader texture: " + path)
			return {"ok": false}
		return {"ok": true, "value": {"path": path, "sha256": checksum}}
	if typeof(value) in [TYPE_OBJECT, TYPE_DICTIONARY, TYPE_CALLABLE, TYPE_SIGNAL, TYPE_RID]:
		push_error("Unsupported scenery shader uniform value: " + name)
		return {"ok": false}
	return {"ok": true, "value": value}
