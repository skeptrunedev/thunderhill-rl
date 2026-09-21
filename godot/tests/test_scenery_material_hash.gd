extends SceneTree
const Verifier = preload("res://tools/bake_scenery.gd")
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var node := MeshInstance3D.new()
	node.name = "MaterialHashFixture"
	node.mesh = BoxMesh.new()
	var shader := Shader.new()
	shader.code = "shader_type spatial;\nuniform float strength = 0.25;\nuniform vec3 tint = vec3(1.0);\nuniform sampler2D surface_color : source_color;\nvoid fragment() { ALBEDO = texture(surface_color, UV).rgb * tint * strength; }"
	var material := ShaderMaterial.new()
	material.shader = shader
	material.set_shader_parameter(
		"surface_color", load("res://assets/materials/dry_cut_grass_v2.png")
	)
	node.material_override = material
	var baseline := Verifier.fingerprint(node)
	check(not baseline.is_empty(), "Shader fingerprint failed")
	material.set_shader_parameter("strength", 0.75)
	check(Verifier.fingerprint(node) != baseline, "Uniform change was ignored")
	material.set_shader_parameter("strength", null)
	check(Verifier.fingerprint(node) == baseline, "Default restoration changed fingerprint")
	if RenderingServer.shader_get_parameter_default(shader.get_rid(), "strength") != null:
		material.set_shader_parameter("strength", 0.25)
		check(
			Verifier.fingerprint(node) == baseline, "Explicit default differs from implicit default"
		)
		material.set_shader_parameter("strength", null)
	shader.code = shader.code.replace("strength = 0.25", "strength = 0.5")
	check(Verifier.fingerprint(node) != baseline, "Changed shader default was ignored")
	shader.code = shader.code.replace("strength = 0.5", "strength = 0.25")
	var original_texture: Texture2D = material.get_shader_parameter("surface_color")
	material.set_shader_parameter(
		"surface_color", load("res://assets/materials/Asphalt010_1K-JPG_Color.jpg")
	)
	check(Verifier.fingerprint(node) != baseline, "Texture change was ignored")
	material.set_shader_parameter("surface_color", original_texture)
	shader.code += "\n// Changed shader source\n"
	check(Verifier.fingerprint(node) != baseline, "Shader source change was ignored")
	material.set_shader_parameter("tint", Vector3(0.4, 0.6, 0.8))
	var expected := Verifier.fingerprint(node)
	var packed := PackedScene.new()
	check(packed.pack(node) == OK, "Cannot pack fixture")
	var path := "user://scenery-material-hash-%s.scn" % str(Time.get_ticks_usec())
	check(ResourceSaver.save(packed, path) == OK, "Cannot save fixture")
	var restored_scene := (
		ResourceLoader.load(path, "PackedScene", ResourceLoader.CACHE_MODE_IGNORE) as PackedScene
	)
	check(restored_scene != null, "Cannot reload fixture")
	if restored_scene != null:
		var restored := restored_scene.instantiate()
		check(
			Verifier.fingerprint(restored) == expected,
			"Shader fingerprint changed after serialization"
		)
		restored.free()
	check(DirAccess.remove_absolute(path) == OK, "Cannot remove own fixture")
	node.free()
	print("SCENERY_MATERIAL_HASH_CHECK failures=", failures)
	quit(failures)
