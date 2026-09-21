extends RefCounted
## Original footage guided material study, independent of production layering.


static func apply(
	track: Node3D, regional: bool = false, structured: bool = false, composition: bool = false
) -> Dictionary:
	if (structured or composition) and not regional:
		return {"error": "Structured and composed ground require regional mode"}
	var material: ShaderMaterial
	if regional:
		material = track.terrain_material.duplicate()
	else:
		material = ShaderMaterial.new()
		material.shader = preload("res://shaders/ground_generated_study.gdshader")
	var texture = load("res://assets/materials/ground_study/flattened-field-v1.res")
	if not texture is Texture2D:
		return {"error": "Generated study texture missing"}
	material.set_shader_parameter("photographic_field" if regional else "field_color", texture)
	if regional:
		material.set_shader_parameter("photographic_field_enabled", true)
		material.set_shader_parameter("photographic_sparse_enabled", false)
		material.set_shader_parameter("photographic_structure_strength", 1.0 if structured else 0.0)
	var sparse_sha256 := ""
	if composition:
		var sparse = load("res://assets/materials/ground_study/sparse-field-v1.res")
		if not sparse is Texture2D:
			return {"error": "Sparse study texture missing"}
		material.set_shader_parameter("photographic_sparse", sparse)
		material.set_shader_parameter("photographic_sparse_enabled", true)
		material.set_shader_parameter("photographic_disturbance_strength", 1.0)
		sparse_sha256 = FileAccess.get_sha256(sparse.resource_path)
	track.get_node("MeasuredTerrain").material_override = material
	track.terrain_material = material
	return {
		"mode":
		(
			"composition"
			if composition
			else ("structured" if structured else ("regional" if regional else "generated"))
		),
		"sparse_sha256": sparse_sha256,
		"sparse_enabled": composition,
		"sparse_base_weight": 0.60 if composition else 0.0,
		"sparse_disturbed_weight": 0.95 if composition else 0.0,
		"sparse_placement": "Existing authored corridor with point tint interruptions",
		"structure_strength": 1.0 if structured else 0.0,
		"texture_sha256": FileAccess.get_sha256(texture.resource_path),
		"shader_sha256": FileAccess.get_sha256(material.shader.resource_path),
		"tile_m": 4.0,
		"color_gain": 0.55,
		"limitation":
		"Original generated appearance, not a measured scan; relief estimated from color"
	}
