extends RefCounted
## Original footage guided material study, independent of production layering.


static func apply(track: Node3D, regional: bool = false, structured: bool = false) -> Dictionary:
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
		material.set_shader_parameter("photographic_structure_strength", 1.0 if structured else 0.0)
	track.get_node("MeasuredTerrain").material_override = material
	track.terrain_material = material
	return {
		"mode": "structured" if structured else ("regional" if regional else "generated"),
		"structure_strength": 1.0 if structured else 0.0,
		"texture_sha256": FileAccess.get_sha256(texture.resource_path),
		"shader_sha256": FileAccess.get_sha256(material.shader.resource_path),
		"tile_m": 4.0,
		"color_gain": 0.55,
		"limitation":
		"Original generated appearance, not a measured scan; relief estimated from color"
	}
