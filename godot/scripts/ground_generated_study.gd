extends RefCounted
## Original footage guided material study, independent of production layering.


static func apply(track: Node3D) -> Dictionary:
	var material := ShaderMaterial.new()
	material.shader = preload("res://shaders/ground_generated_study.gdshader")
	var texture = load("res://assets/materials/ground_study/flattened-field-v1.res")
	if not texture is Texture2D:
		return {"error": "Generated study texture missing"}
	material.set_shader_parameter("field_color", texture)
	track.get_node("MeasuredTerrain").material_override = material
	track.terrain_material = material
	return {
		"mode": "generated",
		"texture_sha256": FileAccess.get_sha256(texture.resource_path),
		"shader_sha256": FileAccess.get_sha256("res://shaders/ground_generated_study.gdshader"),
		"tile_m": 4.0,
		"color_gain": 0.55,
		"limitation":
		"Original generated appearance, not a measured scan; relief estimated from color"
	}
