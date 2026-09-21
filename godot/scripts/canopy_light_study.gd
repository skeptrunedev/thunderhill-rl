extends RefCounted
## Isolated direct diffuse experiment. Never installed by the production scene.
const LIGHT_SOURCE := "res://shaders/canopy_light_study.gdshaderinc"


static func apply(track: Node3D, strength: float) -> Dictionary:
	if not is_finite(strength) or strength < 0.0 or strength > 4.0:
		return {"error": "Canopy strength must be finite and within zero to four"}
	var original: ShaderMaterial = track.terrain_material
	var shader := Shader.new()
	shader.code = original.shader.code + '\n#include "' + LIGHT_SOURCE + '"\n'
	var material := ShaderMaterial.new()
	material.shader = shader
	for parameter in original.shader.get_shader_uniform_list():
		var value = original.get_shader_parameter(parameter.name)
		if value != null:
			material.set_shader_parameter(parameter.name, value)
	material.set_shader_parameter("canopy_backscatter_strength", strength)
	track.get_node("MeasuredTerrain").material_override = material
	track.terrain_material = material
	return {
		"mode": "canopy_direct_diffuse",
		"strength": strength,
		"base_shader_sha256": FileAccess.get_sha256(original.shader.resource_path),
		"light_source_sha256": FileAccess.get_sha256(LIGHT_SOURCE),
		"limitation":
		"Compare with strength zero. Both omit direct specular. Original uncalibrated lobe, not energy conserving."
	}
