extends RefCounted
## Isolated direct diffuse experiment. Never installed by the production scene.
const LIGHT_SOURCE := "res://shaders/canopy_light_study.gdshaderinc"


static func apply(track: Node3D, strength: float, additive: bool = false) -> Dictionary:
	if not is_finite(strength) or strength < 0.0 or strength > 4.0:
		return {"error": "Canopy strength must be finite and within zero to four"}
	var original: ShaderMaterial = track.terrain_material
	if additive and original.next_pass != null:
		return {"error": "Canopy additive study requires an unused next pass"}
	var source := original.shader.code
	if additive:
		var mode := "render_mode cull_disabled;"
		if source.count(mode) != 1:
			return {"error": "Unexpected terrain render mode"}
		source = (
			source
			. replace(
				mode,
				"render_mode cull_disabled, blend_add, depth_draw_never, ambient_light_disabled, specular_disabled, fog_disabled;"
			)
		)
	var shader := Shader.new()
	shader.code = source + '\n#include "' + LIGHT_SOURCE + '"\n'
	var material := ShaderMaterial.new()
	material.shader = shader
	for parameter in original.shader.get_shader_uniform_list():
		var value = original.get_shader_parameter(parameter.name)
		if value != null:
			material.set_shader_parameter(parameter.name, value)
	material.set_shader_parameter("canopy_backscatter_strength", strength)
	material.set_shader_parameter("canopy_additive", additive)
	var installed := material
	if additive:
		installed = original.duplicate()
		installed.next_pass = material
	track.get_node("MeasuredTerrain").material_override = installed
	track.terrain_material = installed
	return {
		"mode": "canopy_additive" if additive else "canopy_direct_diffuse",
		"additive": additive,
		"strength": strength,
		"base_shader_sha256": FileAccess.get_sha256(original.shader.resource_path),
		"light_source_sha256": FileAccess.get_sha256(LIGHT_SOURCE),
		"limitation":
		"Original uncalibrated lobe, not energy conserving. Additive mode preserves base material but omits fog on the added contribution. Replacement mode omits direct specular."
	}
