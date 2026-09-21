extends RefCounted
## Isolate source color, normal and roughness from the authored road shader.


static func apply(track: Node) -> Dictionary:
	var road := track.get_node("RacingSurface") as MeshInstance3D
	var original: ShaderMaterial = road.material_override
	var material := ShaderMaterial.new()
	material.shader = preload("res://shaders/asphalt_scan_study.gdshader")
	var sources := {}
	for pair in [
		["scan_color", "color_map"], ["scan_normal", "normal_map"], ["scan_roughness", "rough_map"]
	]:
		var texture: Texture2D = original.get_shader_parameter(pair[1])
		if texture == null:
			return {"error": "Missing scan map: " + pair[1]}
		material.set_shader_parameter(pair[0], texture)
		sources[pair[0]] = {
			"path": texture.resource_path, "sha256": FileAccess.get_sha256(texture.resource_path)
		}
	road.material_override = material
	return {
		"mode": "scan",
		"source": "https://ambientcg.com/a/Asphalt010",
		"license": "CC0",
		"maps": sources,
		"shader_sha256": FileAccess.get_sha256(material.shader.resource_path),
		"tile_m": 0.5,
		"color_gain": 0.45,
		"normal_strength": 1.0,
		"limitation": "Generic asphalt scan; physical texture scale and color gain are estimates"
	}
