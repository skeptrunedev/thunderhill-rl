extends RefCounted
## Original generated coverage mask, not a measured scan or friction field.


static func apply(track: Node) -> Dictionary:
	var pavement: ShaderMaterial = track.get_node("RacingSurface").material_override
	pavement.set_shader_parameter("exit_scuff_texture_enabled", true)
	pavement.set_shader_parameter(
		"exit_scuff_mask", preload("res://assets/materials/road_scuff_v1.png")
	)
	return {
		"mask": "res://assets/materials/road_scuff_v1.png",
		"metadata":
		JSON.parse_string(
			FileAccess.get_file_as_string("res://assets/materials/road_scuff_v1.json")
		),
		"extent_m": [12.0, 8.0],
		"center_station_m": 1355.5,
		"limitation":
		"Original generated coverage mask, artistic placement and size, no friction change"
	}
