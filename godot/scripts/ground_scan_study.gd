extends RefCounted
## Preview only. Replaces the measured terrain's material without changing meshes,
## collisions, track geometry or the separately rendered grass instances.

const SHADER = preload("res://shaders/ground_scan_study.gdshader")
const ColorMipmaps = preload("res://scripts/color_mipmaps.gd")
const MANIFEST = "res://assets/materials/scan_study/source.json"
const DRY_MANIFEST = "res://assets/materials/scan_study/withered-source.json"
const CHANNELS = {"Diffuse": "scan_color", "nor_gl": "scan_normal", "Rough": "scan_roughness"}


static func apply(
	track: Node, tile_m: float = 2.51, normal_strength: float = 1.0, asset: String = "grass"
) -> Dictionary:
	if asset not in ["grass", "withered"]:
		return {"error": "Unknown scan asset"}
	var manifest_path := DRY_MANIFEST if asset == "withered" else MANIFEST
	if not is_finite(tile_m) or tile_m < 0.25 or tile_m > 8.0:
		return {"error": "Scan tile size must be finite and within 0.25 to 8 metres"}
	if not is_finite(normal_strength) or normal_strength < 0.0 or normal_strength > 2.0:
		return {"error": "Scan normal strength must be finite and within 0 to 2"}
	if track == null:
		return {"error": "Missing track"}
	var terrain := track.get_node_or_null("MeasuredTerrain") as MeshInstance3D
	if terrain == null or not track.get("terrain_material") is ShaderMaterial:
		return {"error": "Track has no measured terrain with a shader material"}
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(manifest_path))
	if not parsed is Dictionary or not parsed.get("files") is Array:
		return {"error": "Invalid scan source manifest"}
	var material := ShaderMaterial.new()
	material.shader = SHADER
	var hashes: Dictionary = {}
	for file: Dictionary in parsed["files"]:
		var channel: String = file.get("channel", "")
		if not CHANNELS.has(channel) or hashes.has(channel):
			return {"error": "Unexpected or duplicate scan channel: " + channel}
		var resource_path := "res://" + str(file.get("path", "")).trim_prefix("godot/")
		var digest := FileAccess.get_sha256(resource_path)
		if digest.is_empty() or digest != file.get("sha256", ""):
			return {"error": "Missing or changed scan source: " + resource_path}
		var source := Image.load_from_file(resource_path)
		if source == null or source.is_empty():
			return {"error": "Cannot decode scan source: " + resource_path}
		source.convert(Image.FORMAT_RGB8)
		if channel == "Diffuse":
			source = ColorMipmaps.build(source)
			if source == null:
				return {"error": "Cannot build linear light scan color mipmaps"}
		elif source.generate_mipmaps(channel == "nor_gl") != OK:
			return {"error": "Cannot build scan data mipmaps: " + channel}
		material.set_shader_parameter(CHANNELS[channel], ImageTexture.create_from_image(source))
		hashes[channel] = digest
	if hashes.size() != CHANNELS.size():
		return {"error": "Scan manifest must supply color, normal and roughness"}
	material.set_shader_parameter("tile_m", tile_m)
	material.set_shader_parameter("normal_strength", normal_strength)
	var albedo_gain := 0.55 if asset == "withered" else 1.0
	material.set_shader_parameter("albedo_gain", albedo_gain)
	terrain.material_override = material
	track.set("terrain_material", material)
	return {
		"study": "scanned_pbr_ground",
		"asset": parsed.get("asset"),
		"source": parsed.get("source"),
		"license": parsed.get("license"),
		"manifest_sha256": FileAccess.get_sha256(manifest_path),
		"map_sha256": hashes,
		"tile_m": tile_m,
		"normal_strength": normal_strength,
		"albedo_gain": albedo_gain,
		"shader_sha256": FileAccess.get_sha256(SHADER.resource_path),
		"color_mips": "linear light",
		"normal_mips": "renormalized",
		"limitations": parsed.get("limitations", []),
	}
