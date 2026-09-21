extends SceneTree
## Preserve linear light when reducing an sRGB color texture. Defaults to asphalt.
const Mipmaps = preload("res://scripts/color_mipmaps.gd")
const SOURCE := "res://assets/materials/racing_asphalt_v1.png"
const OUTPUT := "res://assets/materials/racing_asphalt_v1.res"


func _initialize() -> void:
	var source_path := SOURCE
	var output_path := OUTPUT
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--source="):
			source_path = arg.trim_prefix("--source=")
		elif arg.begins_with("--output="):
			output_path = arg.trim_prefix("--output=")
		else:
			push_error("Unknown color texture bake argument")
			quit(2)
			return
	if (
		not source_path.begins_with("res://assets/")
		or not output_path.begins_with("res://assets/")
		or not source_path.ends_with(".png")
		or not output_path.ends_with(".res")
		or ".." in source_path
		or ".." in output_path
	):
		push_error("Use an asset PNG source and asset RES output")
		quit(2)
		return
	# Offline authoring input, never opened as a loose image in an exported game.
	var source := Image.load_from_file(ProjectSettings.globalize_path(source_path))
	var filtered: Image = Mipmaps.build(source)
	if filtered == null:
		push_error("Cannot build color mipmaps")
		quit(2)
		return
	var texture := ImageTexture.create_from_image(filtered)
	var error := ResourceSaver.save(texture, output_path)
	if error != OK:
		push_error("Cannot save color texture")
		quit(2)
		return
	var metadata := {
		"source_sha256": FileAccess.get_sha256(source_path),
		"builder_sha256": FileAccess.get_sha256("res://scripts/color_mipmaps.gd"),
		"bake_tool_sha256": FileAccess.get_sha256("res://tools/bake_asphalt_material.gd"),
		"output_sha256": FileAccess.get_sha256(output_path),
		"width": filtered.get_width(),
		"height": filtered.get_height(),
		"mipmaps": filtered.get_mipmap_count(),
		"encoding": "sRGB color, mip levels filtered in linear light",
		"limitation": "Original generated appearance estimate, not a measured scan"
	}
	var file := FileAccess.open(output_path.get_basename() + ".json", FileAccess.WRITE)
	if file == null:
		quit(2)
		return
	file.store_string(JSON.stringify(metadata, "  ") + "\n")
	file.flush()
	error = file.get_error()
	file.close()
	print("ASPHALT_MIPMAP_BAKE error=", error, " metadata=", JSON.stringify(metadata))
	quit(0 if error == OK else 2)
