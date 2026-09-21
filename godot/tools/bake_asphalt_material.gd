extends SceneTree
## Preserve linear light when reducing the generated sRGB albedo.
const Mipmaps = preload("res://scripts/color_mipmaps.gd")
const SOURCE := "res://assets/materials/racing_asphalt_v1.png"
const OUTPUT := "res://assets/materials/racing_asphalt_v1.res"


func _initialize() -> void:
	# Offline authoring input, never opened as a loose image in an exported game.
	var source := Image.load_from_file(ProjectSettings.globalize_path(SOURCE))
	var filtered: Image = Mipmaps.build(source)
	if filtered == null:
		push_error("Cannot build asphalt mipmaps")
		quit(2)
		return
	var texture := ImageTexture.create_from_image(filtered)
	var error := ResourceSaver.save(texture, OUTPUT)
	if error != OK:
		push_error("Cannot save asphalt material")
		quit(2)
		return
	var metadata := {
		"source_sha256": FileAccess.get_sha256(SOURCE),
		"builder_sha256": FileAccess.get_sha256("res://scripts/color_mipmaps.gd"),
		"bake_tool_sha256": FileAccess.get_sha256("res://tools/bake_asphalt_material.gd"),
		"output_sha256": FileAccess.get_sha256(OUTPUT),
		"width": filtered.get_width(),
		"height": filtered.get_height(),
		"mipmaps": filtered.get_mipmap_count(),
		"encoding": "sRGB albedo, mip levels filtered in linear light",
		"limitation": "Original generated appearance estimate, not a measured scan"
	}
	var file := FileAccess.open(OUTPUT.get_basename() + ".json", FileAccess.WRITE)
	if file == null:
		quit(2)
		return
	file.store_string(JSON.stringify(metadata, "  ") + "\n")
	file.flush()
	error = file.get_error()
	file.close()
	print("ASPHALT_MIPMAP_BAKE error=", error, " metadata=", JSON.stringify(metadata))
	quit(0 if error == OK else 2)
