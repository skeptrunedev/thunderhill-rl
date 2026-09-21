extends SceneTree
## Bake numerical pavement gains as an explicit, hashable runtime resource.
const SOURCE := "res://assets/materials/pavement_tone.png"
const OUTPUT := "res://assets/materials/pavement_tone.res"
const MANIFEST := "res://assets/materials/pavement_tone_runtime.json"


func _initialize() -> void:
	var provenance: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://assets/materials/pavement_tone.json")
	)
	if provenance.get("output_sha256") != FileAccess.get_sha256(SOURCE):
		push_error("Pavement tone source does not match its provenance")
		quit(2)
		return
	var source := Image.load_from_file(ProjectSettings.globalize_path(SOURCE))
	if source == null or source.is_empty():
		push_error("Cannot load numerical pavement tone source")
		quit(2)
		return
	var original_pixels := source.get_data()
	if source.generate_mipmaps() != OK:
		push_error("Cannot build numerical pavement tone mipmaps")
		quit(2)
		return
	# Values are linear gain data, so no sRGB decoding or color mip builder.
	var error := ResourceSaver.save(ImageTexture.create_from_image(source), OUTPUT)
	if error != OK:
		push_error("Cannot save pavement tone resource")
		quit(2)
		return
	var roundtrip: Image = load(OUTPUT).get_image()
	roundtrip.clear_mipmaps()
	if roundtrip.get_format() != source.get_format() or roundtrip.get_data() != original_pixels:
		push_error("Pavement tone resource changed source data values")
		quit(2)
		return
	var metadata := {
		"source_sha256": FileAccess.get_sha256(SOURCE),
		"output_sha256": FileAccess.get_sha256(OUTPUT),
		"builder_sha256": FileAccess.get_sha256("res://tools/bake_pavement_tone.gd"),
		"dimensions": [source.get_width(), source.get_height()],
		"mipmaps": source.get_mipmap_count(),
		"encoding": "Linear numerical gain data, mipmaps averaged without color conversion"
	}
	var file := FileAccess.open(MANIFEST, FileAccess.WRITE)
	if file == null:
		quit(2)
		return
	file.store_string(JSON.stringify(metadata, "  ") + "\n")
	file.flush()
	error = file.get_error()
	file.close()
	print("PAVEMENT_TONE_BAKE error=", error)
	quit(0 if error == OK else 2)
