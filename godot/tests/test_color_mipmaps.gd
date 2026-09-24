extends SceneTree
const ColorMipmaps = preload("res://scripts/color_mipmaps.gd")
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var requested := false
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--audit-texture="):
			requested = true
			audit_texture(argument.trim_prefix("--audit-texture="))
	check(requested, "Supply --audit-texture=res://path to inspect an actual texture")
	print("COLOR_MIPMAPS_AUDIT failures=", failures)
	quit(failures)


func audit_texture(path: String) -> void:
	var source := Image.load_from_file(ProjectSettings.globalize_path(path))
	var texture := load(path) as Texture2D
	check(source != null and texture != null, "Audit texture must load")
	if source == null or texture == null:
		return
	var imported := texture.get_image()
	check(imported != null, "Audit imported image must be available")
	if imported == null:
		return
	if imported.is_compressed():
		check(imported.decompress() == OK, "Audit texture decompression failed")
	var linear := ColorMipmaps.build(source)
	check(linear != null, "Audit linear mip generation failed")
	if linear == null:
		return
	var encoded := source.duplicate() as Image
	check(encoded.generate_mipmaps() == OK, "Audit encoded mip generation failed")
	check(imported.get_format() == source.get_format(), "Audit formats differ")
	if imported.get_format() != source.get_format():
		return
	check(imported.get_size() == source.get_size(), "Audit image sizes differ")
	check(imported.get_mipmap_count() == linear.get_mipmap_count(), "Audit mip counts differ")
	if (
		imported.get_size() != source.get_size()
		or imported.get_mipmap_count() != linear.get_mipmap_count()
	):
		return
	var source_bytes := source.get_data()
	print("MIP_AUDIT path=", path, " size=", source.get_size(), " format=", source.get_format())
	print(
		"MIP_AUDIT imported_base_equals_source=",
		imported.get_data().slice(0, source_bytes.size()) == source_bytes,
		" imported_chain_equals_encoded_average=",
		imported.get_data() == encoded.get_data(),
		" linear_base_equals_source=",
		linear.get_data().slice(0, source_bytes.size()) == source_bytes
	)
	var channels := 3 if source.get_format() == Image.FORMAT_RGB8 else 4
	var decode := PackedFloat64Array()
	for value in 256:
		decode.append(Color(float(value) / 255.0, 0, 0).srgb_to_linear().r)
	for level in range(imported.get_mipmap_count() + 1):
		var means: Array[float] = []
		for image in [imported, linear]:
			var bytes: PackedByteArray = image.get_data()
			var offset: int = image.get_mipmap_offset(level)
			var end: int = bytes.size()
			if level < image.get_mipmap_count():
				end = image.get_mipmap_offset(level + 1)
			var total := 0.0
			for pixel in range(offset, end, channels):
				total += (
					decode[bytes[pixel]] * 0.2126
					+ decode[bytes[pixel + 1]] * 0.7152
					+ decode[bytes[pixel + 2]] * 0.0722
				)
			means.append(total / float((end - offset) / channels))
		print(
			"MIP_AUDIT level=",
			level,
			" imported_linear_luminance=",
			means[0],
			" corrected_linear_luminance=",
			means[1],
			" corrected_over_imported=",
			means[1] / means[0] if means[0] > 0.0 else 0.0
		)
