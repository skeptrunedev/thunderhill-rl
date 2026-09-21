extends SceneTree
const ColorMipmaps = preload("res://scripts/color_mipmaps.gd")
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--audit-texture="):
			audit_texture(argument.trim_prefix("--audit-texture="))
	check(ColorMipmaps.build(null) == null, "Null input must be rejected")
	check(ColorMipmaps.build(Image.new()) == null, "Empty input must be rejected")
	check(
		ColorMipmaps.build(Image.create(2, 2, false, Image.FORMAT_L8)) == null,
		"Unsupported source format must be rejected"
	)
	for format in [Image.FORMAT_RGB8, Image.FORMAT_RGBA8]:
		var source := Image.create(4, 4, false, format)
		for y in 4:
			for x in 4:
				var value := float((x + y) % 2)
				source.set_pixel(x, y, Color(value, value, value, value))
		# An existing, incorrectly averaged chain must be rebuilt, not retained.
		check(source.generate_mipmaps() == OK, "Fixture mip generation failed")
		var original := source.get_data()
		var output := ColorMipmaps.build(source)
		check(output != null, "Color mip generation failed")
		if output == null:
			continue
		check(source.get_data() == original, "Source image was modified")
		check(output.get_size() == source.get_size(), "Dimensions changed")
		check(output.get_format() == format, "Format changed")
		check(output.get_mipmap_count() == 2, "Wrong mip count")
		var channels := 3 if format == Image.FORMAT_RGB8 else 4
		check(
			output.get_data().slice(0, 16 * channels) == original.slice(0, 16 * channels),
			"Base pixels changed"
		)
		var expected := Color(0.5, 0.5, 0.5).linear_to_srgb().r
		for level in [1, 2]:
			var offset := output.get_mipmap_offset(level)
			var data := output.get_data()
			var count := 4 if level == 1 else 1
			for pixel in count:
				for channel in 3:
					var actual := float(data[offset + pixel * channels + channel]) / 255.0
					check(absf(actual - expected) <= 1.0 / 255.0, "Linear mean not preserved")
				if channels == 4:
					var alpha := float(data[offset + pixel * channels + 3]) / 255.0
					check(absf(alpha - 0.5) <= 1.0 / 255.0, "Alpha must remain linear")
	# Exercise odd dimensions and a one pixel source as well as the square fixture.
	for size in [Vector2i(7, 3), Vector2i(1, 1)]:
		var source := Image.create(size.x, size.y, false, Image.FORMAT_RGB8)
		source.fill(Color(0.3, 0.5, 0.7))
		var output := ColorMipmaps.build(source)
		check(output != null and output.get_size() == size, "Odd or unit size failed")
	print("COLOR_MIPMAPS_CHECK failures=", failures)
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
