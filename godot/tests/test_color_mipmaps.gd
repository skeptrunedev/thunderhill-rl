extends SceneTree
const ColorMipmaps = preload("res://scripts/color_mipmaps.gd")
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
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
