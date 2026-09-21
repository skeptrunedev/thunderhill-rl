extends RefCounted
## Average color in linear light, then store every level as sRGB bytes.
## Alpha is averaged independently as linear coverage, not premultiplied color.


static func build(source: Image) -> Image:
	if source == null or source.is_empty():
		return null
	var format := source.get_format()
	if format != Image.FORMAT_RGB8 and format != Image.FORMAT_RGBA8:
		return null
	var width := source.get_width()
	var height := source.get_height()
	var linear := Image.create(width, height, false, Image.FORMAT_RGBAF)
	for y in height:
		for x in width:
			linear.set_pixel(x, y, source.get_pixel(x, y).srgb_to_linear())
	if linear.generate_mipmaps() != OK:
		return null
	var channels := 3 if format == Image.FORMAT_RGB8 else 4
	# Preserve the source base level exactly, including its alpha bytes.
	var encoded := source.get_data().slice(0, width * height * channels)
	var floats := linear.get_data()
	var level_width := width
	var level_height := height
	for level in range(1, linear.get_mipmap_count() + 1):
		level_width = maxi(1, level_width / 2)
		level_height = maxi(1, level_height / 2)
		var offset := linear.get_mipmap_offset(level)
		var pixels := Image.create_from_data(
			level_width,
			level_height,
			false,
			Image.FORMAT_RGBAF,
			floats.slice(offset, offset + level_width * level_height * 16)
		)
		if pixels == null or pixels.is_empty():
			return null
		var output := Image.create(level_width, level_height, false, format)
		for y in level_height:
			for x in level_width:
				output.set_pixel(x, y, pixels.get_pixel(x, y).linear_to_srgb())
		encoded.append_array(output.get_data())
	return Image.create_from_data(width, height, linear.has_mipmaps(), format, encoded)
