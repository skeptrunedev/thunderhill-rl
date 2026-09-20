extends RefCounted
## Detect the observed empty/black readback failure in this daylight environment.
## Nonblack pixels are necessary, not proof of a complete or correct rendered frame.


static func classify(image: Image, expected_size := Vector2i.ZERO) -> String:
	if image == null or image.is_empty():
		return "empty_image"
	if expected_size != Vector2i.ZERO and image.get_size() != expected_size:
		return "wrong_dimensions"
	var rgb := image.duplicate() as Image
	rgb.convert(Image.FORMAT_RGB8)
	var pixels := rgb.get_data()
	return "all_black" if pixels.count(0) == pixels.size() else "nonblack"
