extends SceneTree
## Projection conventions at cardinal headings and the zenith.
const Patch = preload("res://scripts/sky_patch.gd")


func _initialize() -> void:
	var material := ShaderMaterial.new()
	material.shader = preload("res://shaders/sky.gdshader")
	var texture := ImageTexture.create_from_image(Image.create(4, 2, false, Image.FORMAT_RGB8))
	Patch.configure(material, texture, 90.0, 0.0, 0.0, 0.15)
	assert(material.get_shader_parameter("sky_patch_forward").is_equal_approx(Vector3.BACK))
	assert(material.get_shader_parameter("sky_patch_right").is_equal_approx(Vector3.LEFT))
	assert(material.get_shader_parameter("sky_patch_up").is_equal_approx(Vector3.UP))
	assert(material.get_shader_parameter("sky_patch_tan_half_fov").is_equal_approx(Vector2(1, 0.5)))
	var east := Patch.configure(material, texture, 90.0, 90.0, 0.0, 0.15)
	assert(material.get_shader_parameter("sky_patch_forward").is_equal_approx(Vector3.RIGHT))
	assert(material.get_shader_parameter("sky_patch_right").is_equal_approx(Vector3.BACK))
	assert(absf(east.vertical_fov_degrees - 53.1301023542) < 0.0001)
	for azimuth in [-180.0, 0.0, 90.0]:
		Patch.configure(material, texture, 90.0, azimuth, 90.0, 0.15)
		var forward: Vector3 = material.get_shader_parameter("sky_patch_forward")
		var right: Vector3 = material.get_shader_parameter("sky_patch_right")
		var up: Vector3 = material.get_shader_parameter("sky_patch_up")
		assert(forward.is_equal_approx(Vector3.UP))
		assert(absf(right.length() - 1.0) < 0.00001 and absf(up.length() - 1.0) < 0.00001)
		assert(absf(right.dot(up)) < 0.00001 and absf(up.dot(forward)) < 0.00001)
		assert(right.cross(forward).is_equal_approx(up))
	print("PASS: sky patch cardinal orientation, aspect and polar basis")
	quit()
