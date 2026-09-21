extends RefCounted
## Configure a fixed world rectilinear source shared by sky and environment light.
## Callers validate projection bounds and supply sRGB bytes with linear light mips.


static func configure(
	material: ShaderMaterial,
	texture: Texture2D,
	horizontal_fov_degrees: float,
	center_azimuth_degrees: float,
	center_elevation_degrees: float,
	feather_fraction: float
) -> Dictionary:
	var azimuth := deg_to_rad(center_azimuth_degrees)
	var elevation := deg_to_rad(center_elevation_degrees)
	var forward := Vector3(
		sin(azimuth) * cos(elevation), sin(elevation), cos(azimuth) * cos(elevation)
	)
	# A viewer looking toward positive Z has screen right along negative X.
	# Constructing right from azimuth also preserves its direction at the poles.
	var right := Vector3(-cos(azimuth), 0.0, sin(azimuth))
	var up := right.cross(forward)
	var tan_horizontal := tan(deg_to_rad(horizontal_fov_degrees) * 0.5)
	var tan_vertical := tan_horizontal * float(texture.get_height()) / float(texture.get_width())
	material.set_shader_parameter("sky_patch", texture)
	material.set_shader_parameter("sky_patch_forward", forward)
	material.set_shader_parameter("sky_patch_right", right)
	material.set_shader_parameter("sky_patch_up", up)
	material.set_shader_parameter("sky_patch_tan_half_fov", Vector2(tan_horizontal, tan_vertical))
	material.set_shader_parameter("sky_patch_feather", feather_fraction)
	material.set_shader_parameter("sky_patch_enabled", true)
	return {
		"horizontal_fov_degrees": horizontal_fov_degrees,
		"vertical_fov_degrees": rad_to_deg(2.0 * atan(tan_vertical)),
		"center_azimuth_degrees": center_azimuth_degrees,
		"center_elevation_degrees": center_elevation_degrees,
		"feather_fraction": feather_fraction,
		"forward": [forward.x, forward.y, forward.z],
		"right": [right.x, right.y, right.z],
		"up": [up.x, up.y, up.z],
		"tan_half_fov": [tan_horizontal, tan_vertical],
		"projection": "Rectilinear, fixed world orientation, shared background and radiance"
	}
