shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

// Independent scanned material study, deliberately without the production
// terrain's procedural coverage, aerial color or inferred albedo relief.
uniform sampler2D scan_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D scan_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D scan_roughness : filter_linear_mipmap_anisotropic, repeat_enable;
uniform float tile_m = 2.51;
uniform float normal_strength = 1.0;
varying vec3 world_position;

void vertex() {
	world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
	vec2 scan_uv = world_position.xz / tile_m;
	ALBEDO = texture(scan_color, scan_uv).rgb;
	ROUGHNESS = texture(scan_roughness, scan_uv).r;
	METALLIC = 0.0;
	SPECULAR = 0.3;
	// Derive the basis from this material's world projection, not mesh UVs.
	// VERTEX and NORMAL are view space here. UV handedness is retained.
	vec3 dpdx = dFdx(VERTEX);
	vec3 dpdy = dFdy(VERTEX);
	vec2 duvdx = dFdx(scan_uv);
	vec2 duvdy = dFdy(scan_uv);
	float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
	if (abs(determinant) > 0.0000000001) {
		vec3 tangent = (dpdx * duvdy.y - dpdy * duvdx.y) / determinant;
		vec3 bitangent = (dpdy * duvdx.x - dpdx * duvdy.x) / determinant;
		tangent = normalize(tangent - NORMAL * dot(NORMAL, tangent));
		bitangent = normalize(bitangent - NORMAL * dot(NORMAL, bitangent));
		vec3 sampled = texture(scan_normal, scan_uv).rgb * 2.0 - 1.0;
		sampled.xy *= normal_strength;
		NORMAL = normalize(tangent * sampled.x + bitangent * sampled.y + NORMAL * sampled.z);
	}
}
