shader_type spatial;
render_mode cull_disabled;
// Georeferenced orthoimagery for the MotoGP circuits (tools/build_circuit.py).
// Three nested layers in local metres (x east, z south): 256 m detail tiles along
// the road corridor, a domain overview and a coarse horizon image. Close-range
// relief is generic CC0 material detail, modulating only luminance and fading
// out with distance; it adds no colour or geometry information of its own.
uniform sampler2DArray detail_tiles : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
// Layer index plus one in the red channel (value / 255); zero where no tile exists.
uniform sampler2D tile_index : filter_nearest, repeat_disable;
uniform bool has_detail = false;
uniform vec2 grid_origin;
uniform vec2 grid_cells = vec2(1.0);
uniform float tile_m = 256.0;
uniform sampler2D overview : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform vec4 overview_bounds = vec4(0.0, 0.0, 1.0, 1.0);
uniform sampler2D horizon_image : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform vec4 horizon_bounds = vec4(0.0, 0.0, 1.0, 1.0);
uniform bool pavement = false;
// Imagery coarser than the road width cannot colour it; use the asphalt material.
uniform bool imaged_pavement = true;
uniform sampler2D asphalt_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D asphalt_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D ground_detail : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform float asphalt_tile_m = 1.5;
uniform float ground_tile_m = 2.0;
// Maps the photograph's exposure to albedo (imagery.json albedo_gain, set at build time).
uniform float albedo_gain = 1.0;
uniform float detail_strength : hint_range(0.0, 1.0) = 0.45;
uniform float detail_fade_start_m = 12.0;
uniform float detail_fade_end_m = 70.0;

varying vec3 world_position;

void vertex() {
	world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

vec3 layer_color(sampler2D image, vec4 bounds, vec2 xz, vec2 dx, vec2 dy) {
	vec2 size = bounds.zw - bounds.xy;
	return textureGrad(image, (xz - bounds.xy) / size, dx / size, dy / size).rgb;
}

bool inside(vec4 bounds, vec2 xz) {
	return xz.x >= bounds.x && xz.y >= bounds.y && xz.x <= bounds.z && xz.y <= bounds.w;
}

vec3 imagery(vec2 xz, vec2 dx, vec2 dy) {
	vec3 color = layer_color(horizon_image, horizon_bounds, xz, dx, dy);
	if (inside(overview_bounds, xz)) {
		color = layer_color(overview, overview_bounds, xz, dx, dy);
	}
	if (has_detail) {
		vec2 cell = (xz - grid_origin) / tile_m;
		if (cell.x >= 0.0 && cell.y >= 0.0 && cell.x < grid_cells.x && cell.y < grid_cells.y) {
			float layer = floor(texelFetch(tile_index, ivec2(floor(cell)), 0).r * 255.0 + 0.5) - 1.0;
			if (layer >= 0.0) {
				// Explicit gradients: fract() would otherwise select the smallest mip at tile seams.
				color = textureGrad(detail_tiles, vec3(fract(cell), layer), dx / tile_m, dy / tile_m).rgb;
			}
		}
	}
	return color;
}

float luminance(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void fragment() {
	vec2 xz = world_position.xz;
	vec2 dx = dFdx(xz);
	vec2 dy = dFdy(xz);
	float distance_m = length(VERTEX);
	float near = 1.0 - smoothstep(detail_fade_start_m, detail_fade_end_m, distance_m);
	vec3 albedo = imagery(xz, dx, dy) * albedo_gain;
	if (pavement) {
		vec2 uv = xz / asphalt_tile_m;
		vec3 grain = texture(asphalt_color, uv).rgb;
		vec3 mean = textureLod(asphalt_color, vec2(0.5), 12.0).rgb;
		if (!imaged_pavement) {
			albedo = grain;
		} else {
			float gain = luminance(grain) / max(luminance(mean), 0.02);
			albedo *= mix(1.0, clamp(gain, 0.6, 1.4), detail_strength * near);
		}
		NORMAL_MAP = texture(asphalt_normal, uv).rgb;
		NORMAL_MAP_DEPTH = 0.6 * near;
		ROUGHNESS = 0.82;
		SPECULAR = 0.35;
	} else {
		vec3 grain = texture(ground_detail, xz / ground_tile_m).rgb;
		vec3 mean = textureLod(ground_detail, vec2(0.5), 12.0).rgb;
		float gain = luminance(grain) / max(luminance(mean), 0.02);
		albedo *= mix(1.0, clamp(gain, 0.55, 1.45), detail_strength * near);
		ROUGHNESS = 0.95;
		SPECULAR = 0.2;
	}
	ALBEDO = albedo;
}
