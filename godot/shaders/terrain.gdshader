shader_type spatial;
render_mode cull_disabled;
uniform sampler2D grass_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D soil_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
varying vec3 world_pos;

float terrain_hash(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

float terrain_noise(vec2 position) {
    vec2 cell = floor(position);
    vec2 fraction = fract(position);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    return mix(mix(terrain_hash(cell), terrain_hash(cell + vec2(1.0, 0.0)), fraction.x),
               mix(terrain_hash(cell + vec2(0.0, 1.0)), terrain_hash(cell + vec2(1.0)), fraction.x), fraction.y);
}

void vertex() {
    world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
    // Physical source tile widths: grass 2 m, soil 1.3 m. Broad coverage is
    // original decorative variation, not a vegetation survey or grip map.
    vec2 uv = world_pos.xz / 2.0;
    float coverage = terrain_noise(world_pos.xz / 45.0) * 0.57
                   + terrain_noise(world_pos.xz / 12.0 + vec2(9.1, 4.3)) * 0.29
                   + terrain_noise(world_pos.xz / 3.0 + vec2(3.7, 8.2)) * 0.14;
    mat2 rotation = mat2(vec2(0.8, 0.6), vec2(-0.6, 0.8));
    vec2 secondary_uv = rotation * uv * 0.93 + vec2(0.37, 0.71);
    vec3 grass = mix(texture(grass_color, uv).rgb, texture(grass_color, secondary_uv).rgb, 0.35);
    vec3 soil = texture(soil_color, world_pos.xz / 1.3).rgb;
    ALBEDO = mix(grass, soil, smoothstep(0.32, 0.72, coverage) * 0.6) * vec3(0.88, 0.80, 0.66);
    ROUGHNESS = 0.96;
    NORMAL_MAP = texture(grass_normal, uv).rgb;
    NORMAL_MAP_DEPTH = 0.45;
}
