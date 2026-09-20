shader_type spatial;
render_mode cull_disabled;
uniform sampler2D grass_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D soil_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D soil_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_roughness : filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D soil_roughness : filter_linear_mipmap_anisotropic, repeat_enable;
// Data texture stores linear relative color gains divided by two, not sRGB albedo.
uniform sampler2D terrain_macro : filter_linear_mipmap, repeat_disable;
uniform vec2 macro_origin;
uniform vec2 macro_size;
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
    float soil_weight = smoothstep(0.32, 0.72, coverage) * 0.6;
    vec2 macro_uv = (world_pos.xz - macro_origin) / macro_size;
    vec2 edge_distance = min(macro_uv, vec2(1.0) - macro_uv) * macro_size;
    float macro_weight = smoothstep(0.0, 30.0, min(edge_distance.x, edge_distance.y));
    vec3 macro_gain = mix(vec3(1.0), texture(terrain_macro, macro_uv).rgb * 2.0, macro_weight);
    ALBEDO = mix(grass, soil, soil_weight) * vec3(0.65, 0.56, 0.43) * macro_gain;
    float grass_r = mix(texture(grass_roughness, uv).r, texture(grass_roughness, secondary_uv).r, 0.35);
    ROUGHNESS = clamp(mix(grass_r, texture(soil_roughness, world_pos.xz / 1.3).r, soil_weight), 0.65, 1.0);
    vec3 primary_n = texture(grass_normal, uv).rgb * 2.0 - 1.0;
    vec3 secondary_n = texture(grass_normal, secondary_uv).rgb * 2.0 - 1.0;
    // Bring the rotated texture's tangent directions back to the original UV frame.
    secondary_n.xy = transpose(rotation) * secondary_n.xy;
    vec3 grass_n = normalize(mix(primary_n, secondary_n, 0.35));
    vec3 soil_n = texture(soil_normal, world_pos.xz / 1.3).rgb * 2.0 - 1.0;
    NORMAL_MAP = normalize(mix(grass_n, soil_n, soil_weight)) * 0.5 + 0.5;
    NORMAL_MAP_DEPTH = 0.45;
}
