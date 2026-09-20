shader_type spatial;
render_mode cull_disabled;
uniform sampler2D color_map : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D normal_map : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D rough_map : filter_linear_mipmap_anisotropic, repeat_enable;

float pavement_hash(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

float pavement_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(pavement_hash(cell), pavement_hash(cell + vec2(1.0, 0.0)), f.x),
               mix(pavement_hash(cell + vec2(0.0, 1.0)), pavement_hash(cell + vec2(1.0)), f.x), f.y);
}

void fragment() {
    // UV is measured in metres. The aggregate tile scale is an artistic estimate:
    // the publisher supplies no physical dimensions for Asphalt010.
    vec2 detail = UV * 1.6;
    float broad = pavement_noise(UV / vec2(2.7, 19.0));
    float fine = pavement_noise(UV / vec2(0.6, 4.0) + vec2(8.3, 2.1));
    float variation = mix(0.84, 1.04, broad * 0.7 + fine * 0.3);
    vec3 aggregate = texture(color_map, detail).rgb;
    // Offset secondary albedo suppresses visible repetition without rotating
    // tangent space normals away from the ribbon's actual UV frame.
    aggregate = mix(aggregate, texture(color_map, detail * 0.71 + vec2(0.37, 0.61)).rgb, 0.28);
    // Neutral warm binder under the cool sky, matched to the inspected dry footage.
    ALBEDO = aggregate * vec3(0.53, 0.50, 0.46) * variation;
    // Dry resurfaced asphalt still reflects at grazing angles. These values are
    // appearance tuning against the onboard references, not measured BRDF data.
    ROUGHNESS = clamp(texture(rough_map, detail).r * 0.48 + 0.20, 0.52, 0.79);
    NORMAL_MAP = texture(normal_map, detail).rgb;
    NORMAL_MAP_DEPTH = 0.18;
}
