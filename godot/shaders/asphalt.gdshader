shader_type spatial;
render_mode cull_disabled;
uniform float lap_length_m;
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
    // Reference 00:20 and 00:40 show coherent longitudinal surface variation,
    // not the previous strong isotropic cloud pattern. This is original material
    // detail, not a claim about the measured rubber line or repaving seams.
    float broad = pavement_noise(UV / vec2(2.7, 28.0));
    float streak = pavement_noise(UV / vec2(0.45, 12.0) + vec2(8.3, 2.1));
    float fine_streak = pavement_noise(UV / vec2(0.055, 7.0) + vec2(19.7, 1.4));
    float variation = mix(0.92, 1.04, broad) * mix(0.97, 1.03, streak) * mix(0.98, 1.02, fine_streak);
    vec3 aggregate = texture(color_map, detail).rgb;
    // Offset secondary albedo suppresses visible repetition without rotating
    // tangent space normals away from the ribbon's actual UV frame.
    aggregate = mix(aggregate, texture(color_map, detail * 0.71 + vec2(0.37, 0.61)).rgb, 0.28);
    // Neutral warm binder under the cool sky, matched to the inspected dry footage.
    ALBEDO = aggregate * vec3(0.53, 0.50, 0.46) * variation;
    // Binder-rich edge appearance from the footage. Width and irregularity
    // are artistic estimates; the existing pavement boundary remains intact.
    float edge_distance = max(0.0, min(UV.x, UV2.x - UV.x));
    float phase = UV.y / lap_length_m * TAU;
    vec2 edge_domain = vec2(cos(phase), sin(phase)) * (lap_length_m * 0.7 / TAU);
    float edge_width = mix(0.08, 0.15, pavement_noise(edge_domain));
    float edge_aa = max(fwidth(edge_distance), 0.008);
    float edge_weight = 1.0 - smoothstep(edge_width - edge_aa, edge_width + edge_aa, edge_distance);
    ALBEDO *= mix(1.0, 0.20, edge_weight);
    // Keep source aggregate relief subtle. Broad cloudy source roughness was
    // making the grazing reflection read like coarse concrete in the comparison.
    ROUGHNESS = clamp(0.62 + (texture(rough_map, detail).r - 0.5) * 0.10
                      + (streak - 0.5) * 0.035 + (broad - 0.5) * 0.06, 0.52, 0.72);
    ROUGHNESS = mix(ROUGHNESS, 0.72, edge_weight);
    NORMAL_MAP = texture(normal_map, detail).rgb;
    NORMAL_MAP_DEPTH = 0.07;
}
