// Isolated direct diffuse study. Burley diffuse follows Godot's MIT licensed
// scene_forward_lights_inc.glsl at ed1daf0bf. Direct specular is deliberately
// omitted in BOTH the zero-strength control and candidate, not in production.
// Original broad backscatter lobe is an appearance hypothesis, not a calibrated
// or energy-conserving canopy BRDF. Ambient and shadow attenuation are retained.
uniform float canopy_backscatter_strength : hint_range(0.0, 4.0) = 0.0;
void light() {
    float nl = max(dot(NORMAL, LIGHT), 0.0);
    float nv = max(dot(NORMAL, VIEW), 0.00001);
    vec3 half_sum = VIEW + LIGHT;
    vec3 half_vector = half_sum / max(length(half_sum), 0.00001);
    float lh = clamp(dot(LIGHT, half_vector), 0.0, 1.0);
    float fd90 = 2.0 * lh * lh * ROUGHNESS - 0.5;
    float fdv = 1.0 + fd90 * pow(1.0 - nv, 5.0);
    float fdl = 1.0 + fd90 * pow(1.0 - nl, 5.0);
    float toward_sun = max(dot(VIEW, LIGHT), 0.0);
    float lobe = toward_sun * toward_sun;
    float response = 1.0 + canopy_backscatter_strength * canopy_coverage * lobe;
    DIFFUSE_LIGHT += LIGHT_COLOR * ATTENUATION * nl * fdv * fdl * response / PI;
}
