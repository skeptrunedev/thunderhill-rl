shader_type sky;

uniform sampler2D panorama : filter_linear_mipmap, repeat_enable;
// Appearance tuning, not calibrated radiometry. Background exposure is separate.
uniform float lighting_energy : hint_range(0.0, 2.0) = 0.45;

void sky() {
    vec3 direction = normalize(EYEDIR);
    vec2 uv = vec2(atan(direction.x, direction.z) / TAU + 0.5,
                   acos(clamp(direction.y, -1.0, 1.0)) / PI);
    COLOR = texture(panorama, uv).rgb;
    if (AT_CUBEMAP_PASS) {
        COLOR *= lighting_energy;
    }
}
