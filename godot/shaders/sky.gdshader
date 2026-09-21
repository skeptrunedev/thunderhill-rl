shader_type sky;

uniform sampler2D panorama : filter_linear_mipmap, repeat_enable;
// Appearance tuning, not calibrated radiometry. Background exposure is separate.
uniform float background_energy : hint_range(0.0, 2.0) = 0.7;
uniform float lighting_energy : hint_range(0.0, 2.0) = 0.45;
// Background color study only; does not alter environment lighting radiance.
uniform float background_saturation : hint_range(0.0, 2.0) = 1.0;

void sky() {
    vec3 direction = normalize(EYEDIR);
    vec2 uv = vec2(atan(direction.x, direction.z) / TAU + 0.5,
                   acos(clamp(direction.y, -1.0, 1.0)) / PI);
    COLOR = texture(panorama, uv).rgb;
    if (AT_CUBEMAP_PASS) {
        COLOR *= lighting_energy;
    } else {
        float luminance = dot(COLOR, vec3(0.2126, 0.7152, 0.0722));
        COLOR = max(mix(vec3(luminance), COLOR, background_saturation), vec3(0.0));
        COLOR *= background_energy;
    }
}
