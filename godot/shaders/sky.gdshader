shader_type sky;

uniform sampler2D panorama : filter_linear_mipmap, repeat_enable;

void sky() {
    vec3 direction = normalize(EYEDIR);
    vec2 uv = vec2(atan(direction.x, direction.z) / TAU + 0.5,
                   acos(clamp(direction.y, -1.0, 1.0)) / PI);
    COLOR = texture(panorama, uv).rgb;
}
