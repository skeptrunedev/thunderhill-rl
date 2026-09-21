shader_type spatial;
render_mode cull_disabled;

uniform sampler2D grass_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform vec3 ground_tint = vec3(1.0);
// Short ribbons approximate a tuft canopy, not resolved cylindrical stalks.
uniform float canopy_normal_blend : hint_range(0.0, 1.0) = 0.65;
varying vec3 canopy_up;

void vertex() {
    canopy_up = normalize(MODELVIEW_NORMAL_MATRIX * vec3(0.0, 1.0, 0.0));
}

vec3 linear_color(vec3 encoded) {
    return mix(pow((encoded + vec3(0.055)) / 1.055, vec3(2.4)), encoded / 12.92,
               lessThanEqual(encoded, vec3(0.04045)));
}

void fragment() {
    ALBEDO = texture(grass_color, UV).rgb * ground_tint * linear_color(COLOR.rgb);
    vec3 up = normalize(canopy_up);
    // Orient both ribbon faces toward the tuft canopy before blending. This
    // avoids treating the visible rear of a thin blade as a downward surface.
    vec3 blade_normal = dot(NORMAL, up) < 0.0 ? -NORMAL : NORMAL;
    NORMAL = normalize(mix(blade_normal, up, canopy_normal_blend));
    ROUGHNESS = 0.95;
    SPECULAR = 0.1;
    BACKLIGHT = vec3(0.3, 0.24, 0.15);
}
