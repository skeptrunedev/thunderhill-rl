shader_type spatial;
render_mode cull_disabled;

// Opaque appearance approximation guided by the supplied onboard footage.
// This models a tinted pot and fluid boundary, not optical transmission or slosh.
uniform vec3 fluid_color : source_color = vec3(0.40, 0.19, 0.045);
uniform vec3 empty_color : source_color = vec3(0.34, 0.29, 0.20);
uniform float half_height = 0.0225;
varying float local_height;

void vertex() {
    local_height = VERTEX.y;
}

void fragment() {
    float level = local_height / (2.0 * half_height) + 0.5;
    float boundary = smoothstep(0.67, 0.70, level);
    float meniscus = 1.0 - smoothstep(0.0, 0.025, abs(level - 0.68));
    ALBEDO = mix(fluid_color, empty_color, boundary) * (1.0 + 0.18 * meniscus);
    METALLIC = 0.0;
    ROUGHNESS = 0.27;
    SPECULAR = 0.5;
    CLEARCOAT = 0.45;
    CLEARCOAT_ROUGHNESS = 0.2;
}
