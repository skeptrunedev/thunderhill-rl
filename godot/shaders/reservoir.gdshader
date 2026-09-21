shader_type spatial;
render_mode cull_back;

uniform sampler2D scene_color : hint_screen_texture, repeat_disable, filter_linear_mipmap;
// Estimated absorption per metre, not measured brake fluid optical properties.
uniform vec3 fluid_absorption = vec3(6.0, 24.0, 60.0);
// Independent vessel wall absorption; appearance estimate, not measured plastic.
uniform vec3 wall_absorption = vec3(600.0, 800.0, 1400.0);
// Visual estimates for cloudy molded plastic, not measured scattering data.
uniform float wall_haze : hint_range(0.0, 1.0) = 0.0;
uniform float transmission_blur_lod : hint_range(0.0, 4.0) = 0.0;
uniform float fluid_radius = 0.024;
uniform float fluid_bottom = -0.020;
uniform float fluid_top = 0.0081;
varying vec3 local_position;
varying vec3 local_camera;

void vertex() {
    local_position = VERTEX;
    local_camera = (inverse(MODELVIEW_MATRIX) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
}

float fluid_path(vec3 p, vec3 direction) {
    float a = dot(direction.xz, direction.xz);
    float b = dot(p.xz, direction.xz);
    float c = dot(p.xz, p.xz) - fluid_radius * fluid_radius;
    float near_t = 0.0;
    float far_t = 1000.0;
    if (a < 0.000001) {
        if (c > 0.0) { return 0.0; }
    } else {
        float discriminant = b * b - a * c;
        if (discriminant <= 0.0) { return 0.0; }
        float root = sqrt(discriminant);
        near_t = max(near_t, (-b - root) / a);
        far_t = min(far_t, (-b + root) / a);
    }
    if (abs(direction.y) < 0.000001) {
        if (p.y < fluid_bottom || p.y > fluid_top) { return 0.0; }
    } else {
        float bottom_t = (fluid_bottom - p.y) / direction.y;
        float top_t = (fluid_top - p.y) / direction.y;
        near_t = max(near_t, min(bottom_t, top_t));
        far_t = min(far_t, max(bottom_t, top_t));
    }
    return max(0.0, far_t - near_t);
}

void fragment() {
    float facing = clamp(dot(normalize(NORMAL), normalize(VIEW)), 0.0, 1.0);
    float path_m = fluid_path(local_position, normalize(local_position - local_camera));
    float fresnel = 0.04 + 0.96 * pow(1.0 - facing, 5.0);
    float wall_path = 0.0015 / max(facing, 0.2);
    vec3 transmission = exp(-fluid_absorption * path_m - wall_absorption * wall_path);
    // The opaque scene copy supplies transmitted radiance. Transparent objects
    // are absent from that copy; this is not ray traced refraction or fluid slosh.
    EMISSION = textureLod(scene_color, SCREEN_UV, transmission_blur_lod).rgb * transmission * (1.0 - fresnel) * (1.0 - wall_haze);
    // Divert a restrained share of transmission into lit wall response. This
    // keeps the haze dependent on scene lighting, rather than self illuminating.
    ALBEDO = mix(vec3(0.015), vec3(0.32, 0.29, 0.24), wall_haze);
    ALPHA = 1.0;
    METALLIC = 0.0;
    ROUGHNESS = 0.12;
    SPECULAR = 0.5;
    CLEARCOAT = 0.35;
    CLEARCOAT_ROUGHNESS = 0.1;
}
