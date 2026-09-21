shader_type sky;

uniform sampler2D panorama : filter_linear_mipmap, repeat_enable;
uniform bool panorama_is_srgb = false;
// An authored LDR sky needs an explicit artistic radiance scale, not an HDR claim.
uniform float panorama_energy = 1.0;
// Artistic cloud radiance reconstruction for the authored LDR panorama.
// One is unchanged; this is not a measured HDR conversion.
uniform float cloud_radiance_gain : hint_range(1.0, 4.0) = 1.0;
// Periodic overlap for authored skies without a captured solar disk.
// Captured HDR panoramas retain zero to preserve their angular calibration.
uniform float panorama_seam_overlap : hint_range(0.0, 0.1) = 0.0;
// Rotate background and radiance together; direct sunlight must use the same yaw.
uniform float panorama_yaw = 0.0;
// Optional sky-only study: extend a clear source latitude over its landscape.
// -1 preserves the full panorama. This is not a recovered horizon or atmosphere.
uniform float minimum_source_y : hint_range(-1.0, 1.0) = -1.0;
// Appearance tuning, not calibrated radiometry. Background exposure is separate.
uniform float background_energy : hint_range(0.0, 2.0) = 0.7;
uniform float lighting_energy : hint_range(0.0, 2.0) = 0.45;
// Distant ground contribution for an authored sky with no captured landscape.
// Linear radiance estimate, not a measured irradiance or local reflection probe.
uniform bool use_ground_radiance = false;
uniform vec3 ground_radiance = vec3(0.148, 0.127, 0.098);
// Background color study only; does not alter environment lighting radiance.
uniform float background_saturation : hint_range(0.0, 2.0) = 1.0;

// Optional rectilinear detail source in fixed world coordinates.
uniform bool sky_patch_enabled = false;
uniform sampler2D sky_patch : filter_linear_mipmap, repeat_disable;
uniform vec3 sky_patch_forward = vec3(0.0, 0.70710678, 0.70710678);
uniform vec3 sky_patch_right = vec3(-1.0, 0.0, 0.0);
uniform vec3 sky_patch_up = vec3(0.0, 0.70710678, -0.70710678);
uniform vec2 sky_patch_tan_half_fov = vec2(1.0, 1.0);
uniform float sky_patch_feather = 0.15;

vec3 sample_panorama(vec2 uv, vec2 dx, vec2 dy) {
    vec3 value = textureGrad(panorama, uv, dx, dy).rgb;
    if (panorama_is_srgb) {
        value = mix(value / 12.92, pow((value + vec3(0.055)) / 1.055, vec3(2.4)),
                    step(vec3(0.04045), value));
    }
    return value;
}

void sky() {
    vec3 direction = normalize(EYEDIR);
    vec2 uv = vec2(fract((atan(direction.x, direction.z) - panorama_yaw) / TAU + 0.5),
                   acos(clamp(max(direction.y, minimum_source_y), -1.0, 1.0)) / PI);
    // Longitude wraps by one at atan's branch cut. Implicit derivatives treat
    // that jump as a huge footprint and select unrelated coarse mip levels.
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    if (panorama_seam_overlap > 0.0) {
        uv.x *= 1.0 - panorama_seam_overlap;
        dx.x *= 1.0 - panorama_seam_overlap;
        dy.x *= 1.0 - panorama_seam_overlap;
        COLOR = sample_panorama(uv, dx, dy);
        if (uv.x < panorama_seam_overlap) {
            vec3 end = sample_panorama(uv + vec2(1.0 - panorama_seam_overlap, 0.0), dx, dy);
            COLOR = mix(end, COLOR, smoothstep(0.0, panorama_seam_overlap, uv.x));
        }
    } else {
        COLOR = sample_panorama(uv, dx, dy);
    }
    // The enable branch is uniform. Compute derivatives before the spatial
    // coverage branch, including at the projection's front hemisphere boundary.
    if (sky_patch_enabled) {
        float depth = dot(direction, sky_patch_forward);
        vec2 projected = vec2(dot(direction, sky_patch_right), -dot(direction, sky_patch_up));
        vec2 patch_uv = 0.5 + projected / (2.0 * max(depth, 0.0001) * sky_patch_tan_half_fov);
        vec2 patch_dx = dFdx(patch_uv);
        vec2 patch_dy = dFdy(patch_uv);
        vec2 edge = min(patch_uv, vec2(1.0) - patch_uv);
        float coverage = smoothstep(0.0, sky_patch_feather, min(edge.x, edge.y));
        if (depth > 0.0 && coverage > 0.0) {
            vec3 patch = textureGrad(sky_patch, clamp(patch_uv, vec2(0.0), vec2(1.0)), patch_dx, patch_dy).rgb;
            patch = mix(patch / 12.92, pow((patch + vec3(0.055)) / 1.055, vec3(2.4)),
                        step(vec3(0.04045), patch));
            COLOR = mix(COLOR, patch, coverage);
        }
    }
    float source_luminance = dot(COLOR, vec3(0.2126, 0.7152, 0.0722));
    float source_chroma = (max(max(COLOR.r, COLOR.g), COLOR.b) - min(min(COLOR.r, COLOR.g), COLOR.b)) / max(source_luminance, 0.001);
    float cloud = smoothstep(0.30, 0.80, source_luminance)
                * (1.0 - smoothstep(0.20, 0.65, source_chroma));
    COLOR *= panorama_energy * mix(1.0, cloud_radiance_gain, cloud);
    if (AT_CUBEMAP_PASS) {
        if (use_ground_radiance) {
            float ground_weight = 1.0 - smoothstep(-0.12, 0.02, direction.y);
            COLOR = mix(COLOR, ground_radiance, ground_weight);
        }
        COLOR *= lighting_energy;
    } else {
        float luminance = dot(COLOR, vec3(0.2126, 0.7152, 0.0722));
        COLOR = max(mix(vec3(luminance), COLOR, background_saturation), vec3(0.0));
        COLOR *= background_energy;
    }
}
