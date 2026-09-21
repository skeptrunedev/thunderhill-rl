shader_type spatial;
render_mode cull_disabled;
// Experimental photographic field composition; no aerial or sine band overlays.
uniform sampler2D field_color : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform float tile_m = 4.0;
uniform float color_gain = 0.55;
uniform float relief_m = 0.004;
varying vec3 world_pos;
void vertex() { world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz; }
vec3 patch_hash(vec2 lattice) {
    vec3 p = fract(vec3(lattice.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yzz) * p.zyx);
}

vec3 ground_patch(sampler2D color_source, vec2 uv, vec2 dx, vec2 dy, vec2 lattice, float rotation_spread, float scale_spread) {
    vec3 seed = patch_hash(lattice);
    float angle = seed.z * 6.28318530718 * rotation_spread;
    mat2 rotation = mat2(vec2(cos(angle), sin(angle)), vec2(-sin(angle), cos(angle)));
    // Shared lattice seeds keep transforms continuous across cell boundaries.
    // Scale around the patch anchor rather than moving distant world coordinates.
    float patch_scale = mix(1.0, mix(0.55, 1.65, seed.x), scale_spread);
    vec2 transformed = rotation * ((uv - lattice) * patch_scale + lattice) + seed.xy * 2.0;
    vec2 mirrored = vec2(1.0) - abs(vec2(1.0) - mod(transformed, vec2(2.0)));
    vec2 mirror_sign = mix(vec2(1.0), vec2(-1.0), step(vec2(1.0), mod(transformed, vec2(2.0))));
    return textureGrad(color_source, mirrored, (rotation * dx) * patch_scale * mirror_sign, (rotation * dy) * patch_scale * mirror_sign).rgb;
}

vec3 ground_surface(sampler2D color_source, vec2 uv, float rotation_spread, float scale_spread) {
    vec2 cell = floor(uv);
    vec2 f = fract(uv);
    // Quintic interpolation has zero first and second derivatives at cell
    // borders. This matters because relief differentiates the blended albedo.
    vec2 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    // Concentrate overlap to retain straw contrast; preserve smooth endpoints.
    vec2 lo = (vec2(1.0) - w) * (vec2(1.0) - w);
    vec2 hi = w * w;
    w = hi / (lo + hi);
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    vec3 a = ground_patch(color_source, uv, dx, dy, cell, rotation_spread, scale_spread);
    vec3 b = ground_patch(color_source, uv, dx, dy, cell + vec2(1.0, 0.0), rotation_spread, scale_spread);
    vec3 c = ground_patch(color_source, uv, dx, dy, cell + vec2(0.0, 1.0), rotation_spread, scale_spread);
    vec3 d = ground_patch(color_source, uv, dx, dy, cell + vec2(1.0), rotation_spread, scale_spread);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

void fragment() {
    vec3 field = ground_surface(field_color, world_pos.xz / tile_m, 0.08, 0.25);
    ALBEDO = field * color_gain;
    ROUGHNESS = 0.93;
    float height = dot(field, vec3(0.2126, 0.7152, 0.0722)) * relief_m;
    vec3 dx = dFdx(VERTEX);
    vec3 dy = dFdy(VERTEX);
    vec3 rx = cross(dy, NORMAL);
    vec3 ry = cross(NORMAL, dx);
    float determinant = dot(dx, rx);
    vec3 gradient = dFdx(height) * rx + dFdy(height) * ry;
    NORMAL = normalize(NORMAL - sign(determinant) * gradient / max(abs(determinant), 0.000000000001));
}
