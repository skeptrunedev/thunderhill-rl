shader_type spatial;
render_mode cull_disabled;
uniform sampler2D grass_color : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform sampler2D soil_color : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
// Data texture stores linear relative color gains divided by two, not sRGB albedo.
uniform sampler2D terrain_macro : filter_linear_mipmap_anisotropic, repeat_disable;
// Independent data channels: RGB is fine color gain / 2, alpha is grass coverage.
uniform sampler2D terrain_detail : filter_linear_mipmap_anisotropic, repeat_disable;
uniform vec2 macro_origin;
uniform vec2 macro_size;
uniform vec3 ground_tint = vec3(1.0);
// Appearance control for historical field detail, not recovered reflectance.
uniform float detail_gain_exponent = 1.0;
// Optional appearance study, not land cover labels. Current aerial tone trial
// misses the uniform foreground problem, so production keeps it disabled.
uniform float field_soil_strength : hint_range(0.0, 1.0) = 0.0;
// Original interrupted straw swaths, not measured mowing or vegetation data.
uniform float field_patch_strength : hint_range(0.0, 1.0) = 0.5;
// Original appearance parameters, not measured plant height or reflectance.
uniform float grass_tile_m = 1.0;
// One retains full random rotation; zero preserves the source straw direction.
uniform float grass_rotation_spread : hint_range(0.0, 1.0) = 1.0;
uniform float grass_relief_m = 0.010;
// Original fine soil material, with estimated scale and shallow relief.
uniform float soil_tile_m = 1.0;
uniform float soil_relief_m = 0.0015;
uniform float soil_value : hint_range(0.1, 1.0) = 0.7;
varying vec3 world_pos;

void vertex() {
    world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

// Each shared lattice vertex owns one deterministic source transform. Adjacent
// cells therefore agree along their common edge; no per-cell hard swaps.
vec3 patch_hash(vec2 lattice) {
    vec3 p = fract(vec3(lattice.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yzz) * p.zyx);
}

// Original coverage variation, not a surveyed vegetation mask. Fade noise when
// its world space footprint becomes unresolved; preserve the map endpoints.
float coverage_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    vec2 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float value = mix(mix(patch_hash(cell).x, patch_hash(cell + vec2(1.0, 0.0)).x, w.x),
                      mix(patch_hash(cell + vec2(0.0, 1.0)).x, patch_hash(cell + vec2(1.0)).x, w.x), w.y);
    float visibility = 1.0 - smoothstep(0.25, 1.0, length(fwidth(p)));
    return (value - 0.5) * visibility;
}

vec3 ground_patch(sampler2D color_source, vec2 uv, vec2 dx, vec2 dy, vec2 lattice, float rotation_spread) {
    vec3 seed = patch_hash(lattice);
    float angle = seed.z * 6.28318530718 * rotation_spread;
    mat2 rotation = mat2(vec2(cos(angle), sin(angle)), vec2(-sin(angle), cos(angle)));
    vec2 transformed = rotation * uv + seed.xy * 2.0;
    vec2 mirrored = vec2(1.0) - abs(vec2(1.0) - mod(transformed, vec2(2.0)));
    vec2 mirror_sign = mix(vec2(1.0), vec2(-1.0), step(vec2(1.0), mod(transformed, vec2(2.0))));
    return textureGrad(color_source, mirrored, (rotation * dx) * mirror_sign, (rotation * dy) * mirror_sign).rgb;
}

vec3 ground_surface(sampler2D color_source, vec2 uv, float rotation_spread) {
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
    vec3 a = ground_patch(color_source, uv, dx, dy, cell, rotation_spread);
    vec3 b = ground_patch(color_source, uv, dx, dy, cell + vec2(1.0, 0.0), rotation_spread);
    vec3 c = ground_patch(color_source, uv, dx, dy, cell + vec2(0.0, 1.0), rotation_spread);
    vec3 d = ground_patch(color_source, uv, dx, dy, cell + vec2(1.0), rotation_spread);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

void fragment() {
    // Original texture scale is an appearance estimate, not a surveyed measure.
    vec3 grass = ground_surface(grass_color, world_pos.xz / grass_tile_m, grass_rotation_spread);
    vec3 soil = ground_surface(soil_color, world_pos.xz / soil_tile_m, 1.0);
    vec2 macro_uv = (world_pos.xz - macro_origin) / macro_size;
    vec2 edge_distance = min(macro_uv, vec2(1.0) - macro_uv) * macro_size;
    float macro_weight = smoothstep(0.0, 30.0, min(edge_distance.x, edge_distance.y));
    vec3 macro_gain = mix(vec3(1.0), texture(terrain_macro, macro_uv).rgb * 2.0, macro_weight);
    vec4 detail = texture(terrain_detail, macro_uv);
    vec3 detail_gain = mix(vec3(1.0), pow(max(detail.rgb * 2.0, vec3(0.001)), vec3(detail_gain_exponent)), macro_weight);
    float transition = 4.0 * detail.a * (1.0 - detail.a);
    float coverage_variation = coverage_noise(world_pos.xz / 2.4) * 0.9
                             + coverage_noise(world_pos.xz / 0.35) * 0.25;
    float grass_coverage = clamp(detail.a + transition * coverage_variation, 0.0, 1.0);
    // Darker accepted aerial regions expose some fine earth between straw.
    // Fade to full grass at neutral gain so missing aerial data adds no patches.
    float field_tone = dot(macro_gain, vec3(0.2126, 0.7152, 0.0722));
    float sparse_field = 1.0 - smoothstep(0.65, 1.0, field_tone);
    grass_coverage *= 1.0 - sparse_field * field_soil_strength;
    // Multiple scales interrupt elongated sparse areas instead of making
    // equally spaced stripes. Preserve the existing bare shoulder footprint.
    vec2 field_uv = mat2(vec2(0.94, 0.342), vec2(-0.342, 0.94)) * world_pos.xz;
    float swaths = coverage_noise(field_uv / vec2(2.2, 18.0)) * 0.65
                 + coverage_noise(field_uv / vec2(11.0, 13.0)) * 0.35;
    float sparse_straw = smoothstep(-0.20, 0.30, swaths);
    grass_coverage *= 1.0 - sparse_straw * field_patch_strength;
    float soil_weight = (1.0 - grass_coverage) * macro_weight;
    ALBEDO = mix(grass, soil * soil_value, soil_weight) * ground_tint * macro_gain * detail_gain;
    // A single albedo does not contain measured PBR maps. Bright straw over
    // darker soil supplies an explicitly artistic relief/roughness estimate.
    // Derivatives of the filtered sample naturally remove unresolved relief.
    float straw = dot(grass, vec3(0.2126, 0.7152, 0.0722));
    float grass_r = mix(0.98, 0.84, smoothstep(0.10, 0.48, straw));
    float soil_luminance = dot(soil, vec3(0.2126, 0.7152, 0.0722));
    float soil_r = mix(0.96, 0.88, smoothstep(0.10, 0.50, soil_luminance));
    ROUGHNESS = mix(grass_r, soil_r, soil_weight);
    // Blend material gradients after differentiating their heights. Coverage
    // changes material, not terrain elevation; do not differentiate its
    // footprint dependent visibility or invent a ridge at each blend edge.
    float height_dx = mix(dFdx(straw) * grass_relief_m, dFdx(soil_luminance) * soil_relief_m, soil_weight);
    float height_dy = mix(dFdy(straw) * grass_relief_m, dFdy(soil_luminance) * soil_relief_m, soil_weight);
    vec3 dx = dFdx(VERTEX);
    vec3 dy = dFdy(VERTEX);
    vec3 rx = cross(dy, NORMAL);
    vec3 ry = cross(NORMAL, dx);
    float determinant = dot(dx, rx);
    vec3 gradient = height_dx * rx + height_dy * ry;
    NORMAL = normalize(NORMAL - sign(determinant) * gradient / max(abs(determinant), 0.000000000001));
}
