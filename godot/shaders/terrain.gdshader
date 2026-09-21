shader_type spatial;
render_mode cull_disabled;
uniform sampler2D grass_color : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform sampler2D soil_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D soil_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D soil_roughness : filter_linear_mipmap_anisotropic, repeat_enable;
// Data texture stores linear relative color gains divided by two, not sRGB albedo.
uniform sampler2D terrain_macro : filter_linear_mipmap, repeat_disable;
// Independent data channels: RGB is fine color gain / 2, alpha is grass coverage.
uniform sampler2D terrain_detail : filter_linear_mipmap, repeat_disable;
uniform vec2 macro_origin;
uniform vec2 macro_size;
uniform vec3 ground_tint = vec3(1.0);
// Original appearance parameters, not measured plant height or reflectance.
uniform float grass_tile_m = 2.0;
uniform float grass_relief_m = 0.010;
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

vec3 grass_patch(vec2 uv, vec2 dx, vec2 dy, vec2 lattice) {
    vec3 seed = patch_hash(lattice);
    float angle = seed.z * 6.28318530718;
    mat2 rotation = mat2(vec2(cos(angle), sin(angle)), vec2(-sin(angle), cos(angle)));
    vec2 transformed = rotation * uv + seed.xy * 2.0;
    vec2 mirrored = vec2(1.0) - abs(vec2(1.0) - mod(transformed, vec2(2.0)));
    vec2 mirror_sign = mix(vec2(1.0), vec2(-1.0), step(vec2(1.0), mod(transformed, vec2(2.0))));
    return textureGrad(grass_color, mirrored, (rotation * dx) * mirror_sign, (rotation * dy) * mirror_sign).rgb;
}

vec3 grass_surface(vec2 uv) {
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
    vec3 a = grass_patch(uv, dx, dy, cell);
    vec3 b = grass_patch(uv, dx, dy, cell + vec2(1.0, 0.0));
    vec3 c = grass_patch(uv, dx, dy, cell + vec2(0.0, 1.0));
    vec3 d = grass_patch(uv, dx, dy, cell + vec2(1.0));
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

void fragment() {
    // Original texture scale is an appearance estimate, not a surveyed measure.
    vec3 grass = grass_surface(world_pos.xz / grass_tile_m);
    vec3 soil = texture(soil_color, world_pos.xz / 1.3).rgb;
    vec2 macro_uv = (world_pos.xz - macro_origin) / macro_size;
    vec2 edge_distance = min(macro_uv, vec2(1.0) - macro_uv) * macro_size;
    float macro_weight = smoothstep(0.0, 30.0, min(edge_distance.x, edge_distance.y));
    vec3 macro_gain = mix(vec3(1.0), texture(terrain_macro, macro_uv).rgb * 2.0, macro_weight);
    vec4 detail = texture(terrain_detail, macro_uv);
    vec3 detail_gain = mix(vec3(1.0), detail.rgb * 2.0, macro_weight);
    float soil_weight = (1.0 - detail.a) * macro_weight;
    ALBEDO = mix(grass, soil, soil_weight) * ground_tint * macro_gain * detail_gain;
    // A single albedo does not contain measured PBR maps. Bright straw over
    // darker soil supplies an explicitly artistic relief/roughness estimate.
    // Derivatives of the filtered sample naturally remove unresolved relief.
    float straw = dot(grass, vec3(0.2126, 0.7152, 0.0722));
    float grass_r = mix(0.98, 0.84, smoothstep(0.10, 0.48, straw));
    ROUGHNESS = clamp(mix(grass_r, texture(soil_roughness, world_pos.xz / 1.3).r, soil_weight), 0.65, 1.0);
    float height = straw * grass_relief_m;
    vec3 dx = dFdx(VERTEX);
    vec3 dy = dFdy(VERTEX);
    vec3 rx = cross(dy, NORMAL);
    vec3 ry = cross(NORMAL, dx);
    float determinant = dot(dx, rx);
    vec3 gradient = dFdx(height) * rx + dFdy(height) * ry;
    vec3 grass_n = normalize(NORMAL - sign(determinant) * gradient / max(abs(determinant), 0.000000000001));
    vec3 soil_n = texture(soil_normal, world_pos.xz / 1.3).rgb * 2.0 - 1.0;
    soil_n = normalize(mix(vec3(0.0, 0.0, 1.0), soil_n, 0.45));
    vec3 soil_view_normal = normalize(mat3(TANGENT, BINORMAL, NORMAL) * soil_n);
    NORMAL = normalize(mix(grass_n, soil_view_normal, soil_weight));
}
