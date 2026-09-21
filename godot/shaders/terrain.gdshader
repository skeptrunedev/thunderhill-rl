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
// Original curved mowing appearance, guided by the retained straw corridor.
uniform float straw_swath_strength : hint_range(0.0, 1.0) = 1.0;
// Original appearance parameters, not measured plant height or reflectance.
uniform float grass_tile_m = 1.0;
// One retains full random rotation; zero preserves the source straw direction.
uniform float grass_rotation_spread : hint_range(0.0, 1.0) = 1.0;
// Independent patch scale study; zero preserves the production source scale.
uniform float grass_scale_spread : hint_range(0.0, 1.0) = 0.0;
uniform float grass_relief_m = 0.010;
// Luminance ranks pale straw over soil for material blending, not measured height.
uniform float grass_height_blend : hint_range(0.0, 1.0) = 1.0;
// Estimated field microtopography for shading only, not surveyed contact heights.
// Rendered trials did not improve fidelity enough for production.
uniform float field_relief_m : hint_range(0.0, 0.1) = 0.0;
// Original fine soil material, with estimated scale and shallow relief.
uniform float soil_tile_m = 1.0;
uniform float soil_relief_m = 0.0015;
uniform float soil_value : hint_range(0.1, 1.0) = 0.7;
// Shared traced corridor capsules, also used to exclude standing stubble.
uniform int field_segment_count = 0;
uniform vec4 field_segments[16];
uniform vec2 field_sizes[16];
uniform float field_end_radii[16];
// Linear artistic color multipliers, independently authored per region.
uniform vec3 field_tints[16];
uniform vec2 field_tint_strengths[16];
// Flattened straw can retain its surface texture while removing upright blades.
uniform float field_grass_retentions[16];
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
float field_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    vec2 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float value = mix(mix(patch_hash(cell).x, patch_hash(cell + vec2(1.0, 0.0)).x, w.x),
                      mix(patch_hash(cell + vec2(0.0, 1.0)).x, patch_hash(cell + vec2(1.0)).x, w.x), w.y);
    return value - 0.5;
}

float coverage_noise(vec2 p) {
    float visibility = 1.0 - smoothstep(0.25, 1.0, length(fwidth(p)));
    return field_noise(p) * visibility;
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

vec4 cleared_field(vec2 point, out float grass_retention) {
    grass_retention = 0.0;
    float weight = 0.0;
    vec3 tint = vec3(1.0);
    for (int i = 0; i < field_segment_count; i++) {
        vec2 start = field_segments[i].xy;
        vec2 delta = field_segments[i].zw - start;
        float t = clamp(dot(point - start, delta) / dot(delta, delta), 0.0, 1.0);
        float distance_to_edge = length(point - start - delta * t) - mix(field_sizes[i].x, field_end_radii[i], t);
        float candidate = 1.0 - smoothstep(0.0, field_sizes[i].y, distance_to_edge);
        if (candidate > weight) {
            weight = candidate;
            tint = mix(vec3(1.0), field_tints[i], mix(field_tint_strengths[i].x, field_tint_strengths[i].y, t));
            grass_retention = field_grass_retentions[i];
        }
    }
    return vec4(tint, weight);
}

void fragment() {
    // Original texture scale is an appearance estimate, not a surveyed measure.
    vec3 grass = ground_surface(grass_color, world_pos.xz / grass_tile_m, grass_rotation_spread, grass_scale_spread);
    vec3 soil = ground_surface(soil_color, world_pos.xz / soil_tile_m, 1.0, 0.0);
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
    float straw_abundance = 0.0;
    vec3 straw_color_gain = vec3(1.0);
    if (straw_swath_strength > 0.0) {
        // The existing flattened straw corridors guide broad field appearance.
        // Bare access corridors (zero retained grass) do not seed mowing passes.
        float nearest_distance = 1000000.0;
        float nearest_along = 0.0;
        float nearest_side = 0.0;
        float cumulative_length = 0.0;
        float total_length = 0.0;
        for (int i = 0; i < field_segment_count; i++) {
            if (field_grass_retentions[i] > 0.0) {
                vec2 start = field_segments[i].xy;
                vec2 delta = field_segments[i].zw - start;
                float segment_length = length(delta);
                float t = clamp(dot(world_pos.xz - start, delta) / dot(delta, delta), 0.0, 1.0);
                vec2 offset = world_pos.xz - start - delta * t;
                float distance_to_guide = length(offset);
                if (distance_to_guide < nearest_distance) {
                    nearest_distance = distance_to_guide;
                    nearest_along = cumulative_length + t * segment_length;
                    nearest_side = dot(offset, vec2(delta.y, -delta.x) / segment_length);
                }
                cumulative_length += segment_length;
                total_length += segment_length;
            }
        }
        // Original bounded appearance study, not measured mowing dimensions.
        float field_weight = smoothstep(0.0, 2.0, nearest_side)
                           * (1.0 - smoothstep(24.0, 40.0, nearest_distance))
                           * smoothstep(0.0, 8.0, nearest_along)
                           * smoothstep(0.0, 8.0, total_length - nearest_along);
        vec2 mowing_uv = vec2(nearest_distance, nearest_along);
        mowing_uv.x += field_noise(vec2(nearest_along / 9.0, nearest_distance / 7.0)) * 0.8;
        float rows = coverage_noise(mowing_uv / vec2(1.1, 14.0)) * 0.65
                   + coverage_noise(mowing_uv / vec2(0.4, 6.0) + vec2(13.0, 7.0)) * 0.35;
        float pale_straw = smoothstep(-0.22, 0.24, rows);
        straw_abundance = straw_swath_strength * field_weight;
        straw_color_gain = mix(vec3(1.0), mix(vec3(0.96), vec3(1.25, 1.23, 1.18), pale_straw), straw_abundance);
        grass_coverage *= mix(1.0, mix(0.70, 1.0, pale_straw), straw_abundance);
    }
    grass_coverage *= 1.0 - sparse_straw * field_patch_strength * (1.0 - 0.5 * straw_abundance);
    float retained_grass;
    vec4 region = cleared_field(world_pos.xz, retained_grass);
    float corridor = region.a;
    grass_coverage *= 1.0 - corridor * (1.0 - retained_grass);
    float straw = dot(grass, vec3(0.2126, 0.7152, 0.0722));
    // The source already includes earth beneath the straw. Preserve resolved
    // pale strands instead of crossfading all of them into a second soil image.
    // Endpoint coverage remains exactly bare soil or full source material.
    float straw_rank = smoothstep(0.12, 0.55, straw);
    float layered_coverage = smoothstep(0.0, 1.0, 2.0 * grass_coverage - 1.0 + straw_rank);
    float resolved_straw = 1.0 - smoothstep(0.008, 0.05, length(fwidth(world_pos.xz)) / grass_tile_m);
    grass_coverage = mix(grass_coverage, layered_coverage, grass_height_blend * resolved_straw);
    float soil_weight = (1.0 - grass_coverage) * macro_weight;
    ALBEDO = mix(grass * straw_color_gain, soil * mix(soil_value, 1.0, corridor * (1.0 - retained_grass)), soil_weight) * ground_tint * macro_gain * detail_gain * mix(vec3(1.0), region.rgb, corridor);
    // A single albedo does not contain measured PBR maps. Bright straw over
    // darker soil supplies an explicitly artistic relief/roughness estimate.
    // Derivatives of the filtered sample naturally remove unresolved relief.
    float grass_r = mix(0.98, 0.84, smoothstep(0.10, 0.48, straw));
    float soil_luminance = dot(soil, vec3(0.2126, 0.7152, 0.0722));
    float soil_r = mix(0.96, 0.88, smoothstep(0.10, 0.50, soil_luminance));
    ROUGHNESS = mix(grass_r, soil_r, soil_weight);
    // Blend material gradients after differentiating their heights. Coverage
    // changes material, not terrain elevation; do not differentiate its
    // footprint dependent visibility or invent a ridge at each blend edge.
    float height_dx = mix(dFdx(straw) * grass_relief_m, dFdx(soil_luminance) * soil_relief_m, soil_weight);
    float height_dy = mix(dFdy(straw) * grass_relief_m, dFdy(soil_luminance) * soil_relief_m, soil_weight);
    // Differentiate continuous height first. Apply footprint and coverage
    // weights afterward so material transitions do not become raised ridges.
    if (field_relief_m > 0.0) {
        vec2 relief_uv = field_uv / vec2(0.42, 3.8);
        float field_height = (field_noise(relief_uv) * 0.70
                            + field_noise(field_uv / vec2(1.7, 8.5) + vec2(13.0, 37.0)) * 0.30) * field_relief_m;
        float relief_visibility = 1.0 - smoothstep(0.25, 1.0, length(fwidth(relief_uv)));
        float relief_weight = (1.0 - soil_weight) * relief_visibility;
        height_dx += dFdx(field_height) * relief_weight;
        height_dy += dFdy(field_height) * relief_weight;
    }
    vec3 dx = dFdx(VERTEX);
    vec3 dy = dFdy(VERTEX);
    vec3 rx = cross(dy, NORMAL);
    vec3 ry = cross(NORMAL, dx);
    float determinant = dot(dx, rx);
    vec3 gradient = height_dx * rx + height_dy * ry;
    NORMAL = normalize(NORMAL - sign(determinant) * gradient / max(abs(determinant), 0.000000000001));
}
