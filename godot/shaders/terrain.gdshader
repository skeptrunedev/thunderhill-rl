shader_type spatial;
render_mode cull_disabled;
uniform sampler2D grass_color : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform sampler2D soil_color : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
// Data texture stores linear relative color gains divided by two, not sRGB albedo.
uniform sampler2D terrain_macro : filter_linear_mipmap_anisotropic, repeat_disable;
// Independent data channels: RGB is fine color gain / 2, alpha is grass coverage.
uniform sampler2D terrain_detail : filter_linear_mipmap_anisotropic, repeat_disable;
// Nonrepeating authored region data: target coverage, pale straw weight, unused, blend.
uniform sampler2D field_surface_map : filter_linear_mipmap, repeat_disable;
uniform vec2 field_surface_origin;
uniform vec2 field_surface_size = vec2(1.0);
uniform float field_surface_strength : hint_range(0.0, 1.0) = 0.0;
// Historical aerial contrast inside the independently authored straw region.
uniform float mapped_aerial_contrast : hint_range(0.0, 1.0) = 1.0;
// Regional version of the independently reviewed photographic field study.
uniform bool photographic_field_enabled = false;
// Optional two-material composition, using the existing authored flattened corridor.
uniform bool photographic_sparse_enabled = false;
// Optional compensation for contrast lost when independent source patches blend.
// Study: rank overlapping straw samples instead of averaging their structure.
// Luminance is only an artistic height proxy, not measured geometry.
// Opt in material composition along original curved mowing coordinates.
uniform bool photographic_curved_uv = true;
// Dense source runs horizontally; sparse source wisps run vertically.
// Isolate their orientation before changing composition or physical scale.
uniform bool photographic_sparse_aligned = false;
// Opt in triangular patch blend, fixed source scale and preserved linear variance.
uniform bool photographic_stochastic = false;
uniform float photographic_directional_composition : hint_range(0.0, 1.0) = 0.0;
uniform float photographic_height_blend : hint_range(0.0, 1.0) = 0.0;
uniform float photographic_contrast_preservation : hint_range(0.0, 1.0) = 0.0;
// Keep the uncertain corridor treatment isolated from the mixed base material.
uniform float photographic_disturbance_strength : hint_range(0.0, 1.0) = 0.0;
uniform sampler2D photographic_sparse : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
// Independent preview of retained large field structure over the new detail.
uniform float photographic_structure_strength : hint_range(0.0, 1.0) = 0.0;
uniform sampler2D photographic_field : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform vec3 pale_straw_gain = vec3(1.40, 1.48, 1.55);
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
// Continuous mowing pass study; center fitted to the historical Turn 2 bend.
// Spacing and contrast are authored estimates, not recovered mowing records.
uniform float mowing_band_strength : hint_range(0.0, 1.0) = 1.0;
// Interrupted fine straw streaks follow the same continuous mowing coordinates.
uniform float mowing_detail_strength : hint_range(0.0, 1.0) = 1.0;
uniform vec2 mowing_bend_center = vec2(306.70854, 686.54039);
// Study only: retain centered swath color inside the authored field map.
// Zero preserves the existing regional appearance; no bare patches are restored.
uniform float retained_swath_strength : hint_range(0.0, 1.0) = 0.0;
// Original appearance parameters, not measured plant height or reflectance.
uniform float grass_tile_m = 0.5;
// One retains full random rotation; zero preserves the source straw direction.
uniform float grass_rotation_spread : hint_range(0.0, 1.0) = 1.0;
// Independent patch scale study; zero preserves the production source scale.
uniform float grass_scale_spread : hint_range(0.0, 1.0) = 0.0;
uniform float grass_relief_m = 0.005;
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
// Fragment to light coverage for the isolated canopy study only.
varying float canopy_coverage;

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

vec3 ground_surface(sampler2D color_source, vec2 uv, float rotation_spread, float scale_spread, float contrast_preservation, float height_blend) {
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
    vec3 blended = mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
    if (contrast_preservation > 0.0) {
        vec4 weights = vec4((1.0 - w.x) * (1.0 - w.y), w.x * (1.0 - w.y),
                            (1.0 - w.x) * w.y, w.x * w.y);
        ivec2 dimensions = textureSize(color_source, 0);
        float last_mip = floor(log2(float(max(dimensions.x, dimensions.y))));
        vec3 mean_color = textureLod(color_source, vec2(0.5), last_mip).rgb;
        // Exact for independent equal-variance samples before clamping. The
        // transformed source patches can correlate, so this is an approximation.
        float gain = mix(1.0, inversesqrt(dot(weights, weights)), contrast_preservation);
        blended = clamp(mean_color + (blended - mean_color) * gain, vec3(0.0), vec3(1.0));
    }
    if (height_blend > 0.0) {
        vec4 base = vec4((1.0 - w.x) * (1.0 - w.y), w.x * (1.0 - w.y),
                         (1.0 - w.x) * w.y, w.x * w.y);
        vec3 luminance = vec3(0.2126, 0.7152, 0.0722);
        vec4 rank = vec4(dot(a, luminance), dot(b, luminance),
                         dot(c, luminance), dot(d, luminance)) + base;
        float peak = max(max(rank.x, rank.y), max(rank.z, rank.w));
        vec4 selected = smoothstep(vec4(peak - 0.20), vec4(peak), rank) * base;
        selected /= max(dot(selected, vec4(1.0)), 0.00001);
        vec3 layered = a * selected.x + b * selected.y + c * selected.z + d * selected.w;
        // Revert smoothly to the filtered blend when source structure is unresolved.
        float visibility = 1.0 - smoothstep(0.015, 0.09, max(length(dx), length(dy)));
        blended = mix(blended, layered, height_blend * visibility);
    }
    return blended;
}

// Three shared vertex samples per triangle replace four overlapping square
// patches. Translation changes phase without changing photographed world scale.
// Cubic weights fade the third sample smoothly at an edge; shared vertex seeds
// keep color continuous. Explicit gradients ignore the discontinuous cell IDs.
vec3 photographic_triangle(sampler2D color_source, vec2 uv) {
    mat2 skew = mat2(vec2(1.0, -0.57735026919), vec2(0.0, 1.15470053838));
    mat2 unskew = mat2(vec2(1.0, 0.5), vec2(0.0, 0.86602540378));
    vec2 grid = skew * uv;
    vec2 cell = floor(grid);
    vec2 f = fract(grid);
    vec2 va = cell;
    vec2 vb = cell + vec2(1.0, 0.0);
    vec2 vc = cell + vec2(0.0, 1.0);
    vec3 weights = vec3(1.0 - f.x - f.y, f.x, f.y);
    if (f.x + f.y > 1.0) {
        va = cell + vec2(1.0);
        weights = vec3(f.x + f.y - 1.0, 1.0 - f.y, 1.0 - f.x);
    }
    weights = weights * weights * (vec3(3.0) - 2.0 * weights);
    weights /= dot(weights, vec3(1.0));
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    vec3 a = ground_patch(color_source, uv, dx, dy, unskew * va, 0.0, 0.0);
    vec3 b = ground_patch(color_source, uv, dx, dy, unskew * vb, 0.0, 0.0);
    vec3 c = ground_patch(color_source, uv, dx, dy, unskew * vc, 0.0, 0.0);
    vec3 blended = a * weights.x + b * weights.y + c * weights.z;
    ivec2 dimensions = textureSize(color_source, 0);
    float last_mip = floor(log2(float(max(dimensions.x, dimensions.y))));
    vec3 mean_color = textureLod(color_source, vec2(0.5), last_mip).rgb;
    // Approximate variance correction in linear light. Correlated image samples
    // limit its accuracy; this is an appearance experiment, not measured albedo.
    float resolved = 1.0 - smoothstep(0.015, 0.09, max(length(dx), length(dy)));
    float gain = mix(1.0, inversesqrt(dot(weights, weights)), resolved);
    return clamp(mean_color + (blended - mean_color) * gain, vec3(0.0), vec3(1.0));
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

vec2 mowing_coordinates(vec2 point) {
    vec2 offset = point - mowing_bend_center;
    float across = length(vec2(max(offset.x, 0.0), offset.y));
    float along = offset.x;
    if (offset.x > 0.0) {
        along = atan(offset.x, offset.y) * across;
    } else if (offset.y < 0.0) {
        along = PI * across - offset.x;
    }
    return vec2(across, along);
}

float mowing_gain(vec2 point) {
    if (mowing_band_strength <= 0.0 && mowing_detail_strength <= 0.0) {
        return 1.0;
    }
    // Distance to a westward ray gives continuous straight passes joining
    // concentric passes around the east bend, without polyline corners.
    vec2 mowing_offset = point - mowing_bend_center;
    float mowing_distance = length(vec2(max(mowing_offset.x, 0.0), mowing_offset.y));
    float mowing_region = smoothstep(140.0, 175.0, point.x) * smoothstep(14.0, 30.0, mowing_distance)
                * (1.0 - smoothstep(94.0, 104.0, mowing_distance));
    // Arc length joins the two straight arms continuously. Local interruptions
    // replace coherent sinusoidal rings; dimensions are appearance estimates.
    float along = mowing_offset.x;
    if (mowing_offset.x > 0.0) {
        along = atan(mowing_offset.x, mowing_offset.y) * mowing_distance;
    } else if (mowing_offset.y < 0.0) {
        along = PI * mowing_distance - mowing_offset.x;
    }
    vec2 streak_uv = vec2(mowing_distance, along);
    float mowing_bands = coverage_noise(streak_uv / vec2(1.3, 5.0)) * 0.20;
    float streaks = coverage_noise(streak_uv / vec2(0.18, 3.5)) * 0.36
                 + coverage_noise(streak_uv / vec2(0.55, 8.0) + vec2(17.0, 3.0)) * 0.24;
    return 1.0 + mowing_region * (mowing_bands * mowing_band_strength
                                  + streaks * mowing_detail_strength);
}

// Only the pale field annotation receives the replacement. The darker western
// annotation retains its existing treatment. Bounds prevent clamped edge bleed.
float photographic_weight(vec2 uv) {
    vec4 field = texture(field_surface_map, uv);
    float inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
    return field.a * smoothstep(0.2, 0.6, field.g) * inside;
}

void fragment() {
    // Original texture scale is an appearance estimate, not a surveyed measure.
    vec3 grass = ground_surface(grass_color, world_pos.xz / grass_tile_m, grass_rotation_spread, grass_scale_spread, 0.0, 0.0);
    vec3 soil = ground_surface(soil_color, world_pos.xz / soil_tile_m, 1.0, 0.0, 0.0, 0.0);
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
    // Reuse the existing elongated field pattern throughout the mapped region.
    // This orientation and pattern are original appearance estimates, not
    // reconstructed mowing passes. The regional map below limits their extent.
    vec3 swath_contrast = vec3(0.29, 0.27, 0.22) / vec3(1.105, 1.095, 1.07);
    float centered_field = smoothstep(-0.23, 0.23, swaths) - 0.5;
    vec3 retained_swath_gain = vec3(1.0) + centered_field * swath_contrast;
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
        // Symmetric thresholds give neutral response at unresolved rows = 0.
        // Divide the existing endpoint difference by its midpoint, retaining
        // contrast without its overall brightening. The expected mean is one
        // for a symmetric noise distribution, not a measured finite-map mean.
        float centered_rows = smoothstep(-0.23, 0.23, rows) - 0.5;
        // Blend toward curved structure only where the existing guide supports
        // it, avoiding radial patterns beyond its clamped endpoint coordinates.
        retained_swath_gain = mix(retained_swath_gain,
                                 vec3(1.0) + centered_rows * swath_contrast,
                                 straw_abundance);
    }
    grass_coverage *= 1.0 - sparse_straw * field_patch_strength * (1.0 - 0.5 * straw_abundance);
    if (field_surface_strength > 0.0) {
        vec2 field_map_uv = (world_pos.xz - field_surface_origin) / field_surface_size;
        vec4 mapped_field = texture(field_surface_map, field_map_uv);
        float inside_map = step(0.0, field_map_uv.x) * step(field_map_uv.x, 1.0)
                         * step(0.0, field_map_uv.y) * step(field_map_uv.y, 1.0);
        float mapped_weight = mapped_field.a * field_surface_strength * inside_map;
        float aerial_weight = mix(1.0, mapped_aerial_contrast, mapped_weight);
        macro_gain = mix(vec3(1.0), macro_gain, aerial_weight);
        detail_gain = mix(vec3(1.0), detail_gain, aerial_weight);
        // Preserve the interrupted shoulder blend when applying regional straw.
        // Replacing it with raw detail alpha restored a smooth roadwide ribbon.
        // Flattened straw reaches farther into the existing visual transition.
        // Keep the zero and full coverage endpoints and explicit bare corridors.
        float mapped_base = 1.0 - (1.0 - detail.a) * (1.0 - detail.a);
        float mapped_transition = 4.0 * mapped_base * (1.0 - mapped_base);
        float mapped_shoulder = clamp(mapped_base + mapped_transition * coverage_variation, 0.0, 1.0);
        grass_coverage = mix(grass_coverage, mapped_shoulder * mapped_field.r, mapped_weight);
        vec3 mapped_straw_gain = mix(vec3(1.0), pale_straw_gain, mapped_field.g);
        mapped_straw_gain *= mix(vec3(1.0), retained_swath_gain, retained_swath_strength);
        straw_color_gain = mix(straw_color_gain, mapped_straw_gain, mapped_weight);
    }
    // Mowing structure belongs to the field surface, including sparse earth
    // between cut strands. Applying it only to grass erased it at the shoulder.
    float field_structure = mowing_gain(world_pos.xz);
    float retained_grass;
    vec4 region = cleared_field(world_pos.xz, retained_grass);
    float corridor = region.a;
    // Bare access corridors do not inherit the surrounding cut field pattern.
    field_structure = mix(field_structure, 1.0, corridor * (1.0 - retained_grass));
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
    canopy_coverage = 1.0 - soil_weight;
    ALBEDO = mix(grass * straw_color_gain, soil * mix(soil_value, 1.0, corridor * (1.0 - retained_grass)), soil_weight) * ground_tint * macro_gain * detail_gain * mix(vec3(1.0), region.rgb, corridor) * field_structure;
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
    if (photographic_field_enabled) {
        // Expand the authored field footprint three metres toward the shoulder.
        // Terrain only: the independent asphalt surface is never replaced.
        vec2 field_uv = (world_pos.xz - field_surface_origin) / field_surface_size;
        vec2 radius_uv = vec2(3.0) / field_surface_size;
        float mask = photographic_weight(field_uv);
        mask = max(mask, photographic_weight(field_uv + vec2(radius_uv.x, 0.0)));
        mask = max(mask, photographic_weight(field_uv - vec2(radius_uv.x, 0.0)));
        mask = max(mask, photographic_weight(field_uv + vec2(0.0, radius_uv.y)));
        mask = max(mask, photographic_weight(field_uv - vec2(0.0, radius_uv.y)));
        mask *= field_surface_strength * (1.0 - corridor * (1.0 - retained_grass));
        // Source straw runs mainly along image x. Align x with mowing travel;
        // stretch along travel and compress across for flattened swaths.
        vec2 photo_uv = photographic_curved_uv ? mowing_coordinates(world_pos.xz).yx / vec2(6.0, 2.0) : world_pos.xz / 4.0;
        vec3 photographic = photographic_stochastic ? photographic_triangle(photographic_field, photo_uv) : ground_surface(photographic_field, photo_uv, 0.08, 0.25, photographic_contrast_preservation, photographic_height_blend);
        vec3 photographic_gain = mix(vec3(1.0), retained_swath_gain * field_structure, photographic_structure_strength);
        float photographic_height = dot(photographic, vec3(0.2126, 0.7152, 0.0722)) * 0.004;
        float photo_dx = dFdx(photographic_height);
        float photo_dy = dFdy(photographic_height);
        float photo_roughness = 0.93;
        if (photographic_sparse_enabled) {
            vec2 sparse_uv = photographic_sparse_aligned ? vec2(photo_uv.y, -photo_uv.x) : photo_uv;
            vec3 sparse = photographic_stochastic ? photographic_triangle(photographic_sparse, sparse_uv) : ground_surface(photographic_sparse, sparse_uv, 0.08, 0.25, photographic_contrast_preservation, photographic_height_blend);
            // Authored reference corridor is independent of driving/friction data.
            // Material proportions are artistic estimates, not classified land cover.
            // Reuse the corridor's authored interruptions instead of a uniform band.
            float disturbed = clamp((1.0 - dot(region.rgb, vec3(0.2126, 0.7152, 0.0722))) / 0.42, 0.0, 1.0);
            float sparse_weight = mix(0.60, 0.95, corridor * retained_grass * disturbed * photographic_disturbance_strength);
            // Replace uniform material proportions with interrupted curved
            // passes. Scales and placement are estimates, not mowing records.
            vec2 lay_uv = mowing_coordinates(world_pos.xz);
            float lay = coverage_noise(lay_uv / vec2(1.7, 16.0)) * 0.75
                      + coverage_noise(lay_uv / vec2(0.45, 6.0) + vec2(17.0, 31.0)) * 0.25;
            float lay_weight = smoothstep(-0.18, 0.18, lay);
            float composed_sparse = mix(0.22, 0.98, lay_weight);
            sparse_weight = mix(sparse_weight, composed_sparse, photographic_directional_composition);
            float sparse_height = dot(sparse, vec3(0.2126, 0.7152, 0.0722)) * 0.0015;
            photographic = mix(photographic, sparse, sparse_weight);
            photo_dx = mix(photo_dx, dFdx(sparse_height), sparse_weight);
            photo_dy = mix(photo_dy, dFdy(sparse_height), sparse_weight);
            photo_roughness = mix(photo_roughness, 0.96, sparse_weight);
        }
        canopy_coverage = mix(canopy_coverage, 1.0, mask);
        ALBEDO = mix(ALBEDO, photographic * 0.55 * photographic_gain, mask);
        ROUGHNESS = mix(ROUGHNESS, photo_roughness, mask);
        height_dx = mix(height_dx, photo_dx, mask);
        height_dy = mix(height_dy, photo_dy, mask);
    }
    vec3 dx = dFdx(VERTEX);
    vec3 dy = dFdy(VERTEX);
    vec3 rx = cross(dy, NORMAL);
    vec3 ry = cross(NORMAL, dx);
    float determinant = dot(dx, rx);
    vec3 gradient = height_dx * rx + height_dy * ry;
    NORMAL = normalize(NORMAL - sign(determinant) * gradient / max(abs(determinant), 0.000000000001));
}
