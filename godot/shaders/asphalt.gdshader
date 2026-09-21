shader_type spatial;
render_mode cull_disabled;
uniform float lap_length_m;
// Appearance study offset; independent of tire grip and surface identity.
uniform float roughness_offset : hint_range(-0.3, 0.2) = 0.0;
uniform sampler2D color_map : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D normal_map : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D rough_map : filter_linear_mipmap_anisotropic, repeat_enable;
uniform bool authored_surface = false;
uniform sampler2D authored_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform float authored_color_gain = 0.30;
// Independent strength for correlated broad albedo modulation.
// Neutral midpoint preserves the original midpoint response, not measured mean.
uniform float broad_tone_strength : hint_range(0.0, 1.0) = 0.25;
// Footage guided warm charcoal response, not a measured spectral reflectance.
uniform float authored_warmth : hint_range(0.0, 1.0) = 1.0;
// Smaller aggregate appearance selected against frame 40. Not a measured stone size.
uniform float authored_tile_m = 0.25;
// Preserve relief/tile ratio while changing the physical repeat scale.
uniform float authored_relief_m = 0.000109375;
// Original broad binder variation, not a surveyed rubber or racing line.
uniform float surface_variation_strength : hint_range(0.0, 1.0) = 0.35;
// Original longitudinal binder mottling, not a recovered tire or friction map.
uniform float binder_mottling_strength : hint_range(0.0, 1.0) = 0.25;
// Directional sheen study. Original directional material variation,
// not measured friction, rubber placement, or a recovered roughness scan.
uniform float directional_wear_strength : hint_range(0.0, 1.0) = 0.8;
// Frame 50 appearance study: local paving pass boundary, not surveyed or friction data.
uniform float paving_joint_strength : hint_range(0.0, 1.0) = 1.0;
uniform vec2 paving_joint_station_m = vec2(1263.588613, 1508.580953);
uniform float paving_joint_across = 0.46;
uniform float paving_joint_width_m = 0.018;
// Historical aerial tone, not a measured material or friction field.
uniform sampler2D pavement_tone : filter_linear_mipmap_anisotropic, repeat_enable;
// The historical trial did not improve reference fidelity enough for production.
uniform float pavement_tone_strength : hint_range(0.0, 1.0) = 0.0;

varying vec3 world_position;

void vertex() {
    world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

float pavement_hash(vec2 cell) {
    // Integer lattice hashing gives shared corners identical values. The sine
    // hash amplified floating point differences into visible long stripe seams.
    uvec2 lattice = uvec2(ivec2(cell));
    uint value = lattice.x * 1664525u + lattice.y * 1013904223u;
    value = (value ^ (value >> 16u)) * 2246822519u;
    value = (value ^ (value >> 13u)) * 3266489917u;
    value = value ^ (value >> 16u);
    return float(value & 16777215u) / 16777216.0;
}

float pavement_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(pavement_hash(cell), pavement_hash(cell + vec2(1.0, 0.0)), f.x),
               mix(pavement_hash(cell + vec2(0.0, 1.0)), pavement_hash(cell + vec2(1.0)), f.x), f.y);
}

float filtered_pavement_noise(vec2 p) {
    // Fade unresolved procedural cells instead of letting them crawl at distance.
    float footprint = max(length(dFdx(p)), length(dFdy(p)));
    return mix(pavement_noise(p), 0.5, smoothstep(0.25, 0.8, footprint));
}

void fragment() {
    // UV is measured in metres. The aggregate tile scale is an artistic estimate:
    // the publisher supplies no physical dimensions for Asphalt010.
    vec2 detail = UV * 1.6;
    // Reference 00:20 and 00:40 show coherent longitudinal surface variation,
    // not the previous strong isotropic cloud pattern. This is original material
    // detail, not a claim about the measured rubber line or repaving seams.
    float broad = pavement_noise(UV / vec2(2.7, 28.0));
    float streak = pavement_noise(UV / vec2(0.45, 12.0) + vec2(8.3, 2.1));
    float fine_streak = pavement_noise(UV / vec2(0.055, 7.0) + vec2(19.7, 1.4));
    float variation = mix(0.92, 1.04, broad) * mix(0.97, 1.03, streak) * mix(0.98, 1.02, fine_streak);
    vec2 authored_uv = UV / authored_tile_m;
    vec2 authored_dx = dFdx(authored_uv);
    vec2 authored_dy = dFdy(authored_uv);
    vec3 authored = vec3(0.0);
    if (authored_surface) {
        authored = textureGrad(authored_color, authored_uv, authored_dx, authored_dy).rgb;
        float broad_tone = mix(0.92, 1.04, broad) * mix(0.84, 1.12, broad);
        float fine_tone = mix(0.97, 1.03, streak) * mix(0.98, 1.02, fine_streak);
        ALBEDO = mix(vec3(0.10), authored, 0.70) * authored_color_gain * fine_tone;
        ALBEDO *= mix(0.9604, broad_tone, broad_tone_strength);
        vec3 warm_tint = vec3(1.18, 1.04, 0.83);
        warm_tint /= dot(warm_tint, vec3(0.2126, 0.7152, 0.0722));
        ALBEDO *= mix(vec3(1.0), warm_tint, authored_warmth);
    } else {
        vec3 aggregate = texture(color_map, detail).rgb;
        // Original generic material remains available for matched comparisons.
        aggregate = mix(aggregate, texture(color_map, detail * 0.71 + vec2(0.37, 0.61)).rgb, 0.28);
        ALBEDO = aggregate * vec3(0.53, 0.50, 0.46) * variation;
    }
    float across = clamp(UV.x / max(UV2.x, 0.001), 0.0, 1.0);
    // Slowly varying broad material band; appearance estimates only. The
    // periodic station domain preserves continuity at the lap boundary.
    float band_phase = UV.y / lap_length_m * TAU;
    vec2 band_domain = vec2(cos(band_phase), sin(band_phase)) * (lap_length_m / (TAU * 80.0));
    float band_drift = pavement_noise(band_domain) - 0.5;
    float breakup_strength = binder_mottling_strength * (authored_surface ? 1.0 : 0.0);
    float band_breakup = pavement_noise(band_domain * 4.0 + vec2(51.0, 17.0));
    float band_center = 0.62 + band_drift * 0.12 + (band_breakup - 0.5) * 0.12 * breakup_strength;
    float band_width = (0.22 + band_drift * 0.06) * mix(1.0, 0.45 + band_breakup, breakup_strength);
    float broad_band = 1.0 - smoothstep(band_width * 0.35, band_width, abs(across - band_center));
    float band_weight = broad_band * surface_variation_strength * (authored_surface ? 1.0 : 0.0);
    band_weight *= mix(1.0, 0.35 + band_breakup, breakup_strength);
    ALBEDO *= 1.0 - band_weight * 0.50;
    // Annular domains close at the lap seam and give independent cross-track
    // and longitudinal scales. The radial offset is in cross-track metres.
    vec2 station_direction = vec2(cos(band_phase), sin(band_phase));
    vec2 binder_domain = station_direction * (lap_length_m / (TAU * 7.0) + UV.x / 0.85);
    vec2 streak_domain = station_direction * (lap_length_m / (TAU * 3.5) + UV.x / 0.23);
    float binder_patches = filtered_pavement_noise(binder_domain + vec2(71.0, 29.0));
    float binder_streaks = filtered_pavement_noise(streak_domain + vec2(13.0, 47.0));
    float binder_pattern = mix(binder_patches, binder_streaks, 0.30);
    float mottling = (binder_pattern - 0.5) * binder_mottling_strength * (authored_surface ? 1.0 : 0.0);
    ALBEDO *= 1.0 + mottling * 0.80;
    // A bounded original reconstruction of a visible paving joint. Integrate
    // the narrow line over its projected pixel footprint to avoid distant crawl.
    float joint_region = smoothstep(paving_joint_station_m.x, paving_joint_station_m.x + 16.0, UV.y)
                       * (1.0 - smoothstep(paving_joint_station_m.y - 16.0, paving_joint_station_m.y, UV.y))
                       * paving_joint_strength * (authored_surface ? 1.0 : 0.0);
    float joint_distance = UV.x - UV2.x * paving_joint_across;
    float joint_pixel_m = max(fwidth(joint_distance), 0.001);
    float joint_half_width = paving_joint_width_m * 0.5;
    float joint_overlap = max(0.0, min(joint_distance + joint_pixel_m * 0.5, joint_half_width)
                                  - max(joint_distance - joint_pixel_m * 0.5, -joint_half_width));
    float joint_line = joint_overlap / joint_pixel_m;
    float paving_side = smoothstep(-joint_pixel_m * 0.5, joint_pixel_m * 0.5, joint_distance);
    float paving_gain = mix(1.04, 0.92, paving_side) * (1.0 + joint_line * 0.30);
    ALBEDO *= mix(1.0, paving_gain, joint_region);
    float tone_interior = smoothstep(0.10, 0.22, min(across, 1.0 - across));
    float observed_tone = texture(pavement_tone, vec2(across, UV.y / lap_length_m)).r * 2.0;
    ALBEDO *= mix(1.0, observed_tone, pavement_tone_strength * tone_interior);
    // Binder-rich edge appearance from the footage. Width and irregularity
    // are artistic estimates; the existing pavement boundary remains intact.
    float edge_distance = max(0.0, min(UV.x, UV2.x - UV.x));
    float phase = UV.y / lap_length_m * TAU;
    vec2 edge_domain = vec2(cos(phase), sin(phase)) * (lap_length_m * 0.7 / TAU);
    float edge_width = mix(0.08, 0.15, pavement_noise(edge_domain));
    float edge_aa = max(fwidth(edge_distance), 0.008);
    float edge_weight = 1.0 - smoothstep(edge_width - edge_aa, edge_width + edge_aa, edge_distance);
    // Break the uniform black ribbon into binder and exposed aggregate. Use
    // world coordinates for fine grain so the lap join cannot introduce a seam.
    float edge_grain = pavement_noise(world_position.xz * 48.0);
    float grain_visibility = 1.0 - smoothstep(0.006, 0.035, length(fwidth(world_position.xz)));
    edge_grain = mix(0.5, edge_grain, grain_visibility);
    float binder = mix(0.16, 0.38, pavement_noise(edge_domain * 2.9 + vec2(31.0)));
    float aggregate_binder = binder + (edge_grain - 0.5) * 0.24;
    ALBEDO *= mix(1.0, aggregate_binder, edge_weight);
    if (authored_surface) {
        // Estimated relief from this albedo, not a recovered height scan.
        // Mip filtered samples avoid resolving subpixel stones at a distance.
        vec2 texel = 1.0 / vec2(textureSize(authored_color, 0));
        vec3 weights = vec3(0.2126, 0.7152, 0.0722);
        float h = dot(authored, weights);
        float hx = dot(textureGrad(authored_color, authored_uv + vec2(texel.x, 0.0), authored_dx, authored_dy).rgb, weights);
        float hy = dot(textureGrad(authored_color, authored_uv + vec2(0.0, texel.y), authored_dx, authored_dy).rgb, weights);
        vec2 slope = vec2(hx - h, hy - h) * authored_relief_m / (texel * authored_tile_m);
        NORMAL_MAP = normalize(vec3(-slope, 1.0)) * 0.5 + 0.5;
        NORMAL_MAP_DEPTH = 1.0;
        ROUGHNESS = mix(clamp(0.62 + (h - 0.10) * 0.20 + (broad - 0.5) * 0.06, 0.58, 0.68), 0.72, edge_weight);
        float binder_sheen = smoothstep(0.25, 0.80, streak);
        ROUGHNESS += (0.018 * band_weight - 0.025 * binder_sheen * surface_variation_strength) * (1.0 - edge_weight);
    } else {
        ROUGHNESS = clamp(0.62 + (texture(rough_map, detail).r - 0.5) * 0.10
                          + (streak - 0.5) * 0.035 + (broad - 0.5) * 0.06, 0.52, 0.72);
        ROUGHNESS = mix(ROUGHNESS, 0.72, edge_weight);
        NORMAL_MAP = texture(normal_map, detail).rgb;
        NORMAL_MAP_DEPTH = 0.07;
    }
    // Elongated periodic domains keep wear continuous at the lap join. The
    // two scales break up parallel lanes; filtered noise limits distant crawl.
    vec2 wear_domain = station_direction * (lap_length_m / (TAU * 18.0) + UV.x / 0.32);
    vec2 wear_fine_domain = station_direction * (lap_length_m / (TAU * 9.0) + UV.x / 0.075);
    float wear_large = filtered_pavement_noise(wear_domain + vec2(123.0, 67.0));
    float wear_fine = filtered_pavement_noise(wear_fine_domain + vec2(211.0, 31.0));
    float wear = smoothstep(0.20, 0.80, mix(wear_large, wear_fine, 0.30));
    // Smooth regions occupy interrupted patches, not continuous polished lanes.
    // Most pavement keeps its original matte response, including the edges.
    vec2 wear_patch_domain = station_direction * (lap_length_m / (TAU * 4.0) + UV.x / 1.2);
    float wear_patch = smoothstep(0.55, 0.80,
        filtered_pavement_noise(wear_patch_domain + vec2(43.0, 137.0)));
    float wear_weight = directional_wear_strength * wear_patch
                      * (authored_surface ? 1.0 : 0.0) * (1.0 - edge_weight);
    ROUGHNESS = mix(ROUGHNESS, mix(0.46, 0.70, wear), wear_weight);
    ROUGHNESS = clamp(ROUGHNESS + roughness_offset * (1.0 - edge_weight), 0.25, 0.9);
}
