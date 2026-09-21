shader_type spatial;
render_mode cull_disabled;

// Painted finish over sourced concrete detail. Appearance is not a surveyed material.
uniform vec3 paint_tint : source_color = vec3(1.0);
uniform float wear_amount : hint_range(0.0, 1.0) = 0.2;
uniform bool curb_stripes = false;
uniform bool edge_paint = false;
// Preserve unresolved worn area rather than reverting to intact paint.
uniform bool edge_paint_mean_wear = true;
uniform float edge_paint_width_m = 0.20;
// Appearance estimates: the thin marking follows aggregate rather than being flat.
uniform float paint_aggregate_pitch_m = 0.008;
uniform float paint_aggregate_relief_m = 0.00025;
uniform bool use_mesh_uv = false;
uniform bool concrete_detail = false;
uniform sampler2D concrete_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D concrete_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D concrete_roughness : filter_linear_mipmap_anisotropic, repeat_enable;
uniform float concrete_tile_m = 1.2;
uniform float concrete_normal_depth = 0.16;
uniform vec3 stripe_blue : source_color = vec3(0.1059, 0.2549, 0.4902);
uniform vec3 stripe_white : source_color = vec3(0.898, 0.898, 0.863);
varying vec3 world_position;

float hash_cell(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

float surface_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash_cell(cell), hash_cell(cell + vec2(1.0, 0.0)), f.x),
               mix(hash_cell(cell + vec2(0.0, 1.0)), hash_cell(cell + vec2(1.0)), f.x), f.y);
}

float aggregate_hash(vec2 cell) {
    uvec2 lattice = uvec2(ivec2(cell));
    uint value = lattice.x * 1664525u + lattice.y * 1013904223u;
    value = (value ^ (value >> 16u)) * 2246822519u;
    value = (value ^ (value >> 13u)) * 3266489917u;
    value = value ^ (value >> 16u);
    return float(value & 16777215u) / 16777216.0;
}

float aggregate_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    vec2 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(aggregate_hash(cell), aggregate_hash(cell + vec2(1, 0)), w.x),
               mix(aggregate_hash(cell + vec2(0, 1)), aggregate_hash(cell + vec2(1)), w.x), w.y);
}

void vertex() {
    world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
    vec2 p = use_mesh_uv ? UV : world_position.xz;
    bool discard_paint = false;
    if (edge_paint) {
        // Use the actual marking mesh width. Only remove fragments inside it;
        // do not expand paint onto the shoulder or alter contact geometry.
        float paint_edge_m = min(UV.x, 1.0 - UV.x) * edge_paint_width_m;
        float pixel_m = max(max(fwidth(UV.x) * edge_paint_width_m, length(fwidth(world_position.xz))), 0.00001);
        float resolved = 1.0 - smoothstep(0.0008, 0.003, pixel_m);
        float edge_chip_m = mix(0.0002, 0.0028, surface_noise(world_position.xz * 95.0));
        discard_paint = paint_edge_m < edge_chip_m * resolved;
    }
    float stains = surface_noise(p * 3.0);
    float chips = surface_noise(p * 85.0);
    float detail_weight = 1.0 - smoothstep(0.006, 0.025, length(fwidth(p)));
    float exposed = smoothstep(0.62, 0.84, chips) * wear_amount * detail_weight;
    if (edge_paint && edge_paint_mean_wear) {
        // Approximate mean of thresholded cubic value noise with uniform corner
        // values. This is a filtering estimate, not measured track paint wear.
        exposed += 0.168 * wear_amount * (1.0 - detail_weight);
    }
    vec3 pigment = pow(COLOR.rgb, vec3(2.2)) * paint_tint;
    if (curb_stripes) {
        // Continuous distance avoids quantizing stripe boundaries to track vertices.
        // 2.5 m is provisional appearance, not a surveyed curb measurement.
        float wave = sin(UV.y * 3.14159265 / 2.5);
        float footprint = max(fwidth(UV.y) * 3.14159265 / 2.5, 0.001);
        float stripe = smoothstep(-footprint, footprint, wave);
        pigment = mix(stripe_white, stripe_blue, stripe);
    }
    vec3 concrete = edge_paint ? vec3(0.045, 0.040, 0.033) : vec3(0.25, 0.24, 0.21);
    if (concrete_detail) {
        // Publisher tile width is 1.2 m. Paint suppresses bare concrete contrast
        // and relief; tint and response are artistic, not measured Thunderhill BRDF.
        vec2 detail_uv = UV / concrete_tile_m;
        float grain = dot(texture(concrete_color, detail_uv).rgb, vec3(0.2126, 0.7152, 0.0722));
        float paint_variation = mix(0.90, 1.02, clamp(grain * 1.6, 0.0, 1.0));
        float broad = surface_noise(p * vec2(0.21, 0.45));
        ALBEDO = mix(pigment, concrete, exposed) * paint_variation * mix(0.97, 1.0, broad);
        ROUGHNESS = clamp(0.70 + texture(concrete_roughness, detail_uv).r * 0.25, 0.70, 0.95);
        NORMAL_MAP = texture(concrete_normal, detail_uv).rgb;
        NORMAL_MAP_DEPTH = concrete_normal_depth;
    } else {
        ALBEDO = mix(pigment, concrete, exposed) * mix(0.76, 1.0, stains);
        ROUGHNESS = mix(0.78, 0.96, max(exposed, stains));
        NORMAL_MAP = vec3(0.5, 0.5, 1.0);
        NORMAL_MAP_DEPTH = 0.0;
    }
    if (edge_paint) {
        // Fade unresolved relief rather than letting subpixel aggregate sparkle.
        // Differentiate the continuous height before applying visibility.
        vec2 aggregate_uv = world_position.xz / paint_aggregate_pitch_m;
        float visibility = 1.0 - smoothstep(0.35, 1.0, length(fwidth(aggregate_uv)));
        float grain = aggregate_noise(aggregate_uv) - 0.5;
        float height = grain * paint_aggregate_relief_m;
        vec3 dx = dFdx(VERTEX);
        vec3 dy = dFdy(VERTEX);
        vec3 rx = cross(dy, NORMAL);
        vec3 ry = cross(NORMAL, dx);
        float determinant = dot(dx, rx);
        vec3 gradient = (dFdx(height) * rx + dFdy(height) * ry) * visibility;
        vec3 bumped = normalize(NORMAL - sign(determinant) * gradient / max(abs(determinant), 0.000000000001));
        // This shared shader already uses tangent normals for concrete. Convert
        // the world-pattern gradient into that same output convention.
        NORMAL_MAP = vec3(dot(bumped, normalize(TANGENT)), dot(bumped, normalize(BINORMAL)), dot(bumped, normalize(NORMAL))) * 0.5 + 0.5;
        NORMAL_MAP_DEPTH = 1.0;
        ALBEDO *= 1.0 + grain * visibility * 0.08;
        ROUGHNESS = clamp(ROUGHNESS + grain * visibility * 0.06, 0.0, 1.0);
    }
    // Evaluate all derivatives and implicit texture gradients before discarding.
    if (discard_paint) {
        discard;
    }
}
