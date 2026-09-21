shader_type spatial;

// Original tailoring and finish estimates guided by the onboard footage.
// UV2 stores part identity: x is molded armor, y is the tailored palm shell.
varying vec3 part_position;
varying vec3 part_color;

float grain_hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float grain_noise(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(grain_hash(cell), grain_hash(cell + vec3(1,0,0)), f.x),
                   mix(grain_hash(cell + vec3(0,1,0)), grain_hash(cell + vec3(1,1,0)), f.x), f.y),
               mix(mix(grain_hash(cell + vec3(0,0,1)), grain_hash(cell + vec3(1,0,1)), f.x),
                   mix(grain_hash(cell + vec3(0,1,1)), grain_hash(cell + vec3(1,1,1)), f.x), f.y), f.z);
}

vec3 linear_color(vec3 srgb) {
    return mix(srgb / 12.92, pow((srgb + vec3(0.055)) / 1.055, vec3(2.4)),
               step(vec3(0.04045), srgb));
}

void vertex() {
    part_position = VERTEX;
    part_color = linear_color(COLOR.rgb);
}

void fragment() {
    float armor = UV2.x;
    // Tailored reinforcement has a crisp sewn edge, not an airbrushed gradient.
    vec2 q = abs(vec2(part_position.x, part_position.z - 0.040)) - vec2(0.024, 0.033);
    float edge_distance = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - 0.006;
    float edge_pixel = max(fwidth(edge_distance), 0.00001);
    float top = smoothstep(0.018, 0.028, part_position.y) * UV2.y;
    float panel = (1.0 - smoothstep(-edge_pixel, edge_pixel, edge_distance)) * top;
    float cuff_distance = abs(part_position.z - 0.091) - 0.007;
    float cuff_pixel = max(fwidth(cuff_distance), 0.00001);
    float cuff = (1.0 - smoothstep(-cuff_pixel, cuff_pixel, cuff_distance)) * UV2.y;
    vec3 color = mix(part_color, linear_color(vec3(0.09, 0.10, 0.11)), panel);
    // Footage at 20 seconds shows black gloves with a limited warm cuff accent.
    float cuff_top = cuff * top * (1.0 - smoothstep(0.024, 0.028, abs(part_position.x)));
    color = mix(color, linear_color(vec3(0.72, 0.22, 0.11)), cuff_top);
    float trim = (1.0 - smoothstep(0.0007, 0.0007 + cuff_pixel,
                                  abs(cuff_distance + 0.0015))) * cuff_top;
    color = mix(color, linear_color(vec3(0.72, 0.72, 0.67)), trim);
    // A narrow double seam is filtered in metres at the current screen footprint.
    float seam_distance = min(abs(edge_distance + 0.0014), abs(edge_distance + 0.0030));
    float seam = (1.0 - smoothstep(0.0002, 0.0002 + edge_pixel, seam_distance)) * top;
    color *= 1.0 - 0.22 * seam;
    vec3 grain_position = part_position / 0.0008;
    float footprint = max(length(dFdx(grain_position)), length(dFdy(grain_position)));
    float resolved = 1.0 - smoothstep(0.7, 2.0, footprint);
    float grain = (grain_noise(grain_position) - 0.5) * resolved;
    ALBEDO = color * (1.0 + grain * 0.06 * (1.0 - armor));
    METALLIC = 0.0;
    SPECULAR = 0.5;
    ROUGHNESS = mix(0.56 + grain * 0.08, 0.25, armor);
    float height = grain * mix(0.000012, 0.000001, armor) - seam * 0.00003;
    vec3 dx = dFdx(VERTEX);
    vec3 dy = dFdy(VERTEX);
    vec3 rx = cross(dy, NORMAL);
    vec3 ry = cross(NORMAL, dx);
    float det = dot(dx, rx);
    vec3 gradient = dFdx(height) * rx + dFdy(height) * ry;
    NORMAL = normalize(NORMAL - sign(det) * gradient / max(abs(det), 0.000000000001));
}
