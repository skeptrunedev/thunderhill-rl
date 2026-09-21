shader_type spatial;
render_mode cull_disabled;

// Original object-space finish. These are visual estimates, not scanned parts.
uniform vec3 base_color : source_color = vec3(0.082, 0.090, 0.098);
uniform float metalness = 0.0;
uniform float finish_roughness = 0.52;
uniform float grain_pitch_m = 0.0007;
uniform float relief_m = 0.000015;
varying vec3 finish_position;

float cell_hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float grain_noise(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(cell_hash(cell), cell_hash(cell + vec3(1,0,0)), f.x),
                   mix(cell_hash(cell + vec3(0,1,0)), cell_hash(cell + vec3(1,1,0)), f.x), f.y),
               mix(mix(cell_hash(cell + vec3(0,0,1)), cell_hash(cell + vec3(1,0,1)), f.x),
                   mix(cell_hash(cell + vec3(0,1,1)), cell_hash(cell + vec3(1,1,1)), f.x), f.y), f.z);
}

void vertex() {
    finish_position = VERTEX;
}

void fragment() {
    vec3 grain_position = finish_position / grain_pitch_m;
    float footprint = max(length(dFdx(grain_position)), length(dFdy(grain_position)));
    // Fade unresolved microtexture rather than allowing sparkling noise in motion.
    float resolved = 1.0 - smoothstep(0.7, 2.0, footprint);
    float grain = (grain_noise(grain_position) - 0.5) * resolved;
    ALBEDO = base_color * (1.0 + grain * 0.08);
    METALLIC = metalness;
    ROUGHNESS = clamp(finish_roughness + grain * 0.10, 0.1, 0.9);
    SPECULAR = 0.5;
    // Surface-gradient bump works on meshes without UV/tangent attributes.
    float height = grain * relief_m;
    vec3 dx = dFdx(VERTEX);
    vec3 dy = dFdy(VERTEX);
    vec3 rx = cross(dy, NORMAL);
    vec3 ry = cross(NORMAL, dx);
    float det = dot(dx, rx);
    vec3 gradient = dFdx(height) * rx + dFdy(height) * ry;
    NORMAL = normalize(NORMAL - sign(det) * gradient / max(abs(det), 0.000000000001));
}
