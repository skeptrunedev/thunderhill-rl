shader_type spatial;
render_mode cull_disabled;

// Original material, metre scaled detail. It is not a surveyed curb surface.
uniform vec3 paint_tint : source_color = vec3(1.0);
uniform float wear_amount : hint_range(0.0, 1.0) = 0.2;
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

void vertex() {
    world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
    vec2 p = world_position.xz;
    float stains = surface_noise(p * 3.0);
    float chips = surface_noise(p * 85.0);
    float detail_weight = 1.0 - smoothstep(0.006, 0.025, length(fwidth(p)));
    float exposed = smoothstep(0.62, 0.84, chips) * wear_amount * detail_weight;
    vec3 pigment = pow(COLOR.rgb, vec3(2.2)) * paint_tint;
    vec3 concrete = vec3(0.25, 0.24, 0.21);
    ALBEDO = mix(pigment, concrete, exposed) * mix(0.76, 1.0, stains);
    ROUGHNESS = mix(0.78, 0.96, max(exposed, stains));
}
