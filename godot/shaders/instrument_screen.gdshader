shader_type spatial;
render_mode cull_disabled;

uniform sampler2D display_texture : source_color, filter_linear;
uniform bool footprint_filter = true;
uniform vec2 face_size = vec2(0.177, 0.097);
varying vec2 face_position;

void vertex() {
    face_position = VERTEX.xy;
}

// Average the projected pixel footprint of the live viewport. It has no baked
// mip chain, so ordinary bilinear sampling aliases thin ticks when minified.
vec3 display_sample(vec2 uv) {
    if (!footprint_filter) {
        return texture(display_texture, uv).rgb;
    }
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    vec3 color = vec3(0.0);
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 offset = (vec2(float(x), float(y)) + vec2(0.5)) / 4.0 - vec2(0.5);
            color += texture(display_texture, clamp(uv + dx * offset.x + dy * offset.y, vec2(0.0), vec2(1.0))).rgb;
        }
    }
    return color / 16.0;
}

void fragment() {
    vec2 uv = face_position / face_size + vec2(0.5);
    uv.y = 1.0 - uv.y;
    // The opaque front surface combines a live emissive LCD image with a
    // dielectric reflection. It does not draw a fixed reflection into the art.
    ALBEDO = vec3(0.004, 0.006, 0.009);
    EMISSION = display_sample(uv) * 0.8;
    METALLIC = 0.0;
    SPECULAR = 0.5;
    ROUGHNESS = 0.16;
    CLEARCOAT = 0.6;
    CLEARCOAT_ROUGHNESS = 0.10;
}
