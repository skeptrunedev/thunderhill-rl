shader_type spatial;
render_mode cull_disabled;

uniform sampler2D display_texture : source_color, filter_linear;
uniform vec2 face_size = vec2(0.177, 0.097);
varying vec2 face_position;

void vertex() {
    face_position = VERTEX.xy;
}

void fragment() {
    vec2 uv = face_position / face_size + vec2(0.5);
    uv.y = 1.0 - uv.y;
    // The opaque front surface combines a live emissive LCD image with a
    // dielectric reflection. It does not draw a fixed reflection into the art.
    ALBEDO = vec3(0.004, 0.006, 0.009);
    EMISSION = texture(display_texture, uv).rgb * 0.8;
    METALLIC = 0.0;
    SPECULAR = 0.5;
    ROUGHNESS = 0.16;
    CLEARCOAT = 0.6;
    CLEARCOAT_ROUGHNESS = 0.10;
}
