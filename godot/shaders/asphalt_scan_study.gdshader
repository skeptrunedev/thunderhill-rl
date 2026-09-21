shader_type spatial;
render_mode cull_disabled;
// Independent PBR source study; no synthetic bands or inferred albedo relief.
uniform sampler2D scan_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D scan_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D scan_roughness : filter_linear_mipmap_anisotropic, repeat_enable;
uniform float tile_m = 0.5;
uniform float color_gain = 0.45;
void fragment() {
    vec2 uv = UV / tile_m;
    ALBEDO = texture(scan_color, uv).rgb * color_gain;
    NORMAL_MAP = texture(scan_normal, uv).rgb;
    NORMAL_MAP_DEPTH = 1.0;
    ROUGHNESS = texture(scan_roughness, uv).r;
    SPECULAR = 0.5;
}
