shader_type spatial;
render_mode cull_disabled;
uniform sampler2D color_map : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D normal_map : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D rough_map : filter_linear_mipmap_anisotropic, repeat_enable;
void fragment(){
 vec2 detail=UV*vec2(1.0,1.0);
 float variation=0.95+0.05*sin(UV.y*0.073)*sin(UV.x*0.63);
 ALBEDO=texture(color_map,detail).rgb*0.65*variation;
 ROUGHNESS=clamp(texture(rough_map,detail).r*0.85+0.1,0.65,0.98);
 NORMAL_MAP=texture(normal_map,detail).rgb;
 NORMAL_MAP_DEPTH=0.38;
}
