shader_type spatial;
render_mode cull_disabled;
uniform sampler2D grass_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D soil_color : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
varying vec3 world_pos;
void vertex(){world_pos=(MODEL_MATRIX*vec4(VERTEX,1.0)).xyz;}
void fragment(){
 vec2 uv=world_pos.xz/2.0;
 float patch=sin(world_pos.x*0.039+sin(world_pos.z*0.051)*2.0)*sin(world_pos.z*0.033);
 vec2 secondary_uv=mat2(vec2(0.8,0.6),vec2(-0.6,0.8))*uv*0.93+vec2(0.37,0.71);
 vec3 grass=mix(texture(grass_color,uv).rgb,texture(grass_color,secondary_uv).rgb,0.35);
 vec3 soil=texture(soil_color,uv*1.53).rgb;
 ALBEDO=mix(grass,soil,smoothstep(-0.3,0.65,patch)*0.6)*vec3(0.88,0.80,0.66);
 ROUGHNESS=0.96;
 NORMAL_MAP=texture(grass_normal,uv).rgb;
 NORMAL_MAP_DEPTH=0.45;
}
