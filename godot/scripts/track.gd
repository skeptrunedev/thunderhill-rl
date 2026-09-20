class_name ThunderhillTrack
extends Node3D

var data: Dictionary
var samples: Array
var terrain: Dictionary
var points: PackedVector3Array
var length_m: float
var grid: Dictionary = {}
const CELL: float = 25.0

func _ready() -> void:
 data = JSON.parse_string(FileAccess.get_file_as_string("res://data/track.json"))
 samples = data.samples
 length_m = data.length_m
 terrain = JSON.parse_string(FileAccess.get_file_as_string("res://data/terrain.json"))
 for i in samples.size():
  var p: Array = samples[i].p
  points.append(Vector3(p[0], p[1], p[2]))
  var cell := Vector2i(floori(p[0]/CELL), floori(p[2]/CELL))
  if not grid.has(cell): grid[cell] = []
  grid[cell].append(i)
 _build_terrain()
 _build_road()

func material(color: Color, roughness: float = 0.8) -> StandardMaterial3D:
 var m := StandardMaterial3D.new()
 m.albedo_color = color
 m.roughness = roughness
 m.cull_mode = BaseMaterial3D.CULL_DISABLED
 return m

func _mesh(st: SurfaceTool, mat: Material, label: String) -> void:
 st.generate_normals()
 st.generate_tangents()
 var instance := MeshInstance3D.new()
 instance.name = label
 instance.mesh = st.commit()
 instance.material_override = mat
 add_child(instance)

func _vertex(st: SurfaceTool, p: Vector3, uv: Vector2, color: Color = Color.WHITE) -> void:
 st.set_color(color)
 st.set_uv(uv)
 st.add_vertex(p)

func _quad(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, d: Vector3, u0: Vector2, u1: Vector2, u2: Vector2, u3: Vector2, color: Color = Color.WHITE) -> void:
 for v in [[a,u0],[b,u1],[c,u2],[a,u0],[c,u2],[d,u3]]:
  _vertex(st,v[0],v[1],color)

func _build_terrain() -> void:
 var st := SurfaceTool.new()
 st.begin(Mesh.PRIMITIVE_TRIANGLES)
 var nx: int = terrain.nx
 var nz: int = terrain.nz
 var spacing: float = terrain.step
 var verts := PackedVector3Array()
 for z in nz:
  for x in nx:
   var p := Vector3(terrain.x0+x*spacing,terrain.heights[z*nx+x],terrain.z0+z*spacing)
   var road := sample_world(p)
   var edge_distance: float = absf(road.distance)-road.width*0.5
   if edge_distance < 10.0:
    p.y = lerpf(road.height-0.08,p.y,smoothstep(0.0,10.0,edge_distance))
   verts.append(p)
 for z in nz-1:
  for x in nx-1:
   var a := z*nx+x
   _quad(st,verts[a],verts[a+1],verts[a+nx+1],verts[a+nx],Vector2(x,z),Vector2(x+1,z),Vector2(x+1,z+1),Vector2(x,z+1))
 var mat := ShaderMaterial.new()
 mat.shader = load("res://shaders/terrain.gdshader")
 mat.set_shader_parameter("grass_color",load("res://assets/materials/withered_grass_diff_1k.jpg"))
 mat.set_shader_parameter("soil_color",load("res://assets/materials/brown_mud_dry_diff_1k.jpg"))
 mat.set_shader_parameter("grass_normal",load("res://assets/materials/withered_grass_nor_gl_1k.jpg"))
 _mesh(st,mat,"MeasuredTerrain")

func edge_point(i: int, offset: float, lift: float = 0.04) -> Vector3:
 var next: int = (i+1)%points.size()
 var prev: int = posmod(i-1,points.size())
 var tangent := (points[next]-points[prev]).normalized()
 var left := Vector3(tangent.z,0,-tangent.x).normalized()
 return points[i]+left*offset+Vector3.UP*(tan(float(samples[i].bank))*offset+lift)

func _build_road() -> void:
 var road := SurfaceTool.new()
 var paint := SurfaceTool.new()
 var curb := SurfaceTool.new()
 road.begin(Mesh.PRIMITIVE_TRIANGLES)
 paint.begin(Mesh.PRIMITIVE_TRIANGLES)
 curb.begin(Mesh.PRIMITIVE_TRIANGLES)
 for i in points.size():
  var j: int = (i+1)%points.size()
  var w: float = samples[i].width*0.5
  var wj: float = samples[j].width*0.5
  var s: float = samples[i].s
  var sj: float = samples[j].s if j>0 else length_m
  _quad(road,edge_point(i,-w),edge_point(i,w),edge_point(j,wj),edge_point(j,-wj),Vector2(0,s),Vector2(w*2,s),Vector2(wj*2,sj),Vector2(0,sj))
  for side in [-1.0,1.0]:
   _quad(paint,edge_point(i,side*(w-0.18),0.047),edge_point(i,side*(w-0.06),0.047),edge_point(j,side*(wj-0.06),0.047),edge_point(j,side*(wj-0.18),0.047),Vector2.ZERO,Vector2.RIGHT,Vector2.ONE,Vector2.DOWN)
   # Provisional curb placement follows curvature. Real profile and placement remain editable.
   if absf(float(samples[i].curvature))>0.012 and side*float(samples[i].curvature)>0.0:
    var col := Color("1760a2") if int(s/2.5)%2==0 else Color("e5e5dc")
    _quad(curb,edge_point(i,side*w,0.05),edge_point(i,side*(w+0.9),0.11),edge_point(j,side*(wj+0.9),0.11),edge_point(j,side*wj,0.05),Vector2.ZERO,Vector2.RIGHT,Vector2.ONE,Vector2.DOWN,col)
 var asphalt := ShaderMaterial.new()
 asphalt.shader = load("res://shaders/asphalt.gdshader")
 for pair in [["color_map","Color"],["normal_map","NormalGL"],["rough_map","Roughness"]]:
  asphalt.set_shader_parameter(pair[0],load("res://assets/materials/Asphalt010_1K-JPG_%s.jpg"%pair[1]))
 _mesh(road,asphalt,"RacingSurface")
 _mesh(paint,material(Color("e8e3ce")),"EdgePaint")
 var curb_mat := material(Color.WHITE)
 curb_mat.vertex_color_use_as_albedo = true
 _mesh(curb,curb_mat,"ProvisionalCurbs")

func sample_world(p: Vector3) -> Dictionary:
 var cell := Vector2i(floori(p.x/CELL),floori(p.z/CELL))
 var candidates: Array = []
 for z in range(-1,2):
  for x in range(-1,2):
   candidates.append_array(grid.get(cell+Vector2i(x,z),[]))
 if candidates.is_empty():
  for i in points.size(): candidates.append(i)
 var best := INF
 var index := 0
 var fraction := 0.0
 for i in candidates:
  var a: Vector3 = points[i]
  var b: Vector3 = points[(i+1)%points.size()]
  var ab := Vector2(b.x-a.x,b.z-a.z)
  var t := clampf(Vector2(p.x-a.x,p.z-a.z).dot(ab)/maxf(ab.length_squared(),0.001),0,1)
  var q := a.lerp(b,t)
  var d := Vector2(p.x-q.x,p.z-q.z).length_squared()
  if d<best:
   best=d
   index=i
   fraction=t
 var j: int = (index+1)%points.size()
 var center := points[index].lerp(points[j],fraction)
 var tangent := (points[j]-points[index]).normalized()
 var left := Vector3(tangent.z,0,-tangent.x).normalized()
 var lateral := (p-center).dot(left)
 var bank := lerpf(samples[index].bank,samples[j].bank,fraction)
 var width := lerpf(samples[index].width,samples[j].width,fraction)
 var height := center.y+tan(bank)*lateral
 var normal := (Vector3.UP-left*tan(bank)-Vector3(tangent.x,0,tangent.z)*(tangent.y/maxf(Vector2(tangent.x,tangent.z).length(),0.001))).normalized()
 return {"height":height,"normal":normal,"on_track":absf(lateral)<=width*0.5,"distance":lateral,"progress":fposmod(float(samples[index].s)+fraction*(points[j]-points[index]).length(),length_m)/length_m,"tangent":tangent,"width":width,"index":index,"center":center}
