class_name TrackHorizon
extends Node3D


## Actual USGS terrain surrounding the detailed driving area.
func build(track: Node3D) -> void:
	var data: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/horizon.json")
	)
	var detail: Dictionary = track.terrain
	var xmin: float = detail.x0
	var xmax: float = xmin + (detail.nx - 1) * detail.step
	var zmin: float = detail.z0
	var zmax: float = zmin + (detail.nz - 1) * detail.step
	var xs: Array = []
	var zs: Array = []
	for x in int(data.nx):
		xs.append(float(data.x0) + x * float(data.step))
	for z in int(data.nz):
		zs.append(float(data.z0) + z * float(data.step))
	# Include every fine boundary coordinate to avoid cracks at the terrain seam.
	for x in int(detail.nx):
		xs.append(xmin + x * float(detail.step))
	for z in int(detail.nz):
		zs.append(zmin + z * float(detail.step))
	xs.sort()
	zs.sort()
	var vertices := PackedVector3Array()
	for z in zs:
		for x in xs:
			var p := Vector3(x, 0, z)
			if x >= xmin and x <= xmax and z >= zmin and z <= zmax:
				p.y = track.terrain_height(p)
			else:
				p.y = _height(data, x, z)
			vertices.append(p)
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var nx: int = xs.size()
	for z in zs.size() - 1:
		for x in xs.size() - 1:
			if xs[x + 1] - xs[x] < 0.001 or zs[z + 1] - zs[z] < 0.001:
				continue
			if xs[x] >= xmin and xs[x + 1] <= xmax and zs[z] >= zmin and zs[z + 1] <= zmax:
				continue
			var i := z * nx + x
			track._quad(
				st,
				vertices[i],
				vertices[i + 1],
				vertices[i + nx + 1],
				vertices[i + nx],
				Vector2.ZERO,
				Vector2.RIGHT,
				Vector2.ONE,
				Vector2.DOWN
			)
	st.generate_normals()
	st.generate_tangents()
	var mesh := MeshInstance3D.new()
	mesh.name = "SurroundingMeasuredHills"
	mesh.mesh = st.commit()
	mesh.material_override = track.terrain_material
	mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mesh)


func _height(data: Dictionary, x: float, z: float) -> float:
	var gx := clampf((x - float(data.x0)) / float(data.step), 0, float(data.nx) - 1)
	var gz := clampf((z - float(data.z0)) / float(data.step), 0, float(data.nz) - 1)
	var ix := mini(floori(gx), int(data.nx) - 2)
	var iz := mini(floori(gz), int(data.nz) - 2)
	var nx: int = data.nx
	return lerpf(
		lerpf(data.heights[iz * nx + ix], data.heights[iz * nx + ix + 1], gx - ix),
		lerpf(data.heights[(iz + 1) * nx + ix], data.heights[(iz + 1) * nx + ix + 1], gx - ix),
		gz - iz
	)
