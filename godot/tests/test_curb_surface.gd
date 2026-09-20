extends SceneTree
const Surface = preload("res://scripts/curb_surface.gd")
var failures := 0
var checks := 0


func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)


func add(surface: RefCounted, points: Array) -> String:
	return surface.add_quad(
		points[0],
		points[1],
		points[2],
		points[3],
		Vector2.ZERO,
		Vector2.RIGHT,
		Vector2.ONE,
		Vector2.DOWN
	)


func _initialize() -> void:
	var surface := Surface.new()
	check(surface.sample(Vector3.ZERO).is_empty(), "Empty surface has no contact")
	# Warped, tapered geometry with grade and bank, including negative coordinates.
	for offset in [Vector3.ZERO, Vector3(-64, 3, 32)]:
		for reverse in [false, true]:
			surface.clear()
			var points := [
				Vector3(0, 0, 0), Vector3(2, .2, .25), Vector3(1.5, .8, 4), Vector3(-.5, .4, 3)
			]
			if reverse:
				points.reverse()
			for i in points.size():
				points[i] += offset
			check(add(surface, points).is_empty(), "Valid quad accepted")
			var st: SurfaceTool = surface.surface_tool()
			st.generate_normals()
			var mesh := st.commit()
			var arrays := mesh.surface_get_arrays(0)
			var rendered: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
			var uv: PackedVector2Array = arrays[Mesh.ARRAY_TEX_UV]
			check(rendered.size() == 6, "Exactly two rendered triangles")
			var original_uv := [Vector2.ZERO, Vector2.RIGHT, Vector2.ONE, Vector2.DOWN]
			for i in rendered.size():
				check(
					uv[i] == original_uv[points.find(rendered[i])],
					"UV stays attached to its vertex"
				)
			for start in [0, 3]:
				var a := rendered[start]
				var b := rendered[start + 1]
				var c := rendered[start + 2]
				var normal := -(b - a).cross(c - a).normalized()
				check(normal.y > 0, "Rendered winding faces up")
				for weights in [
					Vector3(.25, .25, .5), Vector3(.125, .75, .125), Vector3(.5, .25, .25)
				]:
					var point: Vector3 = a * weights.x + b * weights.y + c * weights.z
					var result: Dictionary = surface.sample(point)
					check(not result.is_empty(), "Rendered triangle interior has contact")
					if not result.is_empty():
						check(
							absf(result.height - point.y) < .00002,
							"Rendered height agrees with contact"
						)
						check(
							result.normal.distance_to(normal) < .00002,
							"Rendered face normal agrees with contact"
						)
						check(
							result.on_curb and result.triangle_id == start / 3,
							"Triangle identity preserved"
						)
	# Exact representable boundaries, no extrapolation even one tiny step outside.
	surface.clear()
	var square := [Vector3(0, 1, 0), Vector3(2, 1, 0), Vector3(2, 1, 2), Vector3(0, 1, 2)]
	check(add(surface, square).is_empty(), "Square accepted")
	for p in [Vector3.ZERO, Vector3(2, 0, 2), Vector3(1, 0, 1), Vector3(0, 0, 1), Vector3(2, 0, 1)]:
		check(not surface.sample(p).is_empty(), "Boundary included")
	for p in [
		Vector3(-.000001, 0, 1),
		Vector3(2.000001, 0, 1),
		Vector3(1, 0, -.000001),
		Vector3(1, 0, 2.000001),
		Vector3(INF, 0, 0)
	]:
		check(surface.sample(p).is_empty(), "Outside boundary has no contact")
	check(
		surface.sample(Vector3(1, 0, 1)).triangle_id == 0,
		"Shared diagonal deterministically owns first triangle"
	)
	for i in square.size():
		square[i].y = 3
	check(add(surface, square).is_empty(), "Overlapping higher surface accepted")
	var highest: Dictionary = surface.sample(Vector3(1, -100, 1))
	check(
		highest.height == 3 and highest.triangle_id == 2,
		"Highest overlap selected independent of query height"
	)
	check(add(surface, square).is_empty(), "Equal height overlap accepted")
	check(
		surface.sample(Vector3(1, 100, 1)).triangle_id == 2,
		"Equal height chooses first inserted triangle"
	)
	var before := var_to_bytes([surface.vertices, surface.uvs, surface.triangles, surface.grid])
	for invalid in [
		[Vector3.ZERO, Vector3.RIGHT, Vector3(1, 0, 1), Vector3.ZERO],
		[Vector3.ZERO, Vector3(2, 0, 2), Vector3(2, 0, 0), Vector3(0, 0, 2)],
		[Vector3.ZERO, Vector3(INF, 0, 0), Vector3.ONE, Vector3.BACK],
		[Vector3.ZERO, Vector3(1e10, 0, 0), Vector3(1e10, 0, 1e10), Vector3(0, 0, 1e10)]
	]:
		check(not add(surface, invalid).is_empty(), "Invalid quad rejected")
		check(
			(
				var_to_bytes([surface.vertices, surface.uvs, surface.triangles, surface.grid])
				== before
			),
			"Invalid quad leaves all state unchanged"
		)
	check(
		not (
			surface
			. add_quad(
				square[0],
				square[1],
				square[2],
				square[3],
				Vector2(NAN, 0),
				Vector2.ZERO,
				Vector2.ZERO,
				Vector2.ZERO
			)
			. is_empty()
		),
		"Nonfinite UV rejected"
	)
	check(
		var_to_bytes([surface.vertices, surface.uvs, surface.triangles, surface.grid]) == before,
		"Invalid UV atomic"
	)
	surface.clear()
	check(
		(
			surface.vertices.is_empty()
			and surface.uvs.is_empty()
			and surface.triangles.is_empty()
			and surface.grid.is_empty()
		),
		"Clear resets geometry and index"
	)
	print("CURB_SURFACE_CHECK checks=", checks, " failures=", failures)
	quit(failures)
