extends SceneTree
const Surface = preload("res://scripts/offroad_surface.gd")
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var surface := Surface.new()
	surface.configure(
		{
			"bounds": {"x0": 0.0, "z0": 0.0, "x1": 2.0, "z1": 2.0},
			"vertices": [[0, 0, 0], [2, 2, 0], [0, 0, 2]],
			"triangles": [[0, 1, 2]],
			"road_rings": [[[2, 0], [2, 2], [0, 2]]]
		}
	)
	for point in [Vector3(0.5, 100, 0.5), Vector3(1, -100, 0), Vector3(0, 0, 2)]:
		var result := surface.sample(point)
		check(not result.has("error") and not result.road_cutout, "Known triangle point missing")
		check(
			absf(result.height - point.x) < 0.00001, "Height does not match independent plane y=x"
		)
		check(
			result.normal.distance_to(Vector3(-1, 1, 0).normalized()) < 0.00001,
			"Normal does not match analytic plane"
		)
	check(
		surface.sample(Vector3(1.5, 0, 1.5)).get("road_cutout", false), "Road cutout not identified"
	)
	check(absf(surface.sample(Vector3(-1, 0, 0)).height) < 0.00001, "World edge clamp failed")
	surface.road_rings.clear()
	check(
		surface.sample(Vector3(1.5, 0, 1.5)).has("error"), "Uncovered terrain was silently accepted"
	)
	print("OFFROAD_SURFACE_CHECK failures=", failures)
	quit(failures)
