extends SceneTree

const Track = preload("res://scripts/track.gd")
var failures := 0


func check_height(track: Node3D, x: float, z: float, expected: float, label: String) -> void:
	var actual: float = track.terrain_height(Vector3(x, 999.0, z))
	if absf(actual - expected) > 0.00001:
		failures += 1
		push_error("%s: expected %.6f, got %.6f" % [label, expected, actual])


func _initialize() -> void:
	var track := Track.new()
	# Saddle corners, with a translated origin and nonunit grid spacing.
	# Actual triangles have planes y = u-v and y = v-u. Bilinear
	# interpolation would instead give u+v-2uv and fail both interior tests.
	track.terrain = {
		"nx": 2, "nz": 2, "x0": 10.0, "z0": -6.0, "step": 4.0, "heights": [0.0, 1.0, 1.0, 0.0]
	}
	check_height(track, 13.0, -5.0, 0.5, "Triangle a,b,c interior")
	check_height(track, 11.0, -3.0, 0.5, "Triangle a,c,d interior")
	check_height(track, 12.0, -4.0, 0.0, "Shared diagonal")
	check_height(track, 10.0, -6.0, 0.0, "Corner a")
	check_height(track, 14.0, -6.0, 1.0, "Corner b")
	check_height(track, 14.0, -2.0, 0.0, "Corner c")
	check_height(track, 10.0, -2.0, 1.0, "Corner d")
	check_height(track, 12.0, -6.0, 0.5, "Top edge")
	check_height(track, 14.0, -4.0, 0.5, "Right edge")
	check_height(track, 12.0, -2.0, 0.5, "Bottom edge")
	check_height(track, 10.0, -4.0, 0.5, "Left edge")
	check_height(track, -100.0, -4.0, 0.5, "Clamp left")
	check_height(track, 100.0, -4.0, 0.5, "Clamp right")
	check_height(track, 12.0, -100.0, 0.5, "Clamp top")
	check_height(track, 12.0, 100.0, 0.5, "Clamp bottom")
	check_height(track, 100.0, 100.0, 0.0, "Clamp corner")
	# Two adjacent cells must agree on their shared edge, including exact
	# maximum bounds where the final cell uses local coordinates equal to one.
	track.terrain = {
		"nx": 3,
		"nz": 2,
		"x0": 0.0,
		"z0": 0.0,
		"step": 1.0,
		"heights": [0.0, 1.0, 3.0, 1.0, 0.0, 2.0]
	}
	check_height(track, 1.0, 0.25, 0.75, "Internal cell edge")
	check_height(track, 1.0 - 0.000001, 0.25, 0.75, "Internal edge from left")
	check_height(track, 1.0 + 0.000001, 0.25, 0.75, "Internal edge from right")
	check_height(track, 2.0, 0.25, 2.75, "Last cell maximum x")
	track.free()
	print("TERRAIN_HEIGHT_CHECK " + JSON.stringify({"failures": failures}))
	quit(0 if failures == 0 else 1)
