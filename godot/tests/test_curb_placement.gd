extends SceneTree
const Placement = preload("res://scripts/curb_placement.gd")
var failures := 0


func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var placement := Placement.new()
	var row := {"id": "example", "side": 1, "segments": [0, 1]}
	var data := {
		"schema_version": 1,
		"track_sha256": "test",
		"sample_count": 3,
		"width_m": 0.9,
		"inner_height_m": 0.05,
		"outer_height_m": 0.11,
		"runs": [row]
	}
	check(placement.configure(data, "test", 3).is_empty(), "Valid placement rejected")
	check(placement.has_segment(1, 1) and not placement.has_segment(1, -1), "Wrong curb side")
	check(
		not placement.configure(data, "different", 3).is_empty() and placement.active.is_empty(),
		"Stale track placement retained"
	)
	row.segments = [1, 1]
	check(not placement.configure(data, "test", 3).is_empty(), "Duplicate segment accepted")
	row.segments = [0.5]
	check(not placement.configure(data, "test", 3).is_empty(), "Fractional segment accepted")
	row.segments = [3]
	check(not placement.configure(data, "test", 3).is_empty(), "Out of range segment accepted")
	row.segments = [0]
	data.width_m = NAN
	check(not placement.configure(data, "test", 3).is_empty(), "Nonfinite width accepted")
	var track: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/track.json")
	)
	var actual: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://data/curb-placement.json")
	)
	check(
		(
			placement
			. configure(
				actual, FileAccess.get_sha256("res://data/track.json"), track.samples.size()
			)
			. is_empty()
		),
		"Runtime placement invalid"
	)
	var retained := 0
	var removed := 0
	for index in track.samples.size():
		for side: int in [-1, 1]:
			var curvature: float = track.samples[index].curvature
			var baseline: bool = absf(curvature) > 0.012 and side * curvature > 0.0
			# Reviewed T2 inner arc contains only the 23 rejected runs C08..C30.
			var rejected: bool = side == 1 and index >= 324 and index <= 415
			var expected: bool = baseline and not rejected
			check(
				placement.has_segment(index, side) == expected,
				"Unexpected placement change at segment %d side %d" % [index, side]
			)
			retained += int(expected)
			removed += int(baseline and rejected)
	check(retained == 261 and removed == 36, "Curb change count differs from review")
	print("CURB_PLACEMENT_CHECK failures=", failures, " retained=", retained, " removed=", removed)
	quit(failures)
