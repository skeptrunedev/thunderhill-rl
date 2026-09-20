extends SceneTree
const Surface = preload("res://scripts/lidar_height_surface.gd")
var failures := 0


func check(ok: bool, message: String) -> void:
	if not ok:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var surface := Surface.new()
	check(surface.evaluate(0., 0.).is_empty(), "Unconfigured surface rejected")
	var fixture_path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--fixture="):
			fixture_path = arg.trim_prefix("--fixture=")
	if fixture_path.is_empty() or not FileAccess.file_exists(fixture_path):
		push_error("Supply --fixture=/absolute/path to parity fixture")
		quit(1)
		return
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(fixture_path))
	if (
		not parsed is Dictionary
		or not parsed.get("surface") is Dictionary
		or not parsed.get("samples") is Array
	):
		push_error("Malformed parity fixture")
		quit(1)
		return
	var fixture: Dictionary = parsed
	var config: Dictionary = fixture.surface
	check(surface.configure(config), "Fixture configuration accepted")
	if failures > 0:
		quit(1)
		return
	check(not fixture.samples.is_empty(), "Fixture contains samples")
	var max_error := 0.0
	for sample in fixture.samples:
		var result: Dictionary = surface.evaluate(sample.x, sample.z)
		check(not result.is_empty(), "Fixture sample inside bounds")
		if result.is_empty():
			continue
		var errors := [absf(result.height - float(sample.height))]
		for axis in 2:
			errors.append(absf(float(result.gradient[axis]) - float(sample.gradient[axis])))
		for axis in 3:
			errors.append(absf(float(result.hessian[axis]) - float(sample.hessian[axis])))
		for error in errors:
			max_error = maxf(max_error, error)
			check(error < 0.00001, "Python parity error=" + str(error))
	var x0 := float(config.origin[0]) + float(config.knots[0][3])
	var x1 := float(config.origin[0]) + float(config.knots[0][-4])
	var z0 := float(config.origin[1]) + float(config.knots[1][3])
	var z1 := float(config.origin[1]) + float(config.knots[1][-4])
	for point in [
		[x0 - 0.001, z0], [x1 + 0.001, z0], [x0, z0 - 0.001], [x0, z1 + 0.001], [NAN, z0], [x0, INF]
	]:
		check(
			surface.evaluate(point[0], point[1]).is_empty(), "Outside or nonfinite point rejected"
		)
	var baseline: Dictionary = surface.evaluate(x0, z0)
	check(not baseline.is_empty(), "Lower endpoint included")
	check(not surface.evaluate(x1, z1).is_empty(), "Upper endpoint included")
	for pair in [
		["schema_version", 2],
		["degree", [2, 3]],
		["degree", null],
		["origin", [0, NAN]],
		["origin", [0]],
		["knots", []],
		["coefficients", null],
		["coefficients", []]
	]:
		var invalid: Dictionary = config.duplicate(true)
		invalid[pair[0]] = pair[1]
		check(not surface.configure(invalid), "Malformed configuration rejected: " + pair[0])
	var invalid: Dictionary = config.duplicate(true)
	invalid.coefficients[0][0] = INF
	check(not surface.configure(invalid), "Nonfinite coefficient rejected")
	invalid = config.duplicate(true)
	invalid.coefficients[0].pop_back()
	check(not surface.configure(invalid), "Ragged coefficients rejected")
	invalid = config.duplicate(true)
	invalid.knots[0][0] = float(invalid.knots[0][0]) - 1.0
	check(not surface.configure(invalid), "Unclamped knots rejected")
	invalid = config.duplicate(true)
	invalid.knots[0][0] = NAN
	check(not surface.configure(invalid), "Nonfinite knot rejected")
	invalid = config.duplicate(true)
	invalid.knots[0][4] = invalid.knots[0][3]
	check(not surface.configure(invalid), "Excess endpoint multiplicity rejected")
	if config.knots[0].size() > 9:
		invalid = config.duplicate(true)
		invalid.knots[0][5] = invalid.knots[0][4]
		check(not surface.configure(invalid), "Repeated interior knot rejected")
	check(surface.evaluate(x0, z0) == baseline, "Rejected configurations preserve valid state")
	# Configuration owns a snapshot, not references into the caller's arrays.
	var owned: Dictionary = config.duplicate(true)
	check(surface.configure(owned), "Snapshot configuration accepted")
	owned.coefficients[0][0] += 10.0
	check(surface.evaluate(x0, z0) == baseline, "Caller mutation does not alter configured surface")
	var overflow: Dictionary = config.duplicate(true)
	overflow.coefficients[0][0] = 1e300
	check(surface.configure(overflow), "Finite extreme coefficient accepted")
	check(surface.evaluate(x0, z0).is_empty(), "Derivative vector overflow rejected")
	print("LIDAR_HEIGHT_PARITY samples=", fixture.samples.size(), " max_error=", max_error)
	print("LIDAR_HEIGHT_TEST failures=", failures)
	quit(0 if failures == 0 else 1)
