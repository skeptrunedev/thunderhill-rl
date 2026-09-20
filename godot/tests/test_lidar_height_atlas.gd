extends SceneTree
const Atlas = preload("res://scripts/lidar_height_atlas.gd")
const Surface = preload("res://scripts/lidar_height_surface.gd")
var failures := 0


func check(ok: bool, message: String) -> void:
	if not ok:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var atlas := Atlas.new()
	check(atlas.evaluate(0., 0.).is_empty(), "Unconfigured atlas rejected")
	var path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--fixture="):
			path = arg.trim_prefix("--fixture=")
	if path.is_empty() or not FileAccess.file_exists(path):
		push_error("Supply --fixture=/absolute/path to atlas parity fixture")
		quit(1)
		return
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if (
		not parsed is Dictionary
		or not parsed.get("atlas") is Dictionary
		or not parsed.get("samples") is Array
	):
		push_error("Malformed atlas parity fixture")
		quit(1)
		return
	var config: Dictionary = parsed.atlas
	check(atlas.configure(config), "Fixture atlas configuration accepted")
	check(not parsed.samples.is_empty(), "Fixture contains samples")
	if failures > 0:
		quit(1)
		return
	var max_error := 0.0
	for sample in parsed.samples:
		var result: Dictionary = atlas.evaluate(sample.x, sample.z)
		check(not result.is_empty(), "Fixture sample within support")
		if result.is_empty():
			continue
		var errors := [absf(result.height - float(sample.height))]
		for axis in 2:
			errors.append(absf(float(result.gradient[axis]) - float(sample.gradient[axis])))
		for axis in 3:
			errors.append(absf(float(result.hessian[axis]) - float(sample.hessian[axis])))
		for error in errors:
			max_error = maxf(max_error, error)
			check(error < 0.00001, "Python atlas parity error=" + str(error))
	var x := float(parsed.samples[0].x)
	var z := float(parsed.samples[0].z)
	var baseline: Dictionary = atlas.evaluate(x, z)
	for pair in [
		["schema_version", 2],
		["support_radius_m", 0],
		["support_radius_m", -1],
		["support_radius_m", NAN],
		["support_radius_m", INF],
		["patches", []],
		["patches", [null]]
	]:
		var invalid: Dictionary = config.duplicate(true)
		invalid[pair[0]] = pair[1]
		check(not atlas.configure(invalid), "Invalid atlas configuration rejected: " + pair[0])
	var invalid: Dictionary = config.duplicate(true)
	invalid.patches.append(invalid.patches[0].duplicate(true))
	check(not atlas.configure(invalid), "Duplicate patch center rejected")
	invalid = config.duplicate(true)
	invalid.support_radius_m = (
		absf(float(invalid.patches[0].knots[0][3]))
		+ absf(float(invalid.patches[0].knots[0][-4]))
		+ 1.0
	)
	check(not atlas.configure(invalid), "Support beyond spline domain rejected")
	invalid = config.duplicate(true)
	invalid.patches[0].coefficients[0][0] = NAN
	check(not atlas.configure(invalid), "Malformed nested surface rejected")
	invalid = config.duplicate(true)
	invalid.patches[0].origin[0] = 1e100
	check(not atlas.configure(invalid), "Unrepresentable spatial index rejected")
	check(atlas.evaluate(x, z) == baseline, "Invalid configurations preserve valid state")
	var owned: Dictionary = config.duplicate(true)
	check(atlas.configure(owned), "Owned snapshot accepted")
	for patch in owned.patches:
		for row in patch.coefficients:
			for i in row.size():
				row[i] += 10.0
	check(atlas.evaluate(x, z) == baseline, "Caller mutations do not alter atlas")
	for point in [[NAN, z], [x, INF], [1e100, 1e100]]:
		check(
			atlas.evaluate(point[0], point[1]).is_empty(),
			"Nonfinite or unsupported coordinates rejected"
		)
	# One isolated patch exposes exact open support boundaries in both axes.
	var isolated: Dictionary = config.duplicate(true)
	isolated.patches = [isolated.patches[0]]
	check(atlas.configure(isolated), "Isolated patch accepted")
	var isolated_surface := Surface.new()
	check(isolated_surface.configure(isolated.patches[0]), "Isolated reference accepted")
	var radius := float(isolated.support_radius_m)
	var cx := float(isolated.patches[0].origin[0])
	var cz := float(isolated.patches[0].origin[1])
	for sign_value in [-1.0, 1.0]:
		check(
			atlas.evaluate(cx + sign_value * radius, cz).is_empty(), "X support boundary excluded"
		)
		check(
			atlas.evaluate(cx, cz + sign_value * radius).is_empty(), "Z support boundary excluded"
		)
		check(
			not atlas.evaluate(cx + sign_value * radius * 0.999999, cz).is_empty(),
			"Tiny positive weight retains support"
		)
		var near_edge: float = cx + sign_value * radius * 0.999999
		check(
			atlas.evaluate(near_edge, cz) == isolated_surface.evaluate(near_edge, cz),
			"Single patch quotient preserves derivatives at tiny weights"
		)
	print("LIDAR_ATLAS_PARITY samples=", parsed.samples.size(), " max_error=", max_error)
	print("LIDAR_ATLAS_TEST failures=", failures)
	quit(0 if failures == 0 else 1)
