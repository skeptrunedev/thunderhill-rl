extends SceneTree
const Surface = preload("res://scripts/road_surface.gd")
const Contact = preload("res://scripts/surface_contact.gd")
var failures := 0


func check(ok: bool, message: String) -> void:
	if not ok:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var surface := Surface.new()
	var simple := {
		"schema_version": 1,
		"period_m": 10.0,
		"breaks": [0., 10.],
		"road_lift_m": .04,
		"coefficients":
		[[[0, 0, 0, 0, 1, 0], [0, 0, 0, .1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, .02, 0, 0]]]
	}
	check(surface.configure(simple).is_empty(), "Analytic surface config")
	var r: Dictionary = surface.evaluate(2., 3.)
	for item in [
		["R", Vector3(2., .68, -3.)],
		["Rs", Vector3(1., .64, 0.)],
		["Ru", Vector3(0., .08, -1.)],
		["Rss", Vector3(0., .32, 0.)],
		["Rsu", Vector3(0., .08, 0.)],
		["Ruu", Vector3.ZERO]
	]:
		check(r[item[0]].distance_to(item[1]) < 0.000001, "Analytic derivative " + item[0])
	check(surface.evaluate(NAN, 0.).has("error"), "Reject nonfinite station")
	var invalid: Dictionary = simple.duplicate(true)
	invalid.coefficients[0][0][0] = INF
	check(not surface.configure(invalid).is_empty(), "Reject nonfinite coefficients")
	check(
		surface.evaluate(2., 3.).R.distance_to(r.R) < 1e-6,
		"Rejected config preserves valid surface"
	)
	for pair in [["coefficients", 3], ["breaks", null], ["period_m", "bad"], ["road_lift_m", NAN]]:
		var malformed: Dictionary = simple.duplicate(true)
		malformed[pair[0]] = pair[1]
		check(not surface.configure(malformed).is_empty(), "Reject malformed field " + pair[0])
	var overflow: Dictionary = simple.duplicate(true)
	overflow.coefficients[0][1][0] = 1e300
	check(
		surface.configure(overflow).is_empty(),
		"Finite extreme coefficients accepted for evaluation"
	)
	check(surface.evaluate(5., 3.).has("error"), "Reject evaluation overflow")
	var args := OS.get_cmdline_user_args()
	check(args.size() == 2, "Supply road surface and parity JSON paths")
	if args.size() == 2:
		if not FileAccess.file_exists(args[0]) or not FileAccess.file_exists(args[1]):
			push_error("Fixture files missing; supply absolute paths")
			quit(1)
			return
		var actual: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
		var fixtures: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(args[1]))
		check(surface.configure(actual).is_empty(), "Historical surface config")
		var max_error := 0.0
		for case in fixtures.cases:
			var sample: Dictionary = surface.evaluate(case.station, case.lateral)
			check(not sample.has("error"), "Historical sample valid")
			var cv: Array = case.contact_velocity
			var support: Dictionary = Contact.support(
				250.,
				Vector3(cv[0], cv[1], cv[2]),
				Vector3(0., -9.81, 0.),
				sample.Rs,
				sample.Ru,
				sample.Rss,
				sample.Rsu,
				sample.Ruu
			)
			check(not support.has("error"), "Composed contact valid")
			if not support.has("error"):
				check(
					absf(support.normal_acceleration_m_s2 - case.normal_acceleration) < 0.0001,
					"Composed contact normal acceleration"
				)
			for key in case.expected:
				var a: Array = case.expected[key]
				var expected := Vector3(a[0], a[1], a[2])
				var error: float = sample[key].distance_to(expected)
				max_error = maxf(max_error, error)
				check(error < 0.0001, "Python parity " + key + " error=" + str(error))
		print("ROAD_SURFACE_PARITY cases=", fixtures.cases.size(), " max_vector_error=", max_error)
	print("ROAD_SURFACE_TEST failures=", failures)
	quit(0 if failures == 0 else 1)
