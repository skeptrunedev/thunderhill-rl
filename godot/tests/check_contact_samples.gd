extends SceneTree
## Compare full contact records across refactors, including curb boundaries.


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var track := preload("res://scripts/track.gd").new()
	root.add_child(track)
	assert(track.initialization_error.is_empty(), track.initialization_error)
	var digest := HashingContext.new()
	digest.start(HashingContext.HASH_SHA256)
	var output: FileAccess
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--contact-output="):
			output = FileAccess.open(arg.trim_prefix("--contact-output="), FileAccess.WRITE)
			assert(output != null)
	var count := 0
	var failures := 0
	var started := Time.get_ticks_usec()
	for i in track.points.size():
		var road: Dictionary = track._road_sample(track.points[i])
		for side in [-1.0, 1.0]:
			for edge in [-0.001, 0.0, 0.001, 0.8999, 0.9, 0.9001, 1.4, 6.0, 20.0]:
				var point: Vector3 = track.points[i] + road.left * side * (road.width * 0.5 + edge)
				var result: Dictionary = track.sample_world(point)
				if not is_finite(result.height) or result.normal.y <= 0.0:
					failures += 1
					print(
						"INVALID_CONTACT ",
						JSON.stringify(
							{
								"index": i,
								"side": side,
								"edge": edge,
								"point": [point.x, point.y, point.z]
							}
						)
					)
				digest.update(var_to_bytes(result))
				if output != null:
					output.store_line(
						JSON.stringify(
							{
								"index": i,
								"side": side,
								"edge": edge,
								"record": result,
								"bytes": var_to_bytes(result).hex_encode()
							}
						)
					)
				count += 1
	print(
		"CONTACT_SAMPLES ",
		JSON.stringify(
			{
				"count": count,
				"failures": failures,
				"sample_ms": (Time.get_ticks_usec() - started) / 1000.0,
				"sha256": digest.finish().hex_encode(),
				"track_sha256": FileAccess.get_sha256("res://data/track.json"),
				"surface_sha256": FileAccess.get_sha256("res://data/surface.json")
			}
		)
	)
	if output != null:
		output.close()
	quit(1 if failures else 0)
