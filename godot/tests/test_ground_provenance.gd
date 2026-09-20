extends SceneTree
const Replay = preload("res://scripts/replay.gd")
var failures := 0


func _initialize() -> void:
	var manifest := {
		"type": "episode",
		"track_sha256": FileAccess.get_sha256("res://data/track.json"),
		"terrain_sha256": FileAccess.get_sha256("res://data/terrain.json"),
		"surface_sha256": FileAccess.get_sha256("res://data/surface.json")
	}
	for source in ["valid", "terrain", "surface"]:
		var candidate := manifest.duplicate()
		if source != "valid":
			candidate[source + "_sha256"] = "incompatible"
		var path := "user://ground-provenance-fixture.jsonl"
		var file := FileAccess.open(path, FileAccess.WRITE)
		file.store_line(JSON.stringify(candidate))
		file.close()
		var reader := Replay.new()
		var error: String = reader.open_recording(path, manifest.track_sha256)
		if error.is_empty() != (source == "valid"):
			failures += 1
			push_error("Ground provenance validation failed for " + source)
	var track := preload("res://scripts/track.gd").new()
	for source in ["valid", "track", "terrain"]:
		var metadata := {
			"track_sha256": manifest.track_sha256, "terrain_sha256": manifest.terrain_sha256
		}
		if source != "valid":
			metadata[source + "_sha256"] = "incompatible"
		var error: String = track.validate_surface_sources({"metadata": metadata})
		if error.is_empty() != (source == "valid"):
			failures += 1
			push_error("Runtime surface source validation failed for " + source)
	track.free()
	print("GROUND_PROVENANCE_CHECK failures=", failures)
	quit(failures)
