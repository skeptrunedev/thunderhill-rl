class_name RaceReplay
extends RefCounted
## State playback deliberately does not resimulate potentially divergent physics.
var file: FileAccess
var manifest: Dictionary
var finished := false
var consumed_ticks := 0


func open_recording(path: String, expected_track_hash: String) -> String:
	file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		return "Cannot open replay: " + path
	var first: Variant = JSON.parse_string(file.get_line())
	if not first is Dictionary or first.get("type", "") != "episode":
		return "Replay requires episode manifest"
	manifest = first
	if manifest.get("track_sha256", "") != expected_track_hash:
		return "Replay track hash differs from loaded track"
	finished = false
	consumed_ticks = 0
	return ""


func next_state() -> Dictionary:
	while file and file.get_position() < file.get_length():
		var row: Variant = JSON.parse_string(file.get_line())
		if row is Dictionary and row.get("type", "") == "transition":
			consumed_ticks += 1
			return row
	finished = true
	return {}


func apply_state(sim: RefCounted, state: Dictionary) -> void:
	var p: Array = state.position
	sim.position = Vector3(p[0], p[1], p[2])
	# Older models only supported forward motion.
	sim.longitudinal_velocity = float(state.get("longitudinal_velocity", state.speed))
	for key in [
		"heading",
		"speed",
		"lean",
		"lean_rate",
		"steering",
		"gear",
		"rpm",
		"crashed",
		"elapsed",
		"tick",
		"on_track",
		"front_load_n",
		"rear_load_n",
		"longitudinal_acceleration",
		"lateral_acceleration",
		"throttle_applied",
		"front_brake_applied",
		"rear_brake_applied",
		"road_bank_rad",
		"lean_relative_road_rad",
		"gravity_forward_m_s2",
		"gravity_right_m_s2",
		"gravity_normal_m_s2"
	]:
		if state.has(key):
			sim.set(key, state[key])
