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
	# Older recordings lack ground hashes. Keep their existing explicit state
	# playback compatibility; new recordings must match their ground geometry.
	for source in ["terrain", "surface"]:
		var key: String = source + "_sha256"
		if (
			manifest.has(key)
			and manifest[key] != FileAccess.get_sha256("res://data/" + source + ".json")
		):
			return "Replay " + source + " hash differs from loaded ground"
	if (
		manifest.has("obstacle_collision")
		and (
			manifest.obstacle_collision.get("pit_wall_sha256", "")
			!= FileAccess.get_sha256("res://data/pit-wall.json")
		)
	):
		return "Replay pit wall hash differs from loaded obstacle"
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
		"longitudinal_force_scale",
		"front_force_n",
		"rear_force_n",
		"lateral_force_n",
		"front_lateral_force_n",
		"rear_lateral_force_n",
		"front_lateral_capacity_n",
		"rear_lateral_capacity_n",
		"lateral_capacity_n",
		"requested_lateral_force_n",
		"grip_utilization",
		"target_lean",
		"crash_reason",
		"collision_contact",
		"assist_enabled",
		"auto_shift",
		"shift_remaining",
		"last_shift_command",
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
	if state.has("requested_controls"):
		sim.last_controls = state.requested_controls.duplicate(true)
