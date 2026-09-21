extends SceneTree

const Replay = preload("res://scripts/replay.gd")
const Motorcycle = preload("res://scripts/motorcycle.gd")
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 1:
		push_error("Expected one recording path argument")
		quit(1)
		return
	var path: String = args[0]
	var expected_rows: Array[Dictionary] = []
	var source := FileAccess.open(path, FileAccess.READ)
	if source == null:
		push_error("Cannot independently read recording")
		quit(1)
		return
	while source.get_position() < source.get_length():
		var row: Variant = JSON.parse_string(source.get_line())
		if row is Dictionary and row.get("type", "") == "transition":
			expected_rows.append(row)
	source.close()
	check(expected_rows.size() == 12, "Expected twelve recorded transitions")
	var replay := Replay.new()
	var track_hash := FileAccess.get_sha256("res://data/track.json")
	check(not replay.open_recording(path, "wrong-track-hash").is_empty(), "Bad track hash accepted")
	var error: String = replay.open_recording(path, track_hash)
	if not error.is_empty():
		push_error(error)
		quit(1)
		return
	var sim := Motorcycle.new()
	sim.reset(Vector3.ZERO, 0.0)
	var count := 0
	while not replay.finished:
		var row: Dictionary = replay.next_state()
		if row.is_empty():
			break
		if count >= expected_rows.size():
			check(false, "Playback returned extra transitions")
			break
		check(row == expected_rows[count], "Playback row differs from independent recording read")
		if count == 0:
			check(replay.pending_decisions.size() == 1, "Missing decision before first transition")
			if not replay.pending_decisions.is_empty():
				check(
					replay.pending_decisions[0].tick == row.previous_tick, "Decision timing differs"
				)
		else:
			check(replay.pending_decisions.is_empty(), "Decision repeated between calls")
		replay.apply_state(sim, row.state)
		var observed: Dictionary = sim.telemetry()
		var expected: Dictionary = expected_rows[count].state
		var p: Array = expected.position
		check(observed.position == Vector3(p[0], p[1], p[2]), "Playback position differs")
		check(
			is_equal_approx(observed.heading, float(expected.heading)), "Playback heading differs"
		)
		check(is_equal_approx(observed.speed, float(expected.speed)), "Playback speed differs")
		check(
			is_equal_approx(
				observed.longitudinal_velocity,
				float(expected.get("longitudinal_velocity", expected.speed))
			),
			"Playback lost signed velocity"
		)
		for field in [
			"longitudinal_force_scale",
			"front_lateral_force_n",
			"rear_lateral_force_n",
			"front_lateral_capacity_n",
			"rear_lateral_capacity_n",
			"lateral_capacity_n",
			"requested_lateral_force_n",
			"front_force_n",
			"rear_force_n",
			"lateral_force_n",
			"grip_utilization"
		]:
			if expected.has(field):
				check(
					is_equal_approx(float(observed[field]), float(expected[field])),
					"Playback force differs: " + field
				)
		check(observed.tick == expected.tick, "Playback tick differs")
		check(is_equal_approx(observed.elapsed, float(expected.elapsed)), "Playback time differs")
		count += 1
	check(count == 12 and replay.consumed_ticks == 12, "Playback transition count differs")
	check(replay.finished, "Playback did not finish at end of file")
	check(replay.next_state().is_empty(), "Finished playback returned another state")
	print(
		"REPLAY_CHECK ",
		JSON.stringify({"ok": failures == 0, "transitions": count, "failures": failures})
	)
	quit(failures)
