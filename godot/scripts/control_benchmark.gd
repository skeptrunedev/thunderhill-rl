extends RefCounted
## Recorded inputs drive live physics at normal wall time. This is not state playback.
var reader := preload("res://scripts/replay.gd").new()
var source_sha256 := ""
var start_usec := 0
var last_frame_usec := 0
var intervals_ms: Array[float] = []
var max_position_error_m := 0.0
var offtrack_ticks := 0
var expected_state: Dictionary
var resolution := Vector2.ZERO
var slow_frames: Array[Dictionary] = []
var capture_events: Array[Dictionary] = []


func open_trace(path: String, sim: RefCounted, track_hash: String, dt: float) -> String:
	var error := reader.open_recording(path, track_hash)
	if not error.is_empty():
		return error
	if reader.manifest.get("physics_version") != sim.MODEL_VERSION:
		return "Benchmark recording uses a different physics version"
	if reader.manifest.get("parameters") != sim.parameters:
		return "Benchmark recording uses different physics parameters"
	if absf(float(reader.manifest.get("physics_dt", 0.0)) - dt) > 1e-12:
		return "Benchmark recording uses a different physics timestep"
	if reader.manifest.initial_state.get("tick", -1) != 0:
		return "Benchmark recording must begin at tick zero"
	source_sha256 = FileAccess.get_sha256(path)
	return ""


func start() -> void:
	start_usec = Time.get_ticks_usec()


func next_action() -> Dictionary:
	var row: Dictionary = reader.next_state()
	if row.is_empty():
		return {}
	if (
		row.get("tick", -1) != reader.consumed_ticks
		or row.get("previous_tick", -1) != reader.consumed_ticks - 1
	):
		return {"error": "Benchmark recording has discontinuous ticks"}
	if not row.get("requested_controls") is Dictionary:
		return {"error": "Benchmark recording lacks controls"}
	if not row.get("state") is Dictionary:
		return {"error": "Benchmark recording lacks comparison state"}
	var position: Variant = row.state.get("position")
	if not position is Array or position.size() != 3:
		return {"error": "Benchmark comparison position is invalid"}
	for coordinate: Variant in position:
		if not (coordinate is float or coordinate is int) or not is_finite(float(coordinate)):
			return {"error": "Benchmark comparison position is not finite"}
	expected_state = row.state
	return {"controls": row.requested_controls}


func record_step(sim: RefCounted, on_track: bool) -> void:
	var p: Array = expected_state.position
	max_position_error_m = maxf(
		max_position_error_m, sim.position.distance_to(Vector3(p[0], p[1], p[2]))
	)
	if not on_track:
		offtrack_ticks += 1


func record_frame(viewport: Viewport) -> void:
	var now := Time.get_ticks_usec()
	if last_frame_usec > 0:
		var duration_ms := float(now - last_frame_usec) / 1000.0
		intervals_ms.append(duration_ms)
		if duration_ms >= 50.0:
			slow_frames.append(
				{
					"frame_index": intervals_ms.size(),
					"start_ms": float(last_frame_usec - start_usec) / 1000.0,
					"end_ms": float(now - start_usec) / 1000.0,
					"duration_ms": duration_ms
				}
			)
	last_frame_usec = now
	resolution = viewport.get_visible_rect().size


func record_capture(stage: String, begin_usec: int, end_usec: int) -> void:
	capture_events.append(
		{
			"stage": stage,
			"start_ms": float(begin_usec - start_usec) / 1000.0,
			"end_ms": float(end_usec - start_usec) / 1000.0,
			"duration_ms": float(end_usec - begin_usec) / 1000.0
		}
	)


func report(sim: RefCounted, laps: int, failure: String = "") -> Dictionary:
	var elapsed := float(Time.get_ticks_usec() - start_usec) / 1000000.0
	intervals_ms.sort()
	var result := {
		"ok": failure.is_empty() and not sim.crashed and offtrack_ticks == 0 and laps > 0,
		"failure": failure,
		"source_sha256": source_sha256,
		"source_policy_id": reader.manifest.get("policy_id", ""),
		"dynamics_version": sim.MODEL_VERSION,
		"ticks": sim.tick,
		"simulated_seconds": sim.elapsed,
		"wall_seconds": elapsed,
		"completed_laps": laps,
		"offtrack_ticks": offtrack_ticks,
		"max_position_error_m": max_position_error_m,
		"rendered": DisplayServer.get_name() != "headless",
		"resolution": [resolution.x, resolution.y],
		"platform": OS.get_name(),
		"frame_intervals": intervals_ms.size(),
		"slow_frame_threshold_ms": 50.0,
		"slow_frames": slow_frames,
		"capture_events": capture_events,
		"crashed": sim.crashed,
		"measurement":
		"Wall clock process frame intervals, including physics, recording and render warmup; scene setup before the first callback excluded"
	}
	if not intervals_ms.is_empty():
		result["median_ms"] = intervals_ms[int(intervals_ms.size() * 0.5)]
		result["p95_ms"] = intervals_ms[int(intervals_ms.size() * 0.95)]
		result["p99_ms"] = intervals_ms[int(intervals_ms.size() * 0.99)]
		result["max_ms"] = intervals_ms.back()
	return result
