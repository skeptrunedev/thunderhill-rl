class_name CurbPlacement
extends RefCounted
## Explicit segment placement. Source status is recorded in the data manifest.
var active: Dictionary = {}
var width_m := 0.0
var inner_height_m := 0.0
var outer_height_m := 0.0


func configure(data: Variant, track_sha256: String, sample_count: int) -> String:
	active.clear()
	width_m = 0.0
	inner_height_m = 0.0
	outer_height_m = 0.0
	if not data is Dictionary or data.get("schema_version") != 1:
		return "Invalid curb placement document"
	if data.get("track_sha256") != track_sha256 or data.get("sample_count") != sample_count:
		return "Curb placement track source mismatch"
	for key in ["width_m", "inner_height_m", "outer_height_m"]:
		var value: Variant = data.get(key)
		if (
			not (value is float or value is int)
			or not is_finite(float(value))
			or float(value) < 0.0
		):
			return "Invalid curb profile dimension"
	if float(data.width_m) <= 0.0 or float(data.outer_height_m) < float(data.inner_height_m):
		return "Invalid curb width or rising profile"
	if not data.get("runs") is Array:
		return "Curb placement requires runs"
	var pending: Dictionary = {}
	var ids: Dictionary = {}
	for run: Variant in data.runs:
		if (
			not run is Dictionary
			or not run.get("id") is String
			or run.id.is_empty()
			or ids.has(run.id)
		):
			return "Invalid or duplicate curb run identifier"
		ids[run.id] = true
		var side: Variant = run.get("side")
		if not (side is float or side is int) or (float(side) != -1.0 and float(side) != 1.0):
			return "Curb side must be minus or plus one"
		if not run.get("segments") is Array or run.segments.is_empty():
			return "Curb run requires segment indices"
		for index: Variant in run.segments:
			if (
				not (index is float or index is int)
				or not is_finite(float(index))
				or float(index) != floorf(float(index))
				or index < 0
				or index >= sample_count
			):
				return "Invalid curb segment index"
			var key := Vector2i(int(index), int(side))
			if pending.has(key):
				return "Overlapping curb placement runs"
			pending[key] = true
	active = pending
	width_m = float(data.width_m)
	inner_height_m = float(data.inner_height_m)
	outer_height_m = float(data.outer_height_m)
	return ""


func has_segment(index: int, side: int) -> bool:
	return active.has(Vector2i(index, side))
