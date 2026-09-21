extends Control
## Original live race display, composed as a high resolution material texture.
## The supplied cockpit footage guides density and hierarchy, not firmware assets.
var readout: Dictionary = {}
var speed_text := "000"
var gear_text := "N"
var rpm_text := "0 RPM"
var lap_text := "00:00.00"
const WHITE := Color("e0e5dd")
const MUTED := Color("8c9b9e")
const RED := Color("e14338")


func set_readings(speed_mps: float, rpm: float, gear: int, lap_seconds: float) -> bool:
	var next := {
		"speed_kmh": roundi(absf(speed_mps) * 3.6),
		"rpm": roundi(rpm / 100.0) * 100,
		"gear": gear,
		"lap_centiseconds": maxi(0, floori(lap_seconds * 100.0))
	}
	if next == readout:
		return false
	readout = next
	speed_text = "%03d" % next.speed_kmh
	gear_text = str(gear) if gear > 0 else "N"
	rpm_text = "%d RPM" % next.rpm
	var cs: int = next.lap_centiseconds
	lap_text = "%02d:%02d.%02d" % [cs / 6000, (cs / 100) % 60, cs % 100]
	queue_redraw()
	return true


func _text(text: String, at: Vector2, pixels: int, color := WHITE) -> void:
	draw_string(ThemeDB.fallback_font, at, text, HORIZONTAL_ALIGNMENT_LEFT, -1, pixels, color)


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1024, 560), Color("070b10"))
	_text("TRACK", Vector2(48, 49), 27, RED)
	_text("RPM x 1000", Vector2(752, 48), 21, MUTED)
	var rpm: float = readout.get("rpm", 0)
	# A rising scale with small half-thousand ticks remains legible in perspective.
	for tick in range(29):
		var x := 52.0 + tick * 32.4
		var y := 158.0 - minf(float(tick), 16.0) * 2.6
		var color := RED if tick >= 24 else WHITE
		var major := tick % 2 == 0
		draw_line(Vector2(x, y), Vector2(x, y - (16 if major else 8)), color, 2, true)
		if major:
			_text(str(tick / 2), Vector2(x - 7, y - 24), 20, color)
		var active := rpm >= float(tick) * 500.0
		draw_rect(Rect2(x, y + 8, 27, 17), color if active else Color("202b31"))
	draw_line(Vector2(48, 204), Vector2(974, 204), Color("3a4851"), 2, true)
	_text("LAP TIME", Vector2(54, 249), 23, MUTED)
	_text(lap_text, Vector2(50, 342), 78)
	_text("GEAR", Vector2(783, 252), 23, MUTED)
	_text(gear_text, Vector2(776, 372), 132)
	draw_line(Vector2(727, 231), Vector2(727, 390), Color("3a4851"), 2, true)
	_text(speed_text, Vector2(56, 456), 68)
	_text("km/h", Vector2(213, 454), 25, MUTED)
	_text(rpm_text, Vector2(421, 449), 28, MUTED)
	draw_line(Vector2(48, 490), Vector2(974, 490), Color("3a4851"), 2, true)
	_text("THUNDERHILL", Vector2(52, 527), 22, MUTED)
	_text("EAST  /  PRACTICE", Vector2(701, 527), 21, MUTED)
