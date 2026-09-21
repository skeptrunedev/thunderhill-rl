extends Control
## Original live race display, composed as a high resolution material texture.
## Original graphics for the current manufacturer 1280 by 480 display specification.
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
	draw_rect(Rect2(0, 0, 1280, 480), Color("070b10"))
	_text("TRACK", Vector2(48, 42), 26, RED)
	_text("RPM x 1000", Vector2(1050, 42), 21, MUTED)
	var rpm: float = readout.get("rpm", 0)
	for tick in range(29):
		var x := 52.0 + tick * 41.7
		var y := 143.0 - minf(float(tick), 16.0) * 2.0
		var color := RED if tick >= 24 else WHITE
		var major := tick % 2 == 0
		draw_line(Vector2(x, y), Vector2(x, y - (16 if major else 8)), color, 2, true)
		if major:
			_text(str(tick / 2), Vector2(x - 7, y - 24), 20, color)
		var active := rpm >= float(tick) * 500.0
		draw_rect(Rect2(x, y + 8, 35, 15), color if active else Color("202b31"))
	draw_line(Vector2(48, 184), Vector2(1232, 184), Color("3a4851"), 2, true)
	_text("SPEED", Vector2(52, 228), 23, MUTED)
	_text(speed_text, Vector2(48, 336), 98)
	_text("km/h", Vector2(252, 332), 26, MUTED)
	_text("GEAR", Vector2(426, 228), 23, MUTED)
	_text(gear_text, Vector2(433, 362), 144)
	_text("LAP TIME", Vector2(716, 228), 23, MUTED)
	_text(lap_text, Vector2(710, 332), 88)
	for x in [376, 656]:
		draw_line(Vector2(x, 213), Vector2(x, 369), Color("3a4851"), 2, true)
	draw_line(Vector2(48, 400), Vector2(1232, 400), Color("3a4851"), 2, true)
	_text("THUNDERHILL", Vector2(52, 446), 24, MUTED)
	_text(rpm_text, Vector2(501, 446), 26, MUTED)
	_text("EAST / PRACTICE", Vector2(988, 446), 23, MUTED)
