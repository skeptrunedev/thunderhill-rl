extends Control
## Display recorded tool arguments, never infer intention from bike movement.
var game: Node
const INK := Color("e8efed")
const MUTED := Color("9eacae")
const GREEN := Color("7bd6ad")
const ORANGE := Color("f3b478")


func _ready() -> void:
	position = Vector2(38, 100)
	size = Vector2(450, 448)
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func _process(_dt: float) -> void:
	visible = game != null and not game.decision_feed.is_empty()
	if visible:
		queue_redraw()


func _text(at: Vector2, text: String, font_size: int = 17, color: Color = INK) -> void:
	draw_string(ThemeDB.fallback_font, at, text, HORIZONTAL_ALIGNMENT_LEFT, 414, font_size, color)


static func steering_text(value: float) -> String:
	if is_zero_approx(value):
		return "CENTER"
	return "%s %.1f%%" % ["LEFT" if value < 0 else "RIGHT", absf(value) * 100.0]


static func clock_text(seconds: float) -> String:
	return "%02d:%04.1f" % [int(seconds) / 60, fmod(seconds, 60.0)]


func _bar(y: float, name: String, value: float, color: Color) -> void:
	_text(Vector2(18, y), name)
	draw_style_box(_style(Color("26333b"), 3), Rect2(152, y - 12, 192, 10))
	if value > 0:
		draw_style_box(_style(color, 3), Rect2(152, y - 12, 192 * clampf(value, 0, 1), 10))
	_text(Vector2(366, y), "%d%%" % roundi(value * 100), 17, color)


func _style(color: Color, radius: int) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = color
	style.set_corner_radius_all(radius)
	return style


func _draw() -> void:
	if game == null or game.decision_feed.is_empty():
		return
	var decision: Dictionary = game.decision_feed[0]
	var controls: Dictionary = decision.get("controls", {})
	var identity: Dictionary = game.policy_display
	if game.replay != null:
		identity = game.replay.manifest.get("policy_display", {})
	var seconds := float(decision.get("elapsed", float(decision.tick) * game.DT))
	var age := maxf(0, float(game.sim.elapsed) - seconds)
	draw_style_box(_style(Color(0.018, 0.027, 0.036, 0.93), 9), Rect2(Vector2.ZERO, size))
	# Pulse follows simulator time, so offline movie capture stays deterministic.
	draw_rect(Rect2(0, 12, 3, 58), Color(GREEN, 0.35 + 0.65 * maxf(0, 1 - age / 0.08)))
	_text(Vector2(18, 32), str(identity.get("model_name", "MODEL NOT LABELED")), 24)
	var generation := (
		"Generation %02d" % int(identity.generation)
		if identity.has("generation")
		else "Generation not labeled"
	)
	_text(Vector2(18, 59), generation, 17, MUTED)
	draw_line(Vector2(18, 78), Vector2(432, 78), Color("344147"))
	_text(Vector2(18, 107), "CURRENT ACTION", 14, MUTED)
	_text(Vector2(356, 107), clock_text(seconds), 17)
	var steer := float(controls.get("steer", 0))
	_text(Vector2(18, 145), "Steering")
	_text(Vector2(152, 145), steering_text(steer), 19, GREEN)
	draw_line(Vector2(152, 171), Vector2(408, 171), Color("52626a"), 3)
	draw_line(Vector2(280, 163), Vector2(280, 179), MUTED, 1)
	draw_circle(Vector2(280 + 128 * clampf(steer, -1, 1), 171), 5, GREEN)
	_text(Vector2(152, 196), "LEFT", 11, MUTED)
	_text(Vector2(376, 196), "RIGHT", 11, MUTED)
	_bar(227, "Throttle", float(controls.get("throttle", 0)), GREEN)
	_bar(261, "Front brake", float(controls.get("front_brake", 0)), ORANGE)
	_bar(295, "Rear brake", float(controls.get("rear_brake", 0)), ORANGE)
	_text(Vector2(18, 328), "Applies for up to 0.10 seconds", 14, MUTED)
	draw_line(Vector2(18, 345), Vector2(432, 345), Color("344147"))
	_text(Vector2(18, 373), "RECENT CALLS", 14, MUTED)
	for index in range(1, mini(3, game.decision_feed.size())):
		var prior: Dictionary = game.decision_feed[index]
		var c: Dictionary = prior.get("controls", {})
		var stamp := float(prior.get("elapsed", float(prior.tick) * game.DT))
		var description := (
			"%s, throttle %d%%"
			% [
				steering_text(float(c.get("steer", 0))).capitalize(),
				roundi(float(c.get("throttle", 0)) * 100)
			]
		)
		if float(c.get("front_brake", 0)) > 0 or float(c.get("rear_brake", 0)) > 0:
			description = (
				"%s, brakes %d%% / %d%%"
				% [
					steering_text(float(c.get("steer", 0))).capitalize(),
					roundi(float(c.get("front_brake", 0)) * 100),
					roundi(float(c.get("rear_brake", 0)) * 100)
				]
			)
		_text(Vector2(18, 374 + index * 27), clock_text(stamp), 14, MUTED)
		_text(Vector2(89, 374 + index * 27), description, 14, MUTED)
