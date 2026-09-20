class_name RaceHUD
extends Control

var game: Node
var title: Label
var status: Label
var speed_label: Label
var gear_label: Label
var timing: Label
var help: Label
var panel: PanelContainer
var minimap: Control


func label(text: String, size: int, color := Color.WHITE) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", color)
	return l


func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	title = label("THUNDERHILL  /  EAST", 25, Color("eee9da"))
	title.position = Vector2(38, 28)
	add_child(title)
	status = label("STREETFIGHTER V4 S   •   PRACTICE", 13, Color("c3b48d"))
	status.position = Vector2(40, 62)
	add_child(status)
	speed_label = label("0", 76)
	speed_label.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	speed_label.position = Vector2(40, -143)
	add_child(speed_label)
	var unit := label("KM/H", 13, Color("c3b48d"))
	unit.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	unit.position = Vector2(45, -57)
	add_child(unit)
	gear_label = label("1", 43, Color("d4b97c"))
	gear_label.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	gear_label.position = Vector2(187, -108)
	add_child(gear_label)
	timing = label("00:00.000", 24)
	timing.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	timing.position = Vector2(-250, 31)
	add_child(timing)
	help = label(
		"W / S   THROTTLE / BRAKE     A / D   STEER     C   CAMERA     R   RESET     ESC   PAUSE",
		13,
		Color("d0cec7")
	)
	help.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_RIGHT)
	help.position = Vector2(-780, -38)
	add_child(help)
	panel = PanelContainer.new()
	panel.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	panel.position = Vector2(-250, -230)
	panel.custom_minimum_size = Vector2(500, 460)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.035, 0.045, 0.05, 0.94)
	style.border_color = Color("8d7951")
	style.set_border_width_all(1)
	style.content_margin_left = 32
	style.content_margin_right = 32
	style.content_margin_top = 30
	style.content_margin_bottom = 30
	panel.add_theme_stylebox_override("panel", style)
	add_child(panel)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 16)
	panel.add_child(box)
	box.add_child(label("THUNDERHILL", 37))
	box.add_child(label("EAST CIRCUIT  /  CALIFORNIA", 14, Color("c3b48d")))
	box.add_child(
		label("Ducati Streetfighter V4 S\nOpen practice • East Circuit", 16, Color("c7cccd"))
	)
	for item in [
		["Ride", "ride"],
		["Restart on main straight", "reset"],
		["Start at the Cyclone", "cyclone"],
		["Change camera", "camera"],
		["Quit", "quit"]
	]:
		var button := Button.new()
		button.text = item[0]
		button.custom_minimum_size.y = 39
		var action: String = item[1]
		button.pressed.connect(func(): game.menu_action(action))
		box.add_child(button)
	(
		box
		. add_child(
			label(
				"Gamepad: triggers drive and brake, left stick steers.\nRider balance assistance and automatic gears enabled.",
				12,
				Color("a0aaa9")
			)
		)
	)


func _process(_dt: float) -> void:
	if game == null or game.sim == null:
		return
	speed_label.text = "%d" % roundi(game.sim.speed * 3.6)
	gear_label.text = "%d" % game.sim.gear
	var millis := int(game.lap_time * 1000)
	timing.text = "%02d:%02d.%03d" % [millis / 60000, (millis / 1000) % 60, millis % 1000]
	status.text = (
		"STREETFIGHTER V4 S   •   %s"
		% (
			"CRASHED  /  R TO RESET"
			if game.sim.crashed
			else ("AGENT CONTROL" if game.agent_mode else ("PAUSED" if game.paused else "PRACTICE"))
		)
	)
	if not game.environment_failure.is_empty():
		status.text = "RIDING STATE NOT SUPPORTED YET / R TO RESTART"
		status.tooltip_text = str(game.environment_failure.error)
	else:
		status.tooltip_text = ""
	panel.visible = game.paused and not game.agent_mode
	queue_redraw()


func _draw() -> void:
	if game == null or game.track == null:
		return
	var rect := Rect2(size.x - 218, 108, 178, 210)
	draw_style_box(_map_style(), rect.grow(12))
	var pts: PackedVector3Array = game.track.points
	var low := Vector2(INF, INF)
	var high := Vector2(-INF, -INF)
	for p in pts:
		low = low.min(Vector2(p.x, p.z))
		high = high.max(Vector2(p.x, p.z))
	var scale: float = minf(rect.size.x / (high.x - low.x), rect.size.y / (high.y - low.y))
	var line := PackedVector2Array()
	for i in range(0, pts.size(), 4):
		line.append(rect.position + (Vector2(pts[i].x, pts[i].z) - low) * scale)
	if line.size() > 1:
		line.append(line[0])
		draw_polyline(line, Color(0.82, 0.80, 0.71, 0.75), 2, true)
	var p: Vector3 = game.sim.position
	draw_circle(rect.position + (Vector2(p.x, p.z) - low) * scale, 4, Color("ef634d"))
	var rpm_ratio: float = clampf(game.sim.rpm / 13500.0, 0, 1)
	draw_rect(Rect2(42, size.y - 157, 220, 3), Color(0.8, 0.8, 0.8, 0.18))
	draw_rect(
		Rect2(42, size.y - 157, 220 * rpm_ratio, 3),
		Color("d4b97c") if rpm_ratio < 0.88 else Color("ed5848")
	)


func _map_style() -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = Color(0.025, 0.04, 0.04, 0.6)
	s.set_corner_radius_all(5)
	return s
