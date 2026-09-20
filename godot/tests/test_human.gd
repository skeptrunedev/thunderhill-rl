extends SceneTree

var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func key(code: Key, down: bool) -> void:
	var event := InputEventKey.new()
	event.keycode = code
	event.physical_keycode = code
	event.pressed = down
	Input.parse_input_event(event)


func _initialize() -> void:
	run.call_deferred()


func run() -> void:
	var main = load("res://main.tscn").instantiate()
	root.add_child(main)
	await process_frame
	check(main.paused, "Game must begin at riding menu")
	for button in main.hud.panel.find_children("*", "Button", true, false):
		if button.text == "Ride":
			button.pressed.emit()
	key(KEY_W, true)
	for i in 240:
		await physics_frame
	key(KEY_W, false)
	check(main.sim.speed > 2.0, "Keyboard throttle failed to move motorcycle")
	var before: float = main.sim.speed
	key(KEY_S, true)
	for i in 240:
		await physics_frame
	key(KEY_S, false)
	check(main.sim.speed < before, "Keyboard front brake failed")
	key(KEY_C, true)
	await process_frame
	key(KEY_C, false)
	await process_frame
	check(main.camera_mode == 1, "Camera keyboard shortcut failed")
	key(KEY_ESCAPE, true)
	await process_frame
	key(KEY_ESCAPE, false)
	check(main.paused, "Escape did not pause")
	var tick: int = main.sim.tick
	for i in 20:
		await physics_frame
	check(main.sim.tick == tick, "Paused human game advanced physics")
	main.menu_action("reset")
	check(main.sim.tick == 0 and main.sim.speed == 0, "Restart did not clear riding state")
	check(not main.paused, "Restart did not resume")
	print("HUMAN_CONTROLS_CHECK failures=", failures)
	main.queue_free()
	await process_frame
	quit(0 if failures == 0 else 1)
