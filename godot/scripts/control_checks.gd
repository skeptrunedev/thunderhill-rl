extends RefCounted

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


func run(main: Node3D) -> int:
	var tree := main.get_tree()
	await tree.process_frame
	check(main.paused, "Game must begin at riding menu")
	for button in main.hud.panel.find_children("*", "Button", true, false):
		if button.text == "Ride":
			button.pressed.emit()
	key(KEY_W, true)
	for i in 240:
		await tree.physics_frame
	key(KEY_W, false)
	check(main.sim.speed > 2.0, "Keyboard throttle failed to move motorcycle")
	check(not main.sim.crashed, "Motorcycle fell during straight throttle input")
	check(main.environment_failure.is_empty(), "Straight launch hit an unsupported model state")
	key(KEY_D, true)
	for i in 60:
		await tree.physics_frame
	key(KEY_D, false)
	check(main.sim.last_controls.get("steer", 0.0) > 0.5, "Keyboard steering did not reach physics")
	check(absf(main.sim.steering) > 0.0001, "Keyboard steering did not move the front wheel")
	key(KEY_SPACE, true)
	for i in 12:
		await tree.physics_frame
	key(KEY_SPACE, false)
	check(main.sim.rear_brake_applied > 0.0, "Keyboard rear brake failed")
	var before: float = main.sim.speed
	key(KEY_S, true)
	for i in 240:
		await tree.physics_frame
	key(KEY_S, false)
	check(main.sim.speed < before, "Keyboard front brake failed")
	key(KEY_C, true)
	await tree.process_frame
	key(KEY_C, false)
	await tree.process_frame
	check(main.camera_mode == 1, "Camera keyboard shortcut failed")
	key(KEY_ESCAPE, true)
	await tree.process_frame
	key(KEY_ESCAPE, false)
	check(main.paused, "Escape did not pause")
	var tick: int = main.sim.tick
	for i in 20:
		await tree.physics_frame
	check(main.sim.tick == tick, "Paused human game advanced physics")
	main.menu_action("reset")
	check(main.sim.tick == 0 and main.sim.speed == 0, "Restart did not clear riding state")
	check(not main.paused, "Restart did not resume")
	print("HUMAN_CONTROLS_CHECK failures=", failures)
	return failures
