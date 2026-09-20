extends SceneTree


func _initialize() -> void:
	var script = load("res://scripts/main.gd")
	var failures := 0
	for format in [Image.FORMAT_RGB8, Image.FORMAT_RGBA8]:
		var frame := Image.create(4, 4, false, format)
		frame.fill(Color.BLACK)
		var result: Dictionary = script._save_screenshot(frame, "user://capture-test.png")
		if result.error == OK or result.validation != "all_black":
			failures += 1
		frame.set_pixel(2, 2, Color.RED)
		result = script._save_screenshot(frame, "user://capture-test.png")
		if result.error != OK or result.validation != "nonblack":
			failures += 1
		result = script._save_screenshot(frame, "user://missing-directory/capture.png")
		if result.error == OK:
			failures += 1
	for frame in [null, Image.new()]:
		var result: Dictionary = script._save_screenshot(frame, "user://capture-empty.png")
		if result.error == OK or result.validation != "empty_image":
			failures += 1
	print("CAPTURE_CHECK failures=", failures)
	quit(0 if failures == 0 else 1)
