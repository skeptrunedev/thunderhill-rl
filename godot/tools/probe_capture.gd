extends SceneTree
## Diagnostic only: compare root and offscreen readback without game assets.
var output := "user://capture-probe"
var offscreen: SubViewport
var rows: Array = []
var failures := 0


func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--probe-output="):
			output = argument.trim_prefix("--probe-output=")
	_run.call_deferred()


func _run() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("Capture probe requires a renderer")
		quit(2)
		return
	root.size = Vector2i(640, 360)
	root.msaa_3d = Viewport.MSAA_2X
	var scene := Node3D.new()
	root.add_child(scene)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.1, 0.6, 0.2)
	scene.add_child(environment)
	var cube := MeshInstance3D.new()
	cube.mesh = BoxMesh.new()
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.albedo_color = Color(0.9, 0.1, 0.05)
	cube.material_override = material
	scene.add_child(cube)
	var camera := Camera3D.new()
	camera.position = Vector3(0, 0, 3)
	scene.add_child(camera)
	camera.current = true
	offscreen = SubViewport.new()
	offscreen.size = Vector2i(640, 360)
	offscreen.world_3d = root.world_3d
	offscreen.msaa_3d = Viewport.MSAA_2X
	offscreen.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(offscreen)
	var second_camera := Camera3D.new()
	second_camera.position = camera.position
	offscreen.add_child(second_camera)
	second_camera.current = true
	for frame in range(1, 61):
		await RenderingServer.frame_post_draw
		if frame not in [1, 13, 60]:
			continue
		for item in [
			{"name": "root", "viewport": root}, {"name": "offscreen", "viewport": offscreen}
		]:
			var viewport: Viewport = item.viewport
			var read_begin := Time.get_ticks_usec()
			var captured := viewport.get_texture().get_image()
			var readback_usec := Time.get_ticks_usec() - read_begin
			var before := inspect_image(captured)
			var thread := Thread.new()
			var error := thread.start(inspect_image.bind(captured))
			assert(error == OK, "Cannot start diagnostic worker")
			var after: Dictionary = thread.wait_to_finish()
			var passed: bool = not before.empty and not before.all_black and before == after
			if not passed:
				failures += 1
			var path := "%s-%s-%d.png" % [output, item.name, frame]
			var save_error := captured.save_png(path) if captured != null else ERR_INVALID_DATA
			if save_error != OK:
				failures += 1
			var row := {
				"viewport": item.name,
				"readback_usec": readback_usec,
				"sample_frame": frame,
				"process_frames": Engine.get_process_frames(),
				"frames_drawn": Engine.get_frames_drawn(),
				"window_can_draw": root.can_draw(),
				"window_focused": root.has_focus(),
				"draw_calls":
				viewport.get_render_info(
					Viewport.RENDER_INFO_TYPE_VISIBLE, Viewport.RENDER_INFO_DRAW_CALLS_IN_FRAME
				),
				"main_thread": before,
				"worker_thread": after,
				"passed": passed,
				"save_error": save_error
			}
			rows.append(row)
			print("CAPTURE_PROBE ", JSON.stringify(row))
	var file := FileAccess.open(output + ".json", FileAccess.WRITE)
	if file == null:
		quit(2)
		return
	file.store_string(
		JSON.stringify(
			{
				"samples": rows,
				"failures": failures,
				"platform": OS.get_name(),
				"engine": Engine.get_version_info()
			},
			"  "
		)
	)
	file.close()
	quit(0 if failures == 0 else 1)


static func inspect_image(captured: Image) -> Dictionary:
	if captured == null or captured.is_empty():
		return {"empty": true, "all_black": true}
	var rgb := captured.duplicate() as Image
	rgb.convert(Image.FORMAT_RGB8)
	var bytes := rgb.get_data()
	var digest := HashingContext.new()
	digest.start(HashingContext.HASH_SHA256)
	digest.update(bytes)
	return {
		"empty": false,
		"all_black": bytes.count(0) == bytes.size(),
		"rgb_sha256": digest.finish().hex_encode(),
		"format": captured.get_format(),
		"width": captured.get_width(),
		"height": captured.get_height()
	}
