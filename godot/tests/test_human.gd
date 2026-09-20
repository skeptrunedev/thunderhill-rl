extends SceneTree


func _initialize() -> void:
	run.call_deferred()


func run() -> void:
	var main = load("res://main.tscn").instantiate()
	root.add_child(main)
	var checks = preload("res://scripts/control_checks.gd").new()
	var failures: int = await checks.run(main)
	main.queue_free()
	await process_frame
	quit(0 if failures == 0 else 1)
