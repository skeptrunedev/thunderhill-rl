class_name CurbSurface
extends "res://scripts/triangle_ribbon.gd"
## Curb identity is separate from the shared triangle geometry implementation.


func sample(p: Vector3) -> Dictionary:
	var result: Dictionary = super.sample(p)
	if not result.is_empty():
		result.on_curb = true
	return result
