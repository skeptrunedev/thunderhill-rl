extends Node3D
## Articulated rider meshes using the shared rendering and collision pose model.
const Pose = preload("res://scripts/rider_pose.gd")
var front_origin := Vector3.ZERO
var joints: Dictionary = {}


func build(origin: Vector3) -> String:
	if not joints.is_empty():
		return "Rider arms already built"
	for side in [-1.0, 1.0]:
		var bounds := Pose.domain_bounds(origin, side, -Pose.STEERING_LIMIT, Pose.STEERING_LIMIT)
		if bounds.has("error"):
			return bounds.error
	front_origin = origin
	var leather := StandardMaterial3D.new()
	leather.albedo_color = Color("24282a")
	leather.roughness = 0.72
	var glove_material := StandardMaterial3D.new()
	glove_material.vertex_color_use_as_albedo = true
	glove_material.vertex_color_is_srgb = true
	glove_material.roughness = 0.65
	for side in [-1.0, 1.0]:
		var row := {}
		for key in ["upper", "lower", "glove"]:
			var instance := MeshInstance3D.new()
			instance.name = ("Left" if side < 0 else "Right") + key.capitalize()
			if key == "glove":
				instance.mesh = preload("res://scripts/rider_glove.gd").build(side)
				instance.material_override = glove_material
			else:
				var capsule := CapsuleMesh.new()
				capsule.radius = 0.064 if key == "upper" else 0.048
				capsule.height = Pose.BONE_LENGTH + capsule.radius * 1.1
				capsule.radial_segments = 16
				capsule.rings = 6
				instance.mesh = capsule
				instance.material_override = leather
			add_child(instance)
			row[key] = instance
		joints[side] = row
	return set_steering(0.0)


func set_steering(angle: float) -> String:
	if joints.size() != 2:
		return "Rider arms have not been built"
	var poses := {}
	for side in [-1.0, 1.0]:
		var pose := Pose.solve(front_origin, side, angle)
		if pose.has("error"):
			return pose.error
		poses[side] = pose
	for side in poses:
		for key in joints[side]:
			joints[side][key].transform = poses[side][key]
	return ""
