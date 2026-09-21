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
	var leather := _leather_material()
	leather.albedo_color = Color("24282a")
	var glove_material := glove_finish()
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


static func glove_finish() -> ShaderMaterial:
	var material := ShaderMaterial.new()
	material.shader = preload("res://shaders/glove_finish.gdshader")
	return material


static func _leather_material() -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	# Generic CC0 leather grain at the publisher's 0.6 metre tile width.
	# Object space keeps the grain attached while limbs articulate.
	material.uv1_triplanar = true
	material.uv1_scale = Vector3.ONE / 0.6
	material.uv1_triplanar_sharpness = 4.0
	material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	material.normal_enabled = true
	material.normal_texture = preload("res://assets/materials/leather_red_02_nor_gl_1k.jpg")
	material.normal_scale = 0.45
	material.roughness_texture = preload("res://assets/materials/leather_red_02_rough_1k.jpg")
	material.roughness = 0.85
	return material


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
