class_name AgentCamera
extends Node
## Observation camera, independent of presentation cameras and UI canvases.

const VERSION := "rider-rgb-v1"
const WIDTH := 640
const HEIGHT := 360
const RIDER_LAYER := 1 << 19
const RIDER_LIMB_LAYER := 1 << 18
const EYE_HEIGHT_M := 1.23
const EYE_FORWARD_M := 0.05
const LOOK_DOWN_RAD := 0.08
var viewport: SubViewport
var camera: Camera3D


func configure(world: World3D) -> void:
	viewport = SubViewport.new()
	viewport.name = "PolicyObservationViewport"
	viewport.size = Vector2i(WIDTH, HEIGHT)
	viewport.world_3d = world
	viewport.render_target_update_mode = SubViewport.UPDATE_DISABLED
	viewport.msaa_3d = Viewport.MSAA_2X
	add_child(viewport)
	camera = Camera3D.new()
	camera.name = "PolicyRiderCamera"
	camera.fov = 74.0
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	camera.near = 0.06
	camera.far = 3000.0
	camera.cull_mask = ((1 << 20) - 1) & ~(RIDER_LAYER | RIDER_LIMB_LAYER)
	viewport.add_child(camera)
	camera.current = true


static func assign_rider_layer(node: Node, layer: int = RIDER_LAYER) -> void:
	if node is VisualInstance3D:
		node.layers = layer
	for child in node.get_children():
		assign_rider_layer(child, layer)


func capture(sim: RefCounted, normal: Vector3, episode_id: String, folder: String) -> Dictionary:
	if (
		DisplayServer.get_name() == "headless"
		or (
			RenderingServer.get_rendering_device() == null
			and RenderingServer.get_video_adapter_name().is_empty()
		)
	):
		return {
			"error":
			"Camera capture requires a rendered instance; headless dummy rendering is unsupported",
			"failure_type": "infrastructure"
		}
	var tangent: Vector3 = sim.surface_forward(normal)
	var upright := Vector3.UP.slide(tangent).normalized()
	var right := tangent.cross(upright).normalized()
	var rider_up: Vector3 = upright * cos(sim.lean) + right * sin(sim.lean)
	camera.position = sim.position + rider_up * EYE_HEIGHT_M + tangent * EYE_FORWARD_M
	var look_direction := tangent * cos(LOOK_DOWN_RAD) - rider_up * sin(LOOK_DOWN_RAD)
	camera.look_at(camera.position + look_direction, rider_up)
	var capture_tick: int = sim.tick
	var pose := camera.global_transform
	viewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	await RenderingServer.frame_post_draw
	if sim.tick != capture_tick:
		return {
			"error": "Simulation changed during camera capture", "failure_type": "infrastructure"
		}
	var image := _read_image()
	var validation := preload("res://scripts/image_validation.gd").classify(
		image, Vector2i(WIDTH, HEIGHT)
	)
	if validation != "nonblack":
		return {
			"error": "Invalid observation readback: " + validation,
			"failure_type": "infrastructure",
			"validation": validation
		}
	var png := image.save_png_to_buffer()
	if png.is_empty():
		return {"error": "Cannot encode observation image", "failure_type": "infrastructure"}
	var hash_context := HashingContext.new()
	hash_context.start(HashingContext.HASH_SHA256)
	hash_context.update(png)
	var digest := hash_context.finish().hex_encode()
	var image_folder := folder + "/observations"
	var mkdir_error := DirAccess.make_dir_recursive_absolute(image_folder)
	if mkdir_error != OK:
		return {
			"error": "Cannot create observation artifact directory",
			"failure_type": "infrastructure"
		}
	var artifact := image_folder + "/" + digest + ".png"
	if FileAccess.file_exists(artifact):
		if FileAccess.get_sha256(artifact) != digest:
			return {
				"error": "Existing immutable observation artifact has wrong hash",
				"failure_type": "infrastructure"
			}
	else:
		var output := FileAccess.open(artifact, FileAccess.WRITE)
		if output == null:
			return {"error": "Cannot write observation artifact", "failure_type": "infrastructure"}
		output.store_buffer(png)
		output.flush()
		var write_error := output.get_error()
		output.close()
		if write_error != OK:
			return {
				"error": "Cannot persist observation artifact", "failure_type": "infrastructure"
			}
	var focal_px := HEIGHT / (2.0 * tan(deg_to_rad(camera.fov) * 0.5))
	return {
		"episode_id": episode_id,
		"tick": capture_tick,
		"observation_version": VERSION,
		"observation_id": "%s_%d_%s" % [episode_id, capture_tick, digest],
		"image":
		{
			"mime_type": "image/png",
			"base64": Marshalls.raw_to_base64(png),
			"sha256": digest,
			"width": WIDTH,
			"height": HEIGHT
		},
		"artifact": artifact,
		"camera":
		{
			"vertical_fov_degrees": camera.fov,
			"near_m": camera.near,
			"far_m": camera.far,
			"intrinsics": {"fx": focal_px, "fy": focal_px, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0},
			"pose":
			{
				"position": _vector(pose.origin),
				"basis_x": _vector(pose.basis.x),
				"basis_y": _vector(pose.basis.y),
				"basis_z": _vector(pose.basis.z)
			},
			"convention": "Godot world axes, camera forward negative Z, image origin top left",
			"rider_mesh_visible": false,
			"hud_visible": false,
			"mount":
			{"height_m": EYE_HEIGHT_M, "forward_m": EYE_FORWARD_M, "look_down_rad": LOOK_DOWN_RAD},
		},
	}


func _read_image() -> Image:
	return viewport.get_texture().get_image()


func _vector(value: Vector3) -> Array:
	return [value.x, value.y, value.z]
