class_name AgentCamera
extends Node
## Observation camera, independent of presentation cameras and UI canvases.

const VERSION := "hyperion-rig-ftheta-v4"
# Alpamayo's native camera rig: NVIDIA Hyperion 8.1, the rig of the PhysicalAI-AV
# clips it was trained on and of the AlpaSim NuRec scenes it is evaluated in
# (nurec-skills skills/nre/references/configs/pai.yaml). Mounts and F-theta lenses
# are copied from NVIDIA/nurec-skills skills/nre/references/rig-json/rig.json:
# nominalSensor2Rig_FLU roll/pitch/yaw degrees and x forward, y left, z up metres
# from the rig origin on the ground, and native-sensor F-theta intrinsics.
# The rig is level and yaw-only (training/driving_trajectory.py convention): a
# car does not lean, so the leaned motorcycle and its rider are not in view.
const VIEWS := [
	{
		"logical_id": "camera_cross_left_120fov",
		"rpy_degrees": Vector3(-0.6424463737193147, -5.054915363939412, 86.13119534398491),
		"t_m": Vector3(1.327272116773123, 0.8489758889327085, 1.3342820647953455),
		"native_size": Vector2(1920, 1536), "cx": 960.00446, "cy": 768.06354,
		"polynomial_type": "pixeldistance-to-angle",
		"polynomial": [0.0, 1.0432383458901103e-03, -4.6963115778795866e-07, 1.4310197988849062e-09, -1.5854996519735615e-12, 6.6713064407140573e-16],
	},
	{
		"logical_id": "camera_front_wide_120fov",
		"rpy_degrees": Vector3(0.3324443354905707, -1.0629351182416593, -0.3012069447868943),
		"t_m": Vector3(2.264681329025869, 0.026183747224370918, 1.500056797243257),
		"native_size": Vector2(3840, 2160), "cx": 1925.9518, "cy": 1080.4784,
		"polynomial_type": "pixeldistance-to-angle",
		"polynomial": [0.0, 5.6192209864210400e-04, -1.8102698450940052e-07, 2.9070366771547837e-10, -1.8056126816901083e-13, 3.9645365870185843e-17],
	},
	{
		"logical_id": "camera_cross_right_120fov",
		"rpy_degrees": Vector3(-0.29237384070053923, -4.024128706133464, -85.83594591073889),
		"t_m": Vector3(1.298143002290808, -0.7262124766412159, 1.3343150705753346),
		"native_size": Vector2(1920, 1536), "cx": 976.25543, "cy": 784.087,
		"polynomial_type": "pixeldistance-to-angle",
		"polynomial": [0.0, 1.1170196775819588e-03, -1.1183822302582922e-06, 3.1896968932457106e-09, -3.4741025147801833e-12, 1.3681700383160935e-15],
	},
	{
		"logical_id": "camera_front_tele_30fov",
		"rpy_degrees": Vector3(-0.2832814925264894, -1.2606928362558618, -0.13135072700058756),
		"t_m": Vector3(2.214747, 0.001131, 1.4791220000000003),
		"native_size": Vector2(3840, 2160), "cx": 1905.8003, "cy": 1082.2957,
		"polynomial_type": "angle-to-pixeldistance",
		"polynomial": [0.0, 4.5799666829571643e+03, -8.8001218238387153e+01, 5.7337297220928122e+02, -1.7236276014691484e+03, 1.9861127340532416e+03],
	},
]
# PhysicalAI-AV stores every camera as 1920x1080 (physical_ai_av wiki, Sensor
# Details): the 4K sensors halved, the 1920x1536 cross sensors cut to 1080 rows.
# AlpaGym then resizes each frame to input_size 320x512 without cropping
# (alpagym_alpamayo_r1 configs/policy/alpamayo_r1.yaml, policies/alpamayo/buffers.py).
const DATASET_SIZE := Vector2(1920, 1080)
const WIDTH := 512
const HEIGHT := 320
const BIKE_LAYER := 1 << 17
const RIDER_LAYER := 1 << 19
const RIDER_LIMB_LAYER := 1 << 18
const LENS_SHADER := """
shader_type canvas_item;
uniform sampler2D source : filter_linear, repeat_disable;
uniform vec2 principal_point;
uniform float linear_c;
uniform float polynomial[6];
uniform bool angle_to_pixel;
uniform vec2 pinhole_extent;
uniform vec2 output_size;

float evaluate(float x) {
	float y = polynomial[5];
	for (int i = 4; i >= 0; i--) {
		y = y * x + polynomial[i];
	}
	return y;
}

vec3 ray(vec2 pixel) {
	vec2 distorted = vec2((pixel.x - principal_point.x) / linear_c, pixel.y - principal_point.y);
	float radius = length(distorted);
	float theta;
	if (angle_to_pixel) {
		theta = radius / polynomial[1];
		for (int i = 0; i < 4; i++) {
			float slope = polynomial[1] + theta * (2.0 * polynomial[2] + theta * (3.0 * polynomial[3]
				+ theta * (4.0 * polynomial[4] + theta * 5.0 * polynomial[5])));
			theta -= (evaluate(theta) - radius) / slope;
		}
	} else {
		theta = evaluate(radius);
	}
	vec2 direction = radius > 0.0 ? distorted / radius : vec2(0.0);
	return vec3(direction * sin(theta), cos(theta));
}

void fragment() {
	// Four samples per output pixel: the lens compresses the periphery of the
	// pinhole source, which has no mipmaps.
	vec3 color = vec3(0.0);
	for (int i = 0; i < 4; i++) {
		vec2 offset = vec2(float(i % 2) - 0.5, float(i / 2) - 0.5) * 0.5;
		vec3 r = ray(UV * output_size + offset);
		color += texture(source, 0.5 + r.xy / (r.z * 2.0 * pinhole_extent)).rgb;
	}
	COLOR = vec4(color * 0.25, 1.0);
}
"""
var pipelines: Array[Dictionary] = []
var manual_draw := false


func configure(world: World3D, offscreen: bool = false) -> void:
	manual_draw = offscreen
	var shader := Shader.new()
	shader.code = LENS_SHADER
	for spec: Dictionary in VIEWS:
		pipelines.append(_build_pipeline(world, spec, shader))


static func lens(spec: Dictionary) -> Dictionary:
	## F-theta intrinsics of the WIDTHxHEIGHT policy image, in AlpaSim's convention:
	## distorted = ((u - cx) / linear_c, v - cy), angle = poly(|distorted|).
	var native: Vector2 = spec.native_size
	var to_dataset := DATASET_SIZE.x / native.x
	var crop_rows := (native.y * to_dataset - DATASET_SIZE.y) * 0.5
	var scale := Vector2(WIDTH / DATASET_SIZE.x, HEIGHT / DATASET_SIZE.y)
	var radius_scale := to_dataset * scale.y
	var polynomial := []
	for i in spec.polynomial.size():
		if spec.polynomial_type == "pixeldistance-to-angle":
			polynomial.append(spec.polynomial[i] / pow(radius_scale, i))
		else:
			polynomial.append(spec.polynomial[i] * radius_scale)
	return {
		"model": "ftheta",
		"width": WIDTH,
		"height": HEIGHT,
		"cx": spec.cx * to_dataset * scale.x,
		"cy": (spec.cy * to_dataset - crop_rows) * scale.y,
		"polynomial_type": spec.polynomial_type,
		"polynomial": polynomial,
		"linear_c": scale.x / scale.y,
		"linear_d": 0.0,
		"linear_e": 0.0,
		"source": "Hyperion 8.1 rig.json %dx%d, scaled to 1920 wide and centre cropped to 1080 rows, resized to %dx%d" % [native.x, native.y, WIDTH, HEIGHT],
	}


static func _angle(intrinsics: Dictionary, radius: float) -> float:
	var p: Array = intrinsics.polynomial
	var value := func(x: float) -> float:
		var y := 0.0
		for i in range(p.size() - 1, -1, -1):
			y = y * x + p[i]
		return y
	if intrinsics.polynomial_type == "pixeldistance-to-angle":
		return value.call(radius)
	var theta: float = radius / p[1]
	for _i in 8:
		var slope := 0.0
		for i in range(1, p.size()):
			slope += i * p[i] * pow(theta, i - 1)
		theta -= (value.call(theta) - radius) / slope
	return theta


static func _pinhole_extent(intrinsics: Dictionary) -> Vector2:
	## Smallest centred pinhole (tan half angles) that contains every output ray.
	var extent := Vector2.ZERO
	var samples := 64
	for i in samples + 1:
		var t := float(i) / samples
		for pixel: Vector2 in [
			Vector2(t * WIDTH, 0), Vector2(t * WIDTH, HEIGHT), Vector2(0, t * HEIGHT), Vector2(WIDTH, t * HEIGHT)
		]:
			var distorted := Vector2((pixel.x - intrinsics.cx) / intrinsics.linear_c, pixel.y - intrinsics.cy)
			var theta := _angle(intrinsics, distorted.length())
			var planar := distorted.normalized() * tan(theta)
			extent = Vector2(maxf(extent.x, absf(planar.x)), maxf(extent.y, absf(planar.y)))
	return extent * 1.02


func _build_pipeline(world: World3D, spec: Dictionary, shader: Shader) -> Dictionary:
	var intrinsics := lens(spec)
	var extent := _pinhole_extent(intrinsics)
	# Match the lens's own centre resolution in the pinhole source.
	var p: Array = intrinsics.polynomial
	var focal_px: float = 1.0 / p[1] if intrinsics.polynomial_type == "pixeldistance-to-angle" else p[1]
	var output := SubViewport.new()
	output.name = "PolicyObservation_" + spec.logical_id
	output.size = Vector2i(WIDTH, HEIGHT)
	output.disable_3d = true
	output.render_target_update_mode = SubViewport.UPDATE_DISABLED
	add_child(output)
	# Nested: the source renders before the lens pass that samples it.
	var source := SubViewport.new()
	source.name = "PinholeSource"
	source.size = Vector2i(ceili(2.0 * extent.x * focal_px), ceili(2.0 * extent.y * focal_px))
	source.world_3d = world
	source.msaa_3d = Viewport.MSAA_2X
	source.render_target_update_mode = SubViewport.UPDATE_DISABLED
	output.add_child(source)
	var camera := Camera3D.new()
	camera.name = "PolicyRigCamera"
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	camera.fov = rad_to_deg(2.0 * atan(extent.y))
	camera.near = 0.06
	camera.far = 3000.0
	camera.cull_mask = ((1 << 20) - 1) & ~(BIKE_LAYER | RIDER_LAYER | RIDER_LIMB_LAYER)
	source.add_child(camera)
	camera.current = true
	var material := ShaderMaterial.new()
	material.shader = shader
	material.set_shader_parameter("source", source.get_texture())
	material.set_shader_parameter("principal_point", Vector2(intrinsics.cx, intrinsics.cy))
	material.set_shader_parameter("linear_c", intrinsics.linear_c)
	material.set_shader_parameter("polynomial", PackedFloat32Array(p))
	material.set_shader_parameter("angle_to_pixel", intrinsics.polynomial_type == "angle-to-pixeldistance")
	material.set_shader_parameter("pinhole_extent", extent)
	material.set_shader_parameter("output_size", Vector2(WIDTH, HEIGHT))
	var rect := ColorRect.new()
	rect.size = Vector2(WIDTH, HEIGHT)
	rect.material = material
	output.add_child(rect)
	return {
		"output": output, "source": source, "camera": camera, "intrinsics": intrinsics,
		"pinhole_extent": extent, "warmed": false,
	}


static func rig_frame(sim: RefCounted) -> Dictionary:
	## Level, yaw-only rig on the ground at the motorcycle's reference point.
	var forward: Vector3 = sim.forward()
	return {"position": sim.position, "forward": forward, "left": Vector3.UP.cross(forward), "up": Vector3.UP}


static func view_transform(sim: RefCounted, spec: Dictionary) -> Transform3D:
	var rig := rig_frame(sim)
	var rpy: Vector3 = spec.rpy_degrees
	# nominalSensor2Rig_FLU: extrinsic x-y-z Euler angles, R = Rz(yaw) Ry(pitch) Rx(roll).
	var sensor := (
		Basis(Vector3(0, 0, 1), deg_to_rad(rpy.z))
		* Basis(Vector3(0, 1, 0), deg_to_rad(rpy.y))
		* Basis(Vector3(1, 0, 0), deg_to_rad(rpy.x))
	)
	var to_world := func(flu: Vector3) -> Vector3:
		return rig.forward * flu.x + rig.left * flu.y + rig.up * flu.z
	# Godot camera: +X right, +Y up, looking along -Z.
	var basis := Basis(-to_world.call(sensor.y), to_world.call(sensor.z), -to_world.call(sensor.x))
	return Transform3D(basis, rig.position + to_world.call(spec.t_m))


static func assign_rider_layer(node: Node, layer: int = RIDER_LAYER) -> void:
	if node is VisualInstance3D:
		node.layers = layer
	for child in node.get_children():
		assign_rider_layer(child, layer)


func capture(sim: RefCounted, _normal: Vector3, episode_id: String, folder: String) -> Dictionary:
	var tick: int = sim.tick
	var views: Array[Dictionary] = []
	for index in VIEWS.size():
		var view := await _capture_view(sim, episode_id, folder, VIEWS[index], pipelines[index])
		if view.has("error"):
			return view
		if sim.tick != tick:
			return {"error": "Simulation changed during multiview capture", "failure_type": "infrastructure"}
		views.append(view)
	# Keep the front wide receipt at top level for existing camera clients.
	var receipt := views[1].duplicate(true)
	receipt["views"] = views
	return receipt


func _capture_view(sim: RefCounted, episode_id: String, folder: String, spec: Dictionary, pipeline: Dictionary) -> Dictionary:
	var trace_started := Time.get_ticks_usec()
	_trace_capture(spec.logical_id, "begin", trace_started)
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
	var camera: Camera3D = pipeline.camera
	var source: SubViewport = pipeline.source
	var output: SubViewport = pipeline.output
	camera.transform = view_transform(sim, spec)
	var capture_tick: int = sim.tick
	var pose := camera.global_transform
	source.render_target_update_mode = SubViewport.UPDATE_ONCE
	output.render_target_update_mode = SubViewport.UPDATE_ONCE
	if manual_draw:
		# SceneTree must first flush pending Node3D transform notifications.
		# The caller holds the agent request lock across this frame boundary.
		await get_tree().process_frame
		camera.force_update_transform()
		if not pipeline.warmed:
			# Ordinary rendering initializes scene/shadow resources before an
			# observation arrives. On demand rendering needs the same initial pass.
			_trace_capture(spec.logical_id, "warm_draw_begin", trace_started)
			RenderingServer.force_draw(false)
			_trace_capture(spec.logical_id, "warm_draw_end", trace_started)
			await get_tree().process_frame
			camera.force_update_transform()
			pipeline.warmed = true
			source.render_target_update_mode = SubViewport.UPDATE_ONCE
			output.render_target_update_mode = SubViewport.UPDATE_ONCE
		# Readback synchronizes this draw; no X11 surface is presented.
		_trace_capture(spec.logical_id, "draw_begin", trace_started)
		RenderingServer.force_draw(false)
		_trace_capture(spec.logical_id, "draw_end", trace_started)
	else:
		await RenderingServer.frame_post_draw
	if sim.tick != capture_tick:
		return {
			"error": "Simulation changed during camera capture", "failure_type": "infrastructure"
		}
	_trace_capture(spec.logical_id, "readback_begin", trace_started)
	var image := output.get_texture().get_image()
	_trace_capture(spec.logical_id, "readback_end", trace_started)
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
		var output_file := FileAccess.open(artifact, FileAccess.WRITE)
		if output_file == null:
			return {"error": "Cannot write observation artifact", "failure_type": "infrastructure"}
		output_file.store_buffer(png)
		output_file.flush()
		var write_error := output_file.get_error()
		output_file.close()
		if write_error != OK:
			return {
				"error": "Cannot persist observation artifact", "failure_type": "infrastructure"
			}
	var rig := rig_frame(sim)
	return {
		"logical_id": spec.logical_id,
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
			"renderer":
			{
				"adapter": RenderingServer.get_video_adapter_name(),
				"driver": RenderingServer.get_current_rendering_driver_name(),
				"method": RenderingServer.get_current_rendering_method(),
				"offscreen": manual_draw,
			},
			"intrinsics": pipeline.intrinsics,
			"pinhole_source":
			{"size": [source.size.x, source.size.y], "tan_half_angles": [pipeline.pinhole_extent.x, pipeline.pinhole_extent.y]},
			"near_m": camera.near,
			"far_m": camera.far,
			"pose":
			{
				"position": _vector(pose.origin),
				"basis_x": _vector(pose.basis.x),
				"basis_y": _vector(pose.basis.y),
				"basis_z": _vector(pose.basis.z)
			},
			"rig":
			{
				"position": _vector(rig.position),
				"forward": _vector(rig.forward),
				"left": _vector(rig.left),
				"up": _vector(rig.up)
			},
			"convention": "Godot world axes, camera forward negative Z, image origin top left",
			"rider_mesh_visible": false,
			"bike_mesh_visible": false,
			"hud_visible": false,
			"mount":
			{"rig": "NVIDIA Hyperion 8.1", "rpy_degrees": _vector(spec.rpy_degrees), "t_m": _vector(spec.t_m)},
		},
	}


func _vector(value: Vector3) -> Array:
	return [value.x, value.y, value.z]


func _trace_capture(view: String, stage: String, started_usec: int) -> void:
	if OS.get_environment("THUNDERHILL_CAPTURE_DIAGNOSTICS") == "1":
		print("THUNDERHILL_CAPTURE " + JSON.stringify({
			"view": view, "stage": stage,
			"elapsed_ms": (Time.get_ticks_usec() - started_usec) / 1000.0,
		}))
