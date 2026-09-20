extends SceneTree
const Pose = preload("res://scripts/rider_pose.gd")
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var bike := preload("res://scripts/bike_visual.gd").new()
	root.add_child(bike)
	var minimum_height := INF
	var minimum_separation := INF
	var minimum_distance := INF
	var maximum_distance := 0.0
	for side in [-1.0, 1.0]:
		var bounds: Dictionary = Pose.domain_bounds(bike._front.position, side, -0.5, 0.5)
		check(not bounds.has("error"), "Full arm domain was not enclosed")
		print("RIDER_DOMAIN ", side, " ", JSON.stringify(bounds))
		var previous := {}
		for key in ["upper", "lower"]:
			var motion: Dictionary = Pose.point_motion_bounds(bounds, key, 0.2)
			check(
				not motion.has("error") and motion.speed > 0.0 and motion.acceleration > 0.0,
				"Invalid arm derivative bounds"
			)
			for index in range(99):
				var angle := -0.49 + index * 0.01
				var a: Transform3D = Pose.solve(bike._front.position, side, angle - 0.001)[key]
				var b: Transform3D = Pose.solve(bike._front.position, side, angle)[key]
				var c: Transform3D = Pose.solve(bike._front.position, side, angle + 0.001)[key]
				for q in [Vector3(0.2, 0, 0), Vector3(0, 0.2, 0), Vector3(0, 0, 0.2)]:
					check((b * q).length() <= motion.reach, "Arm point escaped reach bound")
					check(
						((c * q - a * q) / 0.002).length() <= motion.speed,
						"Finite difference exceeded speed bound"
					)
					check(
						(
							((c * q - 2.0 * (b * q) + a * q) / 0.000001).length()
							<= motion.acceleration
						),
						"Finite difference exceeded acceleration bound"
					)
		for index in range(1001):
			var steering := -0.5 + index * 0.001
			var pose: Dictionary = Pose.solve(bike._front.position, side, steering)
			check(not pose.has("error"), "Supported steering rejected")
			if pose.has("error"):
				continue
			check(
				pose.distance >= bounds.d_min and pose.distance <= bounds.d_max,
				"Distance escaped domain enclosure"
			)
			check(
				pose.pole_separation >= bounds.s_min and pose.bend_height >= bounds.h_min,
				"Pose escaped nonsingular margins"
			)
			minimum_height = minf(minimum_height, pose.bend_height)
			minimum_separation = minf(minimum_separation, pose.pole_separation)
			minimum_distance = minf(minimum_distance, pose.distance)
			maximum_distance = maxf(maximum_distance, pose.distance)
			bike.update_pose(0.0, steering, 0.0)
			var visual_grip: Transform3D = bike._front.global_transform * Pose.grip_transform(side)
			check(pose.glove.is_equal_approx(visual_grip), "Glove detached from actual front joint")
			check(
				(pose.glove * Pose.CUFF_CENTER).distance_to(pose.wrist) < 0.000001,
				"Cuff detached from wrist"
			)
			for key in ["upper", "lower"]:
				var frame: Transform3D = pose[key]
				check(
					frame.is_finite() and absf(frame.basis.determinant() - 1.0) < 0.00001,
					"Invalid bone frame"
				)
				var a: Vector3 = pose.shoulder if key == "upper" else pose.elbow
				var b: Vector3 = pose.elbow if key == "upper" else pose.wrist
				check(absf(a.distance_to(b) - Pose.BONE_LENGTH) < 0.00001, "Arm bone stretched")
				check(
					(frame * Vector3(0, -Pose.BONE_LENGTH * 0.5, 0)).distance_to(a) < 0.00001,
					"Bone start mismatch"
				)
				check(
					(frame * Vector3(0, Pose.BONE_LENGTH * 0.5, 0)).distance_to(b) < 0.00001,
					"Bone end mismatch"
				)
				if not previous.is_empty():
					check(
						frame.origin.distance_to(previous[key].origin) < 0.002,
						"Discontinuous bone origin"
					)
					check(
						(
							frame.basis.get_rotation_quaternion().angle_to(
								previous[key].basis.get_rotation_quaternion()
							)
							< 0.02
						),
						"Bone frame flipped"
					)
			previous = pose
	var arms := preload("res://scripts/rider_arm_visual.gd").new()
	bike.add_child(arms)
	check(arms.build(bike._front.position).is_empty(), "Arm visual construction failed")
	for angle in [-0.5, 0.0, 0.5]:
		check(arms.set_steering(angle).is_empty(), "Arm visual pose failed")
		for side in [-1.0, 1.0]:
			var pose: Dictionary = Pose.solve(bike._front.position, side, angle)
			for key in ["upper", "lower", "glove"]:
				check(
					arms.joints[side][key].transform.is_equal_approx(pose[key]),
					"Visual joint differs from shared pose"
				)
	var saved: Transform3D = arms.joints[1.0].upper.transform
	check(not arms.set_steering(0.6).is_empty(), "Invalid visual steering accepted")
	check(arms.joints[1.0].upper.transform == saved, "Rejected pose partially mutated visual")
	check(
		Pose.domain_bounds(bike._front.position, 1.0, 0.5, -0.5).has("error"),
		"Reversed interval accepted"
	)
	check(
		Pose.domain_bounds(Vector3(3, 0, 0), 1.0, -0.5, 0.5).has("error"),
		"Oversized origin accepted"
	)
	for steering in [-0.5001, 0.5001, INF, NAN]:
		check(
			Pose.solve(bike._front.position, 1.0, steering).has("error"),
			"Unsupported steering accepted"
		)
	check(Pose.solve(bike._front.position, 0.0, 0.0).has("error"), "Invalid side accepted")
	print(
		"RIDER_POSE_CHECK ",
		JSON.stringify(
			{
				"failures": failures,
				"sampled_min_distance": minimum_distance,
				"sampled_max_distance": maximum_distance,
				"sampled_min_bend_height": minimum_height,
				"sampled_min_pole_separation": minimum_separation
			}
		)
	)
	quit(0 if failures == 0 else 1)
