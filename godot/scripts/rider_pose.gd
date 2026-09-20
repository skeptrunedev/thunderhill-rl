extends RefCounted
## Shared original arm pose study. Artist dimensions, not measured anthropometry.
## Not wired into gameplay until continuous collision bounds cover these joints.
const VERSION := "rider-arm-study-v1"
const STEERING_LIMIT := 0.5
const BONE_LENGTH := 0.33
const CUFF_CENTER := Vector3(0, 0.0145, 0.115)


static func grip_transform(side: float) -> Transform3D:
	var inner := Vector3(side * 0.30, 0.715, 0.25)
	var outer := Vector3(side * 0.40, 0.70, 0.27)
	var x := ((outer - inner) * side).normalized()
	var z := Vector3.BACK.slide(x).normalized()
	return Transform3D(Basis(x, z.cross(x).normalized(), z), (inner + outer) * 0.5)


static func solve(front_origin: Vector3, side: float, steering: float) -> Dictionary:
	if side not in [-1.0, 1.0] or not front_origin.is_finite() or not is_finite(steering):
		return {"error": "Invalid rider pose input"}
	if absf(steering) > STEERING_LIMIT:
		return {"error": "Rider pose steering is outside the supported domain"}
	var front := Transform3D(Basis(Vector3.UP, steering), front_origin)
	var glove := front * grip_transform(side)
	var wrist := glove * CUFF_CENTER
	var shoulder := Vector3(side * 0.185, 1.285, -0.12)
	var delta := wrist - shoulder
	var distance := delta.length()
	if distance <= 0.001 or distance >= 2.0 * BONE_LENGTH:
		return {"error": "Rider arm target is folded or unreachable"}
	var along := delta / distance
	var pole := Vector3(side * 0.5, -1.0, 0.8).normalized()
	var bend := pole.slide(along)
	var separation := bend.length()
	if separation <= 0.001:
		return {"error": "Rider elbow bend plane is singular"}
	var height := sqrt(BONE_LENGTH * BONE_LENGTH - distance * distance * 0.25)
	var elbow := shoulder + delta * 0.5 + bend / separation * height
	var normal := along.cross(pole).normalized()
	return {
		"shoulder": shoulder,
		"elbow": elbow,
		"wrist": wrist,
		"glove": glove,
		"upper": _bone_frame(shoulder, elbow, normal),
		"lower": _bone_frame(elbow, wrist, normal),
		"distance": distance,
		"bend_height": height,
		"pole_separation": separation,
	}


static func _bone_frame(a: Vector3, b: Vector3, normal: Vector3) -> Transform3D:
	var y := (b - a).normalized()
	return Transform3D(Basis(normal, y, normal.cross(y).normalized()), (a + b) * 0.5)


static func domain_bounds(
	front_origin: Vector3, side: float, lower: float, upper: float
) -> Dictionary:
	# Chord enclosures cover every angle between samples. The explicit 0.1 mm
	# broadening covers numeric evaluation at these metre scale local coordinates.
	# This is not a sampled minimum used as a purported continuous bound.
	const MARGIN := 0.0001
	if side not in [-1.0, 1.0] or not front_origin.is_finite():
		return {"error": "Invalid rider domain input"}
	if maxf(absf(front_origin.x), maxf(absf(front_origin.y), absf(front_origin.z))) > 2.0:
		return {"error": "Rider origin exceeds local numeric domain"}
	if (
		not is_finite(lower)
		or not is_finite(upper)
		or lower > upper
		or lower < -STEERING_LIMIT
		or upper > STEERING_LIMIT
	):
		return {"error": "Invalid rider steering interval"}
	var target := grip_transform(side) * CUFF_CENTER
	var radius := Vector2(target.x, target.z).length() + MARGIN
	var shoulder := Vector3(side * 0.185, 1.285, -0.12)
	var pole := Vector3(side * 0.5, -1.0, 0.8).normalized()
	var count := maxi(1, ceili((upper - lower) / 0.01))
	var width := (upper - lower) / count
	var error := 2.0 * radius * sin(width * 0.25) + MARGIN
	var d_min := INF
	var d_max := 0.0
	var s_min := INF
	for index in range(count):
		var angle := lower + (index + 0.5) * width
		var delta := front_origin + Basis(Vector3.UP, angle) * target - shoulder
		var distance := delta.length()
		var low := distance - error
		var high := distance + error
		var cross_low := pole.cross(delta).length() - error
		if low <= 0.0 or high >= 2.0 * BONE_LENGTH or cross_low <= 0.0:
			return {"error": "Rider interval cannot certify a nonsingular pose"}
		d_min = minf(d_min, low)
		d_max = maxf(d_max, high)
		s_min = minf(s_min, cross_low / high)
	return {
		"radius": radius,
		"d_min": d_min,
		"d_max": d_max,
		"s_min": s_min,
		"h_min": sqrt(BONE_LENGTH * BONE_LENGTH - d_max * d_max * 0.25),
		"h_max": sqrt(BONE_LENGTH * BONE_LENGTH - d_min * d_min * 0.25),
		"intervals": count,
		"numerical_margin_m": MARGIN
	}


static func point_motion_bounds(domain: Dictionary, joint: String, radius: float) -> Dictionary:
	# Derivatives are with respect to steering radians, in bike local space.
	# A point is O + q.x*x + q.y*y + q.z*z with |q| <= radius.
	if (
		domain.has("error")
		or joint not in ["upper", "lower"]
		or not is_finite(radius)
		or radius < 0.0
	):
		return {"error": "Invalid arm motion bound input"}
	var r: float = domain.radius
	var d_min: float = domain.d_min
	var d_max: float = domain.d_max
	var s_min: float = domain.s_min
	var h_min: float = domain.h_min
	var h_max: float = domain.h_max
	var u1 := r / d_min
	var u2 := r / d_min + 3.0 * r * r / (d_min * d_min)
	var x1 := u1 / s_min
	var x2 := u2 / s_min + 3.0 * u1 * u1 / (s_min * s_min)
	var n1 := x1 + u1
	var n2 := x2 + 2.0 * x1 * u1 + u2
	var h1 := d_max * r / (4.0 * h_min)
	var h2 := (r * r + d_max * r) / (4.0 * h_min) + pow(d_max * r, 2) / (16.0 * pow(h_min, 3))
	var e1 := r * 0.5 + h1 + h_max * n1
	var e2 := r * 0.5 + h2 + 2.0 * h1 * n1 + h_max * n2
	var endpoint1 := e1 + (r if joint == "lower" else 0.0)
	var endpoint2 := e2 + (r if joint == "lower" else 0.0)
	var y1 := endpoint1 / BONE_LENGTH
	var y2 := endpoint2 / BONE_LENGTH
	var z1 := x1 + y1
	var z2 := x2 + 2.0 * x1 * y1 + y2
	return {
		"reach":
		(
			Vector3(0.185, 1.285, -0.12).length()
			+ BONE_LENGTH * (1.5 if joint == "lower" else 0.5)
			+ radius
		),
		"speed": endpoint1 * 0.5 + radius * sqrt(x1 * x1 + y1 * y1 + z1 * z1),
		"acceleration": endpoint2 * 0.5 + radius * sqrt(x2 * x2 + y2 * y2 + z2 * z2),
	}
