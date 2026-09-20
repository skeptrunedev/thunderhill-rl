class_name LidarHeightAtlas
extends RefCounted
## C2 partition of unity over tensor cubic lidar patches.
const Surface = preload("res://scripts/lidar_height_surface.gd")
var _radius := 0.0
var _patches: Array = []
var _cells: Dictionary = {}


func _finite_number(value: Variant) -> bool:
	return (value is float or value is int) and is_finite(float(value))


func _cell(x: float, z: float, radius: float) -> Variant:
	var ix := floorf(x / radius)
	var iz := floorf(z / radius)
	# Vector2i indices must remain representable, including covered neighbors.
	if not is_finite(ix) or not is_finite(iz) or absf(ix) > 2147483646.0 or absf(iz) > 2147483646.0:
		return null
	return Vector2i(int(ix), int(iz))


func configure(data: Dictionary) -> bool:
	if not _finite_number(data.get("schema_version")) or data.schema_version != 1:
		return false
	if not _finite_number(data.get("support_radius_m")) or data.support_radius_m <= 0.0:
		return false
	var radius := float(data.support_radius_m)
	if not is_finite(1.0 / radius / radius):
		return false
	if not data.get("patches") is Array or data.patches.is_empty():
		return false
	var patches: Array = []
	var cells: Dictionary = {}
	var centers: Dictionary = {}
	for config in data.patches:
		if not config is Dictionary:
			return false
		var surface := Surface.new()
		if not surface.configure(config):
			return false
		for axis in 2:
			if float(config.knots[axis][3]) > -radius or float(config.knots[axis][-4]) < radius:
				return false
		var x := float(config.origin[0])
		var z := float(config.origin[1])
		# Nested scalar keys preserve full origin precision.
		if centers.has(x) and centers[x].has(z):
			return false
		if not centers.has(x):
			centers[x] = {}
		centers[x][z] = true
		var lo: Variant = _cell(x - radius, z - radius, radius)
		var hi: Variant = _cell(x + radius, z + radius, radius)
		if lo == null or hi == null or x - radius == x + radius or z - radius == z + radius:
			return false
		var index := patches.size()
		patches.append({"surface": surface, "x": x, "z": z})
		for ix in range(lo.x, hi.x + 1):
			for iz in range(lo.y, hi.y + 1):
				var key := Vector2i(ix, iz)
				if not cells.has(key):
					cells[key] = []
				cells[key].append(index)
	_radius = radius
	_patches = patches
	_cells = cells
	return true


func _weight(q: float) -> Array:
	var t := q / _radius
	if absf(t) >= 1.0:
		return [0.0, 0.0, 0.0]
	var a := 1.0 - t * t
	return [
		a * a * a, -6.0 * t * a * a / _radius, (-6.0 * a * a + 24.0 * t * t * a) / _radius / _radius
	]


func evaluate(x: float, z: float) -> Dictionary:
	if _patches.is_empty() or not is_finite(x) or not is_finite(z):
		return {}
	var key: Variant = _cell(x, z, _radius)
	if key == null or not _cells.has(key):
		return {}
	# Scalar doubles keep quotient cancellation out of float32 vectors.
	var w := PackedFloat64Array([0., 0., 0., 0., 0., 0.])
	var f := PackedFloat64Array([0., 0., 0., 0., 0., 0.])
	var active_count := 0
	var sole_sample: Dictionary = {}
	for index in _cells[key]:
		var patch: Dictionary = _patches[index]
		var wx := _weight(x - patch.x)
		var wz := _weight(z - patch.z)
		if wx[0] <= 0.0 or wz[0] <= 0.0:
			continue
		var sample: Dictionary = patch.surface.evaluate(x, z)
		if sample.is_empty():
			return {}
		active_count += 1
		sole_sample = sample
		var p := [
			wx[0] * wz[0], wx[1] * wz[0], wx[0] * wz[1], wx[2] * wz[0], wx[1] * wz[1], wx[0] * wz[2]
		]
		var h := float(sample.height)
		var hx := float(sample.gradient.x)
		var hz := float(sample.gradient.y)
		for i in 6:
			w[i] += p[i]
			f[i] += p[i] * h
		f[1] += p[0] * hx
		f[2] += p[0] * hz
		f[3] += 2.0 * p[1] * hx + p[0] * sample.hessian.x
		f[4] += p[1] * hz + p[2] * hx + p[0] * sample.hessian.y
		f[5] += 2.0 * p[2] * hz + p[0] * sample.hessian.z
	if active_count == 1:
		# The normalized weight is identically one. Avoid cancellation near the
		# outer support boundary, where both numerator and denominator vanish.
		return sole_sample
	if w[0] <= 0.0 or not is_finite(w[0]):
		return {}
	var height := f[0] / w[0]
	var hx := (f[1] - height * w[1]) / w[0]
	var hz := (f[2] - height * w[2]) / w[0]
	var hxx := (f[3] - height * w[3] - 2.0 * hx * w[1]) / w[0]
	var hxz := (f[4] - height * w[4] - hx * w[2] - hz * w[1]) / w[0]
	var hzz := (f[5] - height * w[5] - 2.0 * hz * w[2]) / w[0]
	var gradient := Vector2(hx, hz)
	var hessian := Vector3(hxx, hxz, hzz)
	if not is_finite(height) or not gradient.is_finite() or not hessian.is_finite():
		return {}
	return {"height": height, "gradient": gradient, "hessian": hessian}
