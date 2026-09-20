class_name LidarHeightSurface
extends RefCounted
## Analytic C2 tensor cubic height field. Coordinates and heights are local metres.
var _knots: Array = []
var _coefficients: Array = []
var _origin: Array = []


func _finite_number(value: Variant) -> bool:
	return (value is float or value is int) and is_finite(float(value))


func configure(data: Dictionary) -> bool:
	if not _finite_number(data.get("schema_version")) or data.schema_version != 1:
		return false
	if not data.get("degree") is Array or data.degree.size() != 2:
		return false
	for value in data.degree:
		if not _finite_number(value) or value != 3:
			return false
	if not data.get("origin") is Array or data.origin.size() != 2:
		return false
	for value in data.origin:
		if not _finite_number(value):
			return false
	if not data.get("knots") is Array or data.knots.size() != 2:
		return false
	for axis in data.knots:
		if not axis is Array or axis.size() < 8:
			return false
		for value in axis:
			if not _finite_number(value):
				return false
		var count: int = axis.size() - 4
		if axis[3] >= axis[count] or not is_finite(float(axis[count]) - float(axis[3])):
			return false
		for i in 4:
			if axis[i] != axis[3] or axis[count + i] != axis[count]:
				return false
		# Strictly increasing interior knots guarantee C2 continuity.
		for i in range(4, count + 1):
			if axis[i] <= axis[i - 1]:
				return false
	if not data.get("coefficients") is Array:
		return false
	if data.coefficients.size() != data.knots[0].size() - 4:
		return false
	for row in data.coefficients:
		if not row is Array or row.size() != data.knots[1].size() - 4:
			return false
		for value in row:
			if not _finite_number(value):
				return false
	_knots = data.knots.duplicate(true)
	_coefficients = data.coefficients.duplicate(true)
	_origin = data.origin.duplicate(true)
	return true


func _basis(knots: Array, q: float) -> Array:
	# Only four cubic basis functions have support at any point.
	var span := knots.size() - 5
	if q < float(knots[-4]):
		var lo := 3
		var hi := knots.size() - 4
		while lo + 1 < hi:
			var mid := (lo + hi) / 2
			if q >= float(knots[mid]):
				lo = mid
			else:
				hi = mid
		span = lo
	# At the upper endpoint the last nonempty span supplies the left limit.
	var values := PackedFloat64Array([1.0])
	var first := PackedFloat64Array([0.0])
	var second := PackedFloat64Array([0.0])
	for degree in range(1, 4):
		var next_values := PackedFloat64Array()
		next_values.resize(degree + 1)
		var next_first := next_values.duplicate()
		var next_second := next_values.duplicate()
		for k in degree + 1:
			var i := span - degree + k
			var left := float(knots[i + degree]) - float(knots[i])
			var right := float(knots[i + degree + 1]) - float(knots[i + 1])
			if k > 0 and left > 0.0:
				next_values[k] += (q - float(knots[i])) / left * values[k - 1]
				next_first[k] += degree * values[k - 1] / left
				next_second[k] += degree * first[k - 1] / left
			if k < degree and right > 0.0:
				next_values[k] += (float(knots[i + degree + 1]) - q) / right * values[k]
				next_first[k] -= degree * values[k] / right
				next_second[k] -= degree * first[k] / right
		values = next_values
		first = next_first
		second = next_second
	return [values, first, second, span - 3]


func evaluate(x: float, z: float) -> Dictionary:
	if _knots.is_empty() or not is_finite(x) or not is_finite(z):
		return {}
	var q := [x - float(_origin[0]), z - float(_origin[1])]
	for axis in 2:
		if not is_finite(q[axis]) or q[axis] < _knots[axis][3] or q[axis] > _knots[axis][-4]:
			return {}
	var bx := _basis(_knots[0], q[0])
	var bz := _basis(_knots[1], q[1])
	var height := 0.0
	var hx := 0.0
	var hz := 0.0
	var hxx := 0.0
	var hxz := 0.0
	var hzz := 0.0
	for i in 4:
		for j in 4:
			var c := float(_coefficients[bx[3] + i][bz[3] + j])
			height += c * bx[0][i] * bz[0][j]
			hx += c * bx[1][i] * bz[0][j]
			hz += c * bx[0][i] * bz[1][j]
			hxx += c * bx[2][i] * bz[0][j]
			hxz += c * bx[1][i] * bz[1][j]
			hzz += c * bx[0][i] * bz[2][j]
	var gradient := Vector2(hx, hz)
	var hessian := Vector3(hxx, hxz, hzz)
	if not is_finite(height) or not gradient.is_finite() or not hessian.is_finite():
		return {}
	return {"height": height, "gradient": gradient, "hessian": hessian}
