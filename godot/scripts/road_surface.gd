class_name RoadSurface
extends RefCounted
## Experimental analytic ribbon. No dependency on rendering or physics state.
var period := 0.0
var lift := 0.0
var breaks: Array = []
var coefficients: Array = []


func _finite_number(value: Variant) -> bool:
	return (value is float or value is int) and is_finite(float(value))


func configure(data: Dictionary) -> String:
	if data.get("schema_version", 0) != 1:
		return "Unsupported road surface schema"
	if not _finite_number(data.get("period_m")) or not _finite_number(data.get("road_lift_m")):
		return "Invalid road period or lift"
	if not data.get("breaks") is Array or not data.get("coefficients") is Array:
		return "Road surface breaks and coefficients must be arrays"
	var next_period := float(data.period_m)
	var next_breaks: Array = data.get("breaks", [])
	var next_coefficients: Array = data.get("coefficients", [])
	if not is_finite(next_period) or next_period <= 0.0 or next_breaks.size() < 2:
		return "Invalid road surface period or breaks"
	for value in next_breaks:
		if not _finite_number(value):
			return "Invalid road surface break"
	if next_breaks[0] != 0.0 or next_breaks[-1] != next_period:
		return "Road surface breaks must span the period"
	if next_coefficients.size() != next_breaks.size() - 1:
		return "Road surface interval count mismatch"
	for i in next_coefficients.size():
		if not is_finite(float(next_breaks[i])) or next_breaks[i + 1] <= next_breaks[i]:
			return "Invalid road surface interval"
		if not next_coefficients[i] is Array or next_coefficients[i].size() != 4:
			return "Road surface requires four channels"
		for channel in next_coefficients[i]:
			if not channel is Array or channel.size() != 6:
				return "Road surface requires six coefficients per channel"
			for value in channel:
				if not _finite_number(value):
					return "Nonfinite road surface coefficient"
	var next_lift := float(data.get("road_lift_m", NAN))
	if not is_finite(next_lift):
		return "Invalid road lift"
	period = next_period
	lift = next_lift
	breaks = next_breaks.duplicate(true)
	coefficients = next_coefficients.duplicate(true)
	return ""


func _polynomial(c: Array, x: float, derivative: int) -> float:
	var result := 0.0
	for i in range(6 - derivative):
		var factor := 1.0
		for j in derivative:
			factor *= float(5 - i - j)
		result = result * x + float(c[i]) * factor
	return result


func evaluate(station: float, lateral: float) -> Dictionary:
	if period <= 0.0 or not is_finite(station) or not is_finite(lateral):
		return {"error": "Unconfigured surface or nonfinite coordinates"}
	var s := fposmod(station, period)
	var lo := 0
	var hi := coefficients.size()
	while lo + 1 < hi:
		var mid := (lo + hi) / 2
		if float(breaks[mid]) <= s:
			lo = mid
		else:
			hi = mid
	var x := s - float(breaks[lo])
	var c: Array = coefficients[lo]
	var d: Array[Vector3] = []
	var b: Array[float] = []
	for order in 4:
		d.append(
			Vector3(
				_polynomial(c[0], x, order),
				_polynomial(c[1], x, order),
				_polynomial(c[2], x, order)
			)
		)
		b.append(_polynomial(c[3], x, order))
		if not d[order].is_finite() or not is_finite(b[order]):
			return {"error": "Road polynomial evaluation overflowed"}
	var v := Vector3(d[1].z, 0.0, -d[1].x)
	var a := Vector3(d[2].z, 0.0, -d[2].x)
	var j := Vector3(d[3].z, 0.0, -d[3].x)
	var q := v.length()
	if q <= 1e-9 or not is_finite(q):
		return {"error": "Degenerate planar tangent"}
	var qp := v.dot(a) / q
	var qpp := (a.dot(a) + v.dot(j)) / q - pow(v.dot(a), 2) / pow(q, 3)
	var across := v / q + Vector3.UP * b[0]
	var across1 := a / q - v * qp / (q * q) + Vector3.UP * b[1]
	var across2 := (
		j / q
		- 2.0 * a * qp / (q * q)
		- v * qpp / (q * q)
		+ 2.0 * v * qp * qp / pow(q, 3)
		+ Vector3.UP * b[2]
	)
	var rs := d[1] + lateral * across1
	var cross := rs.cross(across)
	if not cross.is_finite() or cross.y <= 1e-9:
		return {"error": "Road surface is folded or degenerate"}
	var normal := cross.normalized()
	if not normal.is_finite() or normal.length_squared() < 0.5:
		return {"error": "Road normal normalization overflowed"}
	var result := {
		"R": d[0] + lateral * across + Vector3.UP * lift,
		"Rs": rs,
		"Ru": across,
		"Rss": d[2] + lateral * across2,
		"Rsu": across1,
		"Ruu": Vector3.ZERO,
		"normal": normal
	}
	for value: Vector3 in result.values():
		if not value.is_finite():
			return {"error": "Road differential evaluation overflowed"}
	return result
