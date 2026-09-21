class_name FieldCoverage
extends RefCounted
## Historical visual material regions only. Never used for contact or friction.
const MAX_SEGMENTS := 16
var segments := PackedVector4Array()
var sizes := PackedVector2Array()


func configure(data: Variant) -> String:
	segments.clear()
	sizes.clear()
	if (
		not data is Dictionary
		or data.get("schema_version") != 1
		or not data.get("corridors") is Array
	):
		return "Invalid field coverage document"
	var pending_segments := PackedVector4Array()
	var pending_sizes := PackedVector2Array()
	for row: Variant in data.corridors:
		if not row is Dictionary or not row.get("points_local_xz") is Array:
			return "Invalid field corridor"
		var width: Variant = row.get("width_m")
		var feather: Variant = row.get("feather_m")
		if not _positive_number(width) or not _positive_number(feather):
			return "Field corridor requires positive finite width and feather"
		var points := PackedVector2Array()
		for point: Variant in row.points_local_xz:
			if not point is Array or point.size() != 2:
				return "Invalid field corridor point"
			for coordinate: Variant in point:
				if (
					not (coordinate is float or coordinate is int)
					or not is_finite(float(coordinate))
				):
					return "Invalid field corridor coordinate"
			points.append(Vector2(point[0], point[1]))
		if points.size() < 2:
			return "Field corridor requires at least two points"
		for index in points.size() - 1:
			var a := points[index]
			var b := points[index + 1]
			if a.distance_squared_to(b) < 0.0001:
				return "Degenerate field corridor segment"
			pending_segments.append(Vector4(a.x, a.y, b.x, b.y))
			pending_sizes.append(Vector2(float(width) * 0.5, float(feather)))
	if pending_segments.size() > MAX_SEGMENTS:
		return "Too many field corridor segments"
	segments = pending_segments
	sizes = pending_sizes
	return ""


static func _positive_number(value: Variant) -> bool:
	return (value is float or value is int) and is_finite(float(value)) and float(value) > 0.0


func signed_distance(point: Vector2, index: int) -> float:
	var segment := segments[index]
	var a := Vector2(segment.x, segment.y)
	var delta := Vector2(segment.z, segment.w) - a
	var t := clampf((point - a).dot(delta) / delta.length_squared(), 0.0, 1.0)
	return point.distance_to(a + delta * t) - sizes[index].x


func excludes_stubble(point: Vector2, clump_radius: float) -> bool:
	for index in segments.size():
		if signed_distance(point, index) < clump_radius:
			return true
	return false


func apply_material(material: ShaderMaterial) -> void:
	var padded_segments := segments.duplicate()
	var padded_sizes := sizes.duplicate()
	padded_segments.resize(MAX_SEGMENTS)
	padded_sizes.resize(MAX_SEGMENTS)
	material.set_shader_parameter("field_segment_count", segments.size())
	material.set_shader_parameter("field_segments", padded_segments)
	material.set_shader_parameter("field_sizes", padded_sizes)
