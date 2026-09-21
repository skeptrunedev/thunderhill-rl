class_name FieldCoverage
extends RefCounted
## Historical visual material regions only. Never used for contact or friction.
const MAX_SEGMENTS := 16
var segments := PackedVector4Array()
var sizes := PackedVector2Array()
var tints := PackedVector3Array()
var grass_retentions := PackedFloat32Array()
var end_radii := PackedFloat32Array()
var tint_strengths := PackedVector2Array()


func configure(data: Variant) -> String:
	segments.clear()
	sizes.clear()
	tints.clear()
	grass_retentions.clear()
	end_radii.clear()
	tint_strengths.clear()
	if (
		not data is Dictionary
		or data.get("schema_version") != 1
		or not data.get("corridors") is Array
	):
		return "Invalid field coverage document"
	var pending_segments := PackedVector4Array()
	var pending_sizes := PackedVector2Array()
	var pending_tints := PackedVector3Array()
	var pending_retentions := PackedFloat32Array()
	var pending_end_radii := PackedFloat32Array()
	var pending_strengths := PackedVector2Array()
	for row: Variant in data.corridors:
		if not row is Dictionary or not row.get("points_local_xz") is Array:
			return "Invalid field corridor"
		var width: Variant = row.get("width_m")
		var feather: Variant = row.get("feather_m")
		if not _positive_number(width) or not _positive_number(feather):
			return "Field corridor requires positive finite width and feather"
		var tint_data: Variant = row.get("tint_linear", [1.0, 1.0, 1.0])
		if not tint_data is Array or tint_data.size() != 3:
			return "Field tint requires three linear channels"
		for channel: Variant in tint_data:
			if not _positive_number(channel) or float(channel) > 1.0:
				return "Field tint channels must be finite and within zero exclusive to one"
		var tint := Vector3(tint_data[0], tint_data[1], tint_data[2])
		var retention: Variant = row.get("grass_retention", 0.0)
		if (
			not (retention is float or retention is int)
			or not is_finite(float(retention))
			or float(retention) < 0.0
			or float(retention) > 1.0
		):
			return "Field grass retention must be finite and between zero and one"
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
		var widths: Variant = row.get("point_widths_m", [])
		if not widths is Array or (not widths.is_empty() and widths.size() != points.size()):
			return "Field point widths must match corridor points"
		for point_width: Variant in widths:
			if (
				not (point_width is float or point_width is int)
				or not is_finite(float(point_width))
				or float(point_width) < 0.0
				or float(point_width) > float(width)
			):
				return "Field point width must be finite and within zero to corridor width"
		var strengths: Variant = row.get("point_tint_strengths", [])
		if (
			not strengths is Array
			or (not strengths.is_empty() and strengths.size() != points.size())
		):
			return "Field tint strengths must match corridor points"
		for strength: Variant in strengths:
			if (
				not (strength is float or strength is int)
				or not is_finite(float(strength))
				or float(strength) < 0.0
				or float(strength) > 1.0
			):
				return "Field tint strength must be finite and between zero and one"
		for index in points.size() - 1:
			var a := points[index]
			var b := points[index + 1]
			if a.distance_squared_to(b) < 0.0001:
				return "Degenerate field corridor segment"
			pending_segments.append(Vector4(a.x, a.y, b.x, b.y))
			var start_radius := float(width if widths.is_empty() else widths[index]) * 0.5
			var end_radius := float(width if widths.is_empty() else widths[index + 1]) * 0.5
			pending_sizes.append(Vector2(start_radius, float(feather)))
			pending_end_radii.append(end_radius)
			pending_strengths.append(
				(
					Vector2.ONE
					if strengths.is_empty()
					else Vector2(strengths[index], strengths[index + 1])
				)
			)
			pending_tints.append(tint)
			pending_retentions.append(float(retention))
	if pending_segments.size() > MAX_SEGMENTS:
		return "Too many field corridor segments"
	segments = pending_segments
	sizes = pending_sizes
	tints = pending_tints
	grass_retentions = pending_retentions
	end_radii = pending_end_radii
	tint_strengths = pending_strengths
	return ""


static func _positive_number(value: Variant) -> bool:
	return (value is float or value is int) and is_finite(float(value)) and float(value) > 0.0


func signed_distance(point: Vector2, index: int) -> float:
	var segment := segments[index]
	var a := Vector2(segment.x, segment.y)
	var delta := Vector2(segment.z, segment.w) - a
	var t := clampf((point - a).dot(delta) / delta.length_squared(), 0.0, 1.0)
	return point.distance_to(a + delta * t) - lerpf(sizes[index].x, end_radii[index], t)


func excludes_stubble(point: Vector2, clump_radius: float) -> bool:
	for index in segments.size():
		if signed_distance(point, index) < clump_radius:
			return true
	return false


func apply_material(material: ShaderMaterial) -> void:
	var padded_segments := segments.duplicate()
	var padded_sizes := sizes.duplicate()
	var padded_tints := tints.duplicate()
	var padded_retentions := grass_retentions.duplicate()
	var padded_end_radii := end_radii.duplicate()
	var padded_strengths := tint_strengths.duplicate()
	padded_segments.resize(MAX_SEGMENTS)
	padded_sizes.resize(MAX_SEGMENTS)
	padded_tints.resize(MAX_SEGMENTS)
	padded_retentions.resize(MAX_SEGMENTS)
	padded_end_radii.resize(MAX_SEGMENTS)
	padded_strengths.resize(MAX_SEGMENTS)
	material.set_shader_parameter("field_segment_count", segments.size())
	material.set_shader_parameter("field_segments", padded_segments)
	material.set_shader_parameter("field_sizes", padded_sizes)
	material.set_shader_parameter("field_tints", padded_tints)
	material.set_shader_parameter("field_grass_retentions", padded_retentions)
	material.set_shader_parameter("field_end_radii", padded_end_radii)
	material.set_shader_parameter("field_tint_strengths", padded_strengths)
