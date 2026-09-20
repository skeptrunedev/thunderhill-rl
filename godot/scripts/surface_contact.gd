class_name SurfaceContact
extends RefCounted
## Point contact mechanics for the upcoming free contact model.
## Not a suspension, tire, chassis or collision detection implementation.
## Surface derivatives must describe one smooth surface in world coordinates.


static func support(
	mass_kg: float,
	velocity: Vector3,
	external_acceleration: Vector3,
	du: Vector3,
	dv: Vector3,
	duu: Vector3,
	duv: Vector3,
	dvv: Vector3
) -> Dictionary:
	if not is_finite(mass_kg) or mass_kg <= 0.0:
		return {"error": "Contact mass must be finite and positive"}
	for value: Vector3 in [velocity, external_acceleration, du, dv, duu, duv, dvv]:
		if not value.is_finite():
			return {"error": "Contact vectors must be finite"}
	# Solve the first fundamental form for coordinate rates. This supports
	# nonorthogonal coordinates and parameters that are not measured in metres.
	var e := du.dot(du)
	var f := du.dot(dv)
	var g := dv.dot(dv)
	var determinant := e * g - f * f
	if not is_finite(determinant) or e <= 0.0 or g <= 0.0 or determinant <= 1e-8 * e * g:
		return {"error": "Surface coordinates are singular or nearly parallel"}
	var normal := du.cross(dv).normalized()
	if normal.y < 0.0:
		normal = -normal
	if not normal.is_finite() or normal.y <= 0.0:
		return {"error": "Road contact requires an upward surface normal"}
	var separating_speed := velocity.dot(normal)
	if absf(separating_speed) > 1e-5 * maxf(1.0, velocity.length()):
		return {
			"error": "Support requires tangent velocity at contact; resolve flight or impact first"
		}
	var vu := velocity.dot(du)
	var vv := velocity.dot(dv)
	var u_rate := (g * vu - f * vv) / determinant
	var v_rate := (e * vv - f * vu) / determinant
	# Tangent acceleration terms have zero normal projection. Only the second
	# fundamental form supplies acceleration needed to follow the surface.
	var normal_acceleration := normal.dot(
		duu * u_rate * u_rate + duv * (2.0 * u_rate * v_rate) + dvv * v_rate * v_rate
	)
	var required_force := mass_kg * (normal_acceleration - external_acceleration.dot(normal))
	if not is_finite(required_force):
		return {"error": "Contact calculation overflowed"}
	return {
		"normal": normal,
		"coordinate_velocity": Vector2(u_rate, v_rate),
		"normal_acceleration_m_s2": normal_acceleration,
		"required_normal_force_n": required_force,
		"normal_force_n": maxf(0.0, required_force),
		"release": required_force <= 0.0
	}


static func impact(
	mass_kg: float, velocity: Vector3, normal: Vector3, restitution: float
) -> Dictionary:
	# Call only at an independently located contact event. No position snapping.
	if not is_finite(mass_kg) or mass_kg <= 0.0:
		return {"error": "Impact mass must be finite and positive"}
	if (
		not velocity.is_finite()
		or not normal.is_finite()
		or not is_finite(normal.length_squared())
		or normal.length_squared() <= 0.0
	):
		return {"error": "Impact requires finite velocity and a nonzero normal"}
	if not is_finite(restitution) or restitution < 0.0 or restitution > 1.0:
		return {"error": "Restitution must be finite and between zero and one"}
	normal = normal.normalized()
	var incoming := minf(velocity.dot(normal), 0.0)
	var impulse := -(1.0 + restitution) * mass_kg * incoming
	var outgoing := velocity + normal * (impulse / mass_kg)
	var lost_energy := 0.5 * mass_kg * (1.0 - restitution * restitution) * incoming * incoming
	if not outgoing.is_finite() or not is_finite(impulse) or not is_finite(lost_energy):
		return {"error": "Impact calculation overflowed"}
	return {"velocity": outgoing, "normal_impulse_ns": impulse, "energy_dissipated_j": lost_energy}


static func free_motion(
	position: Vector3, velocity: Vector3, external_acceleration: Vector3, dt: float
) -> Dictionary:
	# Exact only while external acceleration is constant. The caller must split
	# integration at contact events and force changes.
	if (
		not position.is_finite()
		or not velocity.is_finite()
		or not external_acceleration.is_finite()
		or not is_finite(dt)
		or dt < 0.0
	):
		return {"error": "Free motion requires finite state and nonnegative time"}
	var next_position := position + velocity * dt + external_acceleration * (0.5 * dt * dt)
	var next_velocity := velocity + external_acceleration * dt
	if not next_position.is_finite() or not next_velocity.is_finite():
		return {"error": "Free motion calculation overflowed"}
	return {"position": next_position, "velocity": next_velocity}
