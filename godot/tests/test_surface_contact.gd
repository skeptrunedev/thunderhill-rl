extends SceneTree

const Contact = preload("res://scripts/surface_contact.gd")
const MASS := 277.0
const GRAVITY := Vector3(0.0, -9.81, 0.0)
var failures := 0


func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)


func near(actual: float, expected: float, message: String) -> void:
	check(absf(actual - expected) < 0.0001 * maxf(1.0, absf(expected)), message)


func _initialize() -> void:
	# Crest/dip of a circular profile at its horizontal tangent: k = +/-1/R.
	for curvature: float in [-0.01, 0.0, 0.01]:
		for speed: float in [0.0, 10.0, 20.0, 40.0]:
			var result := Contact.support(
				MASS,
				Vector3(speed, 0, 0),
				GRAVITY,
				Vector3.RIGHT,
				Vector3.BACK,
				Vector3(0, curvature, 0),
				Vector3.ZERO,
				Vector3.ZERO
			)
			check(not result.has("error"), "Valid crest/dip rejected")
			var expected := MASS * (9.81 + curvature * speed * speed)
			near(
				result.required_normal_force_n,
				expected,
				"Crest/dip support differs from Newton balance"
			)
			check(result.release == (expected <= 0), "Tensile support did not release")
			near(result.normal_force_n, maxf(0, expected), "Contact applied a pulling force")
	# Slope and cross slope: static support must be mg*cos(inclination).
	var du := Vector3(1.0, 0.3, 0.0)
	var dv := Vector3(0.0, -0.2, 1.0)
	var banked := Contact.support(
		MASS, du * 1.2 + dv * 0.4, GRAVITY, du, dv, Vector3.ZERO, Vector3.ZERO, Vector3.ZERO
	)
	near(
		banked.normal_force_n,
		MASS * 9.81 / sqrt(1.13),
		"Plane support differs from gravity projection"
	)
	# A paraboloid with mixed curvature. Change its coordinates without changing
	# physical position/velocity. The answer must be invariant to that choice.
	var velocity := Vector3(12, 0, -7)
	var expected_acceleration := 0.008 * 144 + 2 * 0.003 * 12 * -7 - 0.004 * 49
	for axes: Array in [
		[Vector2(1, 0), Vector2(0, 1)],
		[Vector2(2, 0.5), Vector2(-0.4, 0.8)],
		[Vector2(0.01, 0), Vector2(0, 100)]
	]:
		var u: Vector2 = axes[0]
		var v: Vector2 = axes[1]
		var result := Contact.support(
			MASS,
			velocity,
			GRAVITY,
			Vector3(u.x, 0, u.y),
			Vector3(v.x, 0, v.y),
			Vector3.UP * quadratic(u, u),
			Vector3.UP * quadratic(u, v),
			Vector3.UP * quadratic(v, v)
		)
		near(
			result.normal_acceleration_m_s2,
			expected_acceleration,
			"Support depends on surface parameterization"
		)
	# Impulse must preserve tangent velocity and dissipate the predicted energy.
	var normal := Vector3(0.2, 1, -0.3).normalized()
	var tangent := Vector3(7, 2, 4).slide(normal)
	for incoming: float in [-8.0, -1.0, 0.0, 3.0]:
		for restitution: float in [0.0, 0.35, 1.0]:
			var before := tangent + normal * incoming
			var result := Contact.impact(MASS, before, normal, restitution)
			var after: Vector3 = result.velocity
			check(
				after.slide(normal).distance_to(tangent) < 1e-5,
				"Normal impact altered tangent velocity"
			)
			near(
				after.dot(normal),
				-incoming * restitution if incoming < 0 else incoming,
				"Impact restitution incorrect"
			)
			var initial_energy := 0.5 * MASS * before.length_squared()
			var lost_energy := initial_energy - 0.5 * MASS * after.length_squared()
			# Vector3 uses single precision; near-elastic subtraction loses digits.
			check(
				absf(result.energy_dissipated_j - lost_energy) < 1e-6 * maxf(1.0, initial_energy),
				"Impact energy balance incorrect"
			)
			check(result.normal_impulse_ns >= 0.0, "Separating contact applied a pulling impulse")
	# Exact constant-gravity flight and landing velocity from energy conservation.
	var height := 5.0
	var duration := sqrt(2 * height / 9.81)
	var flight := Contact.free_motion(Vector3(0, height, 0), Vector3(10, 0, 0), GRAVITY, duration)
	near(flight.position.y, 0.0, "Ballistic landing height incorrect")
	near(flight.velocity.y, -sqrt(2 * 9.81 * height), "Ballistic landing speed incorrect")
	var first := Contact.free_motion(
		Vector3(0, height, 0), Vector3(10, 0, 0), GRAVITY, duration / 3
	)
	var second := Contact.free_motion(first.position, first.velocity, GRAVITY, duration * 2 / 3)
	check(
		second.position.distance_to(flight.position) < 1e-5,
		"Ballistic path depends on timestep partition"
	)
	check(
		second.velocity.distance_to(flight.velocity) < 1e-5,
		"Ballistic velocity depends on timestep partition"
	)
	check(
		(
			Contact
			. support(
				MASS,
				Vector3.UP,
				GRAVITY,
				Vector3.RIGHT,
				Vector3.BACK,
				Vector3.ZERO,
				Vector3.ZERO,
				Vector3.ZERO
			)
			. has("error")
		),
		"Support accepted a separating velocity"
	)
	check(
		(
			Contact
			. support(
				MASS,
				Vector3.ZERO,
				GRAVITY,
				Vector3.RIGHT,
				Vector3.RIGHT,
				Vector3.ZERO,
				Vector3.ZERO,
				Vector3.ZERO
			)
			. has("error")
		),
		"Singular surface accepted"
	)
	check(
		Contact.impact(MASS, Vector3.ZERO, Vector3.UP, 1.1).has("error"),
		"Impact accepted energy-generating restitution"
	)
	check(
		Contact.free_motion(Vector3.ZERO, Vector3.ZERO, GRAVITY, -1).has("error"),
		"Negative time accepted"
	)
	print("SURFACE_CONTACT_CHECK failures=", failures)
	quit(failures)


func quadratic(a: Vector2, b: Vector2) -> float:
	return 0.008 * a.x * b.x + 0.003 * (a.x * b.y + a.y * b.x) - 0.004 * a.y * b.y
