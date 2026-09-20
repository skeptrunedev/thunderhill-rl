# Smooth surface and free contact work

The playable model still uses `reduced-order-combined-assist-v5`. This work prepares its replacement; it does not silently change the reviewed game or claim to provide suspension.

## Why this is a coordinated change

The existing simulator pins position to ground height at every step and uses the normal component of gravity for total tire load. On a curved surface, the acceleration needed to follow the surface also contributes. A crest can require zero support, at which point a unilateral contact must release. Adding a curvature force while retaining the ground pin would still give the wrong motion.

The current road samples are joined with straight segments. Their tangent changes at segment boundaries. Road contact and rendered edges also use different tangent constructions. Differentiating those pieces does not supply a smooth physical curvature. The new road evaluator must replace projection, height, normal and mesh sampling together, including the boundary shared with offroad terrain.

Fork and Borrelli's [A General 3D Road Model for Motorcycle Racing](https://arxiv.org/html/2406.01726v1) motivates treating nonplanar surface geometry explicitly. Its constrained motorcycle model makes simplifying assumptions about the camber axis, pitch and contact. It is a reference for differential geometry, not a flight or suspension implementation to copy without those assumptions.

## Contact mechanics primitive

`godot/scripts/surface_contact.gd` provides stateless point contact calculations. For a smooth parametric surface R(u,v), the tangent metric solves for coordinate rates from world velocity. The second surface derivatives give the normal acceleration:

```
a_n = n dot (R_uu * u_dot^2 + 2 * R_uv * u_dot * v_dot + R_vv * v_dot^2)
N_required = mass * (a_n - external_acceleration dot n)
```

A nonpositive required support reports release and never applies a pulling force. Support requires tangent velocity at a contact event. Flight and impact must be resolved before using this constrained calculation. The primitive also provides exact constant acceleration free motion and a normal impact impulse with caller supplied restitution between zero and one. The impulse preserves tangent velocity and reports dissipated energy. No Ducati restitution value is assumed.

Analytical tests cover circular crest and dip tangencies, bank and grade, a paraboloid with mixed curvature under nonorthogonal and rescaled coordinates, contact release, impulse momentum and energy, constant gravity flight and timestep partition invariance. Degenerate surfaces and invalid inputs fail explicitly. Single precision Vector3 impact energy comparisons use a tolerance relative to initial energy to account for cancellation in nearly elastic cases.

## Integration boundary

This primitive has no collision event search, suspension travel, axle contact geometry, tire slip or chassis inertia. Its point must be the point whose acceleration is being balanced. Road reference curvature cannot be called COM acceleration when the COM is offset, leans or moves with suspension. The eventual motorcycle integration must account for those offsets and motions, then allocate axle reactions consistently.

Next steps are to validate smooth surface interpolation and inversion, generate the visible road and terrain cutout from the same surface, implement release and landing event detection, and remove unconditional ground snapping. Version the physics and geometry and regenerate reference laps only after those pieces pass analytical and whole circuit tests. The current Mac review build remains available while this work proceeds. No training has begun.

## Experimental road evaluator

`tools/build_road_surface.py` creates periodic quintic centerline polynomials and periodic cubic bank slope polynomials from the existing observations. `road_surface.gd` evaluates position and its first and second surface derivatives from those same coefficients. Quintic centerline continuity is needed because the lateral direction is derived from its tangent, and the second lateral direction derivative contains the centerline third derivative. The generated data stays in ignored `artifacts/road-surface/`; the live track does not load it.

The sampled historical audit evaluates nine stations per original interval and five lateral positions across its width, for 69,120 points. Maximum center displacement from the previous chords is 0.197754 m. The minimum sampled planar Jacobian is positive at 0.210854, but this is not an exhaustive proof of a valid ribbon. Centerline normal curvature ranges from approximately negative 0.009173 to positive 0.005506 per metre. At the positive lateral edge near stations 1820.392 and 1821.946, values reach positive 0.179396 and negative 0.235268 per metre. The corresponding station tangent magnitudes shrink to 0.274135 and 0.239310, magnifying derivatives when expressed per distance travelled. These inside Cyclone edge values require validation against source pavement geometry; no curvature clamp or arbitrary smoothing has been used to conceal them. Large values alone do not prove an impossible surface, nor do matching software calculations establish physical accuracy.

Reproduce the artifact and audit with `uv run tools/build_road_surface.py`. Run Python analytical and seam tests with `uv run --with numpy==2.4.3 --with scipy==1.17.1 python -m unittest discover -s tools -p test_road_surface.py`. Run the Godot point mechanics tests with `godot --headless --path godot --script res://tests/test_surface_contact.gd`. The road evaluator test requires the absolute paths of the generated `road-surface.json` and `road-surface.parity.json` after the `--` separator. It compares serialized Python derivatives and composed normal acceleration to Godot, including off center and mixed lateral motion cases. These are development checks, not a new playable dynamics release.

Verification completed with five Python analytical/continuity tests, all historical knot boundaries checked at left edge, center and right edge, and 38 Godot versus Python differential/contact cases including the actual extreme edge locations. Maximum vector discrepancy was 0.00003052 in the returned SI derivative vectors, consistent with Godot's single precision Vector3 representation. Point contact checks and full project import also pass. The extreme edge audit separates the normal curvature contributions: rapidly changing lateral frame dominates there, rather than bank slope interpolation alone. Raw logs are `artifacts/surface-contact-check.log`, `artifacts/contact-foundation-import.log`, `artifacts/road-surface-python-tests.log` and `artifacts/road-surface-godot-tests.log`.
