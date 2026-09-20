# Continuous motorcycle contact with the pit divider

The game now queries the authored motorcycle and rider envelope against the solid historical pit divider on every successful simulation tick. Human input and agent actions use the same integration callback. A contact terminates the episode at the preceding clear pose and records the event. It does not simulate an impact impulse, deformation, rider separation, sliding or injury.

## Motion and geometric bound

`bike_sweep.gd` defines the path between two simulation poses explicitly. Root translation is linear. Root rotation follows the shortest quaternion arc at constant angular speed. Lean, steering and wheel angles are linear, with wheel revolutions retained rather than wrapped away. The path approximates motion within the fixed simulation tick; it is not a continuous solution of the motorcycle dynamics.

The quaternion path is constructed from a relative axis and angle. Godot's generic slerp has a linear interpolation branch for nearby quaternions, so relying on that method would not implement the exact constant angular speed assumed by the bound. See the [engine implementation](https://raw.githubusercontent.com/godotengine/godot/master/core/math/quaternion.cpp).

For each component, fixed local transforms are applied to its vertices first. Each translated joint term then has a rotation speed bound equal to the sum of the enclosing angular speeds. A term with radius `r` and angular speed bound `w` contributes `r*w` to the speed bound and `r*w*w` to the acceleration bound. Root linear displacement contributes to speed only. Adding the contributions gives `L` and `M` over a unit tick interval.

Over a fraction interval of length `h`, a vertex remains within `M*h*h/8` of its endpoint chord. The convex hull of both endpoint vertex sets, expanded by that amount in each coordinate, therefore encloses the component's entire path. Empty overlap queries reject complete intervals. Nonempty intervals are subdivided chronologically. At a terminal interval the enclosure is within `L*h + sqrt(3)*padding` of the shape at its start.

The declared spatial tolerance is 2 millimetres. Padding includes a separate 0.5 millimetre numerical allowance. Query vertices are centered locally. These are numerical engineering tolerances, not proof that all physics backends have exact geometric predicates. Root coordinates beyond 4096 metres on any axis are rejected explicitly; the current track's largest absolute sample coordinate is approximately 783.406 metres. Geometry source uncertainty and conservative convex filling of individual mesh concavities are additional limitations, separate from the sweep tolerance.

A box enclosing the complete root segment and maximum articulated reach rejects distant obstacles cheaply. Component endpoint boxes reject individual parts before expensive hull queries. Extremely large angular intervals subdivide before constructing oversized engine hulls. A motion whose precision cannot be achieved within 32 subdivisions fails explicitly, as does exhaustion of the 4096 interval budget. Query errors invalidate the rollout and restore the complete simulator state through the existing atomic step mechanism.

## Contact result and recording

`possible_contact` is deliberately not an exact impact claim. It contains `safe_fraction`, `interval_end_fraction`, component identity and spatial uncertainty. The interval end is not guaranteed to overlap. For `initial_overlap`, both fractions are zero and there is no preceding clear pose.

Gameplay freezes position and the complete root orientation at the selected pose, including the wheel angle, and zeros longitudinal and roll velocity. Other force and drivetrain fields remain the last integrated candidate values; they are not an impact solution. The fixed tick and elapsed time still advance by one complete tick. The fractions locate possible contact within that tick. The lap becomes invalid and the episode terminates. Subsequent human physics and agent actions cannot advance the terminal state until reset.

`collision_contact` is included in telemetry and state playback. A single `obstacle_contact` event records the unconstrained candidate state before freezing, component ID, obstacle kind, fractions and pose. Process local collider IDs are removed. The manifest records the envelope and sweep versions, wall source hash, tolerance and terminal response assumption. Playback restores the stored root and wheel pose and verifies the wall hash for recordings that contain the new collision manifest. Older state recordings retain their existing playback compatibility.

## Verification and remaining work

`test_bike_sweep.gd` checks fast crossings, initial overlaps, masks, a 1 millimetre near miss, intermediate yaw, lean and steering impacts with clear endpoints, and a complete wheel revolution. Independent actual overlap queries audit the reported clear prefix and distance uncertainty. An extreme wheel rotation initially exposed an incorrect clear result from oversized engine queries; that case now returns an explicit precision error.

`test_wall_contact.gd` exercises the actual game with a perpendicular obstacle at 80 metres per second, comparing human and agent state, event counts, terminal freezing, reset, playback and failed query rollback. The historical wall fixture also passes at 80 metres per second, stopping on the front wheel component with a reported 1.496 millimetre uncertainty. The rendered test passed 92 checks with zero failures and the image was inspected (`artifacts/wall-contact-actual.png`, `artifacts/wall-contact-rendered.log`). The final headless run passed 91 checks (excluding screenshot saving), and measured the actual historical wall contact step at 322.752 ms. This confirms the stall also affects an ordinary wall crash, not just the wide helmet rotation fixture.

The full circuit diagnostic completed 40,383 ticks and all 32 ordered gates in 336.525 simulated seconds, with no offtrack ticks or crashes. The trace is `artifacts/qa/wall-contact-lap`. This is a privileged path following diagnostic, not training or evidence of physically accurate racing.

The short rendered human control check passed with zero failures at 1920 by 1200 on the local RTX 2080 Ti. It recorded a median frame interval of 17.330 ms and p95 of 18.114 ms. This does not establish Mac or whole circuit performance.

In an isolated local query diagnostic, clear broadphase rejection averaged 0.01138 ms. Ten authored bike steps parallel to a nearby wall had median 1.594 ms and maximum 1.764 ms. A complex authored helmet contact took 315.407 ms, which remains a material stall to optimize. Unbundled development recordings also hash the visual, envelope and sweep scripts, because edits to the artwork change the collision inputs. Native collision performance, physical impact dynamics and a polished crash presentation remain unfinished.

## Performance investigation

Optional `--profile-sweep` on the sweep and wall integration tests records vertex preparation, hull construction, engine query and initial overlap costs. The flag is diagnostic only and does not change the returned contact or recorded transition.

The baseline helmet fixture spent approximately 190 ms constructing hulls, 68 ms preparing 263,424 input vertices, and 41 ms querying the resulting hulls. This identifies repeated padded hull construction as the main target. A larger proxy is not automatically faster: an inscribed sphere based endpoint expansion reduced the helmet timing but increased the historical wall step to 394.804 ms. A cached fixed hull proxy measured 521.151 ms and a hybrid measured 380.167 ms. These variants generated extra candidate intervals and are not accepted performance improvements.

A separate checkout tested the engine's Jolt backend. The spherical query margin variant measured approximately 69 ms for the helmet and 54 ms for the historical wall, but failed the existing helmet uncertainty check. Both that variant and the original cube expansion emitted hull construction errors during the historical wall fixture. The subsequent null shape could produce an empty intersection result. Passing integration assertions alone therefore did not establish safe collision detection. This experiment is not grounds to switch the game's physics backend.

### Exact cube corner reduction

For a cube corner with signs `(sx, sy, sz)`, an endpoint vertex `p` need not contribute that corner when another endpoint vertex `q` is strictly farther in all three signed coordinates. In every support direction of that octant, `q` supplies at least the same support. Every removed corner has a strict dominance witness, and the finite dominance graph is acyclic. A retained maximal witness therefore remains. This removes redundant input points without replacing the authored envelope with a larger approximation. A small set of neighboring source indices and coordinate extremes supplies candidates; a missing witness keeps the corner.

Masks are computed on transformed world points before subtracting the interval center. Subtracting a common center and adding a common cube offset preserve coordinatewise ordering under floating point rounding, although strict ordering can become equality. This permits endpoint points and masks to be reused across subdivisions. Cache fraction keys must retain full script floating point precision; packing fractions into a single precision vector would alias deep subdivisions. The cache is cleared on each sweep.

The endpoint cache holds at most 131,072 points and 2,048 entries per environment. Point and mask arrays contribute at most approximately 2 MiB of cached payload, excluding dictionary overhead and transient hull construction. When full, the solver recomputes uncached endpoints. The cap does not reject a query or report clearance. The geometry audit checks 52,203 cases covering all 300 authored parts, structured and random support directions, translated coordinates, deep fraction keys and both cache limits.

An alternating comparison in the same process ran three historical wall contacts for each implementation. Baseline times were 335.455, 332.187 and 324.296 ms; the reduced implementation took 267.593, 253.611 and 250.875 ms. The median decreased from 332.187 to 253.611 ms (23.7 percent). All six contact records matched exactly. This comparison passed 142 checks and is recorded in `artifacts/sweep-reduction-wall-ab.log`. The remaining quarter second impact stall is unacceptable for final performance acceptance.

The final support audit passed 52,203 checks, the sweep suite passed 96 checks and the wall integration suite passed 91 checks. The rendered human control diagnostic also passed with zero failures, with median process frame interval 17.290 ms and p95 17.825 ms on the local RTX 2080 Ti. These measurements do not certify native Mac impact performance.
