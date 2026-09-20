# Authored motorcycle collision envelope

`godot/scripts/bike_collision_envelope.gd` snapshots a constructed `BikeVisual` into convex parts. Each mesh contributes one part. The rider and instrument meshes remain included when hidden by the cockpit camera or instrument state. Labels do not contribute geometry. The model identifier is `authored-convex-parts-v1`.

This is a collision approximation of the original artwork, not Ducati CAD or measured rider geometry. Each hull fills that mesh's concavities, including wheel centres, spring coils and body panel recesses. Separate meshes remain separate parts so the entire motorcycle is not enclosed in one large convex solid.

The shape inputs preserve every distinct authored vertex. The initial use of `Mesh.create_convex_shape(true, false)` failed strict support comparisons because cleaning contracted some meshes, including approximately 0.084 millimetres on a tire. Construction now removes only exact duplicate vertices. Input preservation is not a claim of exact numerical collision boundaries inside every physics backend.

`build` returns an error and leaves no components when geometry is missing, nonfinite, planar or collapsed. It rejects unsupported root offsets, joint scales and independent world transforms. Child index paths identify parts consistently across scene instances; these identifiers are tied to the authored hierarchy and source revision, not permanent semantic part names.

`transforms` calculates each part's pose from the supplied world root transform and explicit lean, steering and wheel angles. It does not read the last rendered pose. Angles follow `BikeVisual.update_pose`, so the caller must apply the simulation's sign convention. Front steering and both wheel rotations use their actual authored pivot positions. The current artwork still rotates steering about a vertical axis and has a fixed rider pose; this utility does not improve those physical assumptions.

`overlaps` uses the actual physics space and an explicit collision mask. It reports at most one obstacle per component, with the component identifier and joint. It provides discrete overlap existence, not a full contact manifold, contact normal, continuous sweep or impact response. It is not yet connected to gameplay termination. The next integration must cover translation and rotation between physics steps, initially overlapping poses, a last certified clear pose, human and agent parity, and recorded crash events. Simply calling this endpoint query from the simulation would miss fast crossings.

Run the actual engine checks with:

```sh
godot --headless --path godot --script tests/test_bike_envelope.gd
```

The checks compare every mesh and exact shape input vertex set, sampled convex support in 74 directions, four independent world and joint poses, IDs across regenerated scenes and atomic invalid input rejection. Physics fixtures intersect a hidden helmet and a steered handlebar while root and wheel point queries remain clear. Mask exclusion and a separated obstacle are also checked. These tests do not certify continuous collision detection, post impact dynamics or Mac frame time.
