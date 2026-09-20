# Rider glove and articulation study

The original rider uses capsules and ellipsoids, with all limbs rigidly attached
to the body. Close human cameras and the policy camera exclude the entire rider.
Showing that rider in the cockpit would expose hands that do not follow steering.

`godot/scripts/rider_glove.gd` is an original unbranded glove mesh study. A shaped
palm and cuff shell replaces the initial overlapping ellipsoid study. Four curved
fingers wrap around the existing 21 mm grip radius, with a separate thumb form,
knuckle protection and a restrained contrasting cuff panel. Each glove is one
ArrayMesh. Dimensions and anatomy remain artistic estimates. The material is a
plain leather appearance approximation; it is not a scanned or finished asset.
The cuff is open to receive a future sleeve.

The asset is deliberately not connected to the playable rider yet. It does not
change current collision geometry, human camera output or agent observations.
The final local preview is `artifacts/glove-study-smooth.png`. Run:

```
godot --path godot --script res://tests/preview_glove.gd -- --screenshot=/absolute/path/gloves.png
godot --headless --path godot --script res://tests/preview_glove.gd
```

Both mirrored meshes passed finite vertex, normalized normal and hull volume
checks. The rendered preview was visually inspected. These checks do not certify
anatomical accuracy, material realism, grip contact or completed articulation.

## Required shared arm pose model

The existing right arm coordinates in BikeVisual give an upper arm length of
0.25164 m and forearm length of 0.30798 m, totaling 0.55962 m. Rotating its current
wrist about the front steering origin by the supported 0.50 rad limit puts it
0.66149 m from the fixed shoulder. The left side mirrors this. These figures
are calculations from the current artwork, not measured human dimensions.
The current skeleton therefore cannot simply receive fixed shoulder two bone
IK and remain attached at full steering lock.

Next, establish wrist targets at the new cuff, arm dimensions and shoulder motion
with positive reach and bend plane margins over the full steering range. Then
share one pure pose function between rendering, collision envelope transforms and
continuous sweep endpoint transforms. Existing envelope joints cover only body,
front, front wheel and rear wheel. Gloves can use the front joint, but animated
arms require explicit continuous speed and acceleration bounds. Clamping an
unreachable target or using only endpoint rotations would not preserve the
existing continuous contact guarantees.

Verification must include both steering limits, intermediate steering, mirrored
lean, grip attachment, continuity, construction from already moved poses, visual
vertices versus envelope transforms and obstacles hit only between endpoints.
The current sweep accepts arbitrary finite steering values, so any narrower
valid arm domain must be checked before broadphase rejection. Camera visibility
also needs a dedicated limb layer; placing hands under the front node without
mask changes would silently alter policy RGB observations.
