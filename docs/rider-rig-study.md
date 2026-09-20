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

## Shared pose prototype

`rider_pose.gd` now supplies the glove, upper arm and forearm transforms used by
`rider_arm_visual.gd`. The new cuff center is attached to a grip basis derived
from the existing inner and outer grip endpoints. The study uses fixed shoulders
and equal 0.33 m bones with an outward, downward and rearward elbow pole. These
are original artistic proportions, not measured anatomy. Invalid or unreachable
poses are rejected; no clamp or stretch is used. Visual pose application validates
both sides before changing either.

The equal length construction uses `E = S + D/2 + h*n`, where D points from
shoulder to wrist, `h = sqrt(L*L - |D|*|D|/4)`, and n is the normalized elbow pole
projected perpendicular to D. The bone frame uses the continuous bend plane
normal rather than switching reference axes during animation.

`domain_bounds` covers the steering interval with cells at most 0.01 rad wide.
For wrist steering radius R and cell half width delta, the circular trajectory
lies within `2*R*sin(delta/2)` of its center sample. This bounds shoulder distance
and pole cross product throughout each cell. Local coordinates are limited to
2 m, with explicit 0.1 mm numerical broadening. The current full domain encloses
shoulder distance between 0.299975 and 0.597738 m, elbow height above 0.139919 m,
and normalized pole separation above 0.935421. These margins avoid folded,
fully extended and undefined bend plane configurations across the supported range.

`point_motion_bounds` applies normalization and product rule bounds to the
elbow and bone frames. For a local mesh point radius r it returns whole point
reach P0 and first and second steering derivative bounds P1 and P2. Future
sweep composition with root plus lean angular displacement w and steering
change a is:

```
speed <= root_translation + w*P0 + a*P1
acceleration <= w*w*P0 + 2*w*a*P1 + a*a*P2
```

This motion bound implementation is not wired into the current gameplay sweep.
Integration still requires new joint identification, independent envelope poses,
whole bike broadphase reach, domain rejection before broadphase, and actual swept
arm obstacle tests. The current playable bike still uses its original rider.

`test_rider_pose.gd` passed with zero failures across 2,002 sampled poses, checking
bone lengths, grip and cuff attachment, bone frames, continuity, interval
containment, invalid inputs and atomic visual updates. Finite differences also
stayed within the analytic derivative bounds; those samples are sanity checks,
not the proof of continuous coverage. Rendered neutral and full lock poses were
inspected in `artifacts/rider-arms-neutral.png` and `artifacts/rider-arms-lock.png`.
The sleeves and gloves remain visually provisional, without scanned leather or
validated anatomy. The test log is `artifacts/rider-pose-check.log`.
