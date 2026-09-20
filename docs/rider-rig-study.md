# Rider glove and articulation study

The playable rider now uses articulated arms and original glove meshes. Human
close cameras show these limbs while excluding the helmet and torso. The policy
camera continues to exclude all rider geometry.

`godot/scripts/rider_glove.gd` is an original unbranded glove mesh study. A shaped
palm and cuff shell replaces the initial overlapping ellipsoid study. Four curved
fingers wrap around the existing 21 mm grip radius, with a separate thumb form,
knuckle protection and a restrained contrasting cuff panel. Each glove is one
ArrayMesh. Dimensions and anatomy remain artistic estimates. The material uses generic CC0 leather normal and roughness maps; it is not a
scan of the reference rider equipment or a finished tailored garment.
The cuff is open to receive a future sleeve.

The asset is connected to the playable rider and included in collision geometry.
Human close views show it; the versioned policy camera continues to exclude it.
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

The integrated rig uses a shared pure pose function for rendering, envelope
transforms and continuous sweep endpoints. Six explicit limb joints complement
the existing body, steering and wheel joints. Steering outside the certified
range is rejected before broadphase queries. The arm root must have an identity
transform and the same steering origin as the motorcycle.

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
reach P0 and first and second steering derivative bounds P1 and P2. The integrated
sweep composition with root plus lean angular displacement w and steering
change a is:

```
speed <= root_translation + w*P0 + a*P1
acceleration <= w*w*P0 + 2*w*a*P1 + a*a*P2
```

The gameplay sweep uses these bounds for articulated arm parts and rigid steering
bounds for gloves. The envelope version is `authored-convex-parts-v2` and the
sweep version is `articulated-conservative-sweep-v4`. The envelope audit passed
2,128 checks across 298 mesh parts. A real forearm fixture has clear endpoints
but intersects a small obstacle during steering; independent engine overlap
queries verify the sweep result. The sweep suite passed 115 checks, wall contact
passed 91, and camera checks passed with both human close views including limbs
and the agent view excluding them. The local cockpit capture is
`artifacts/articulated-game.png`. Native verification remains outstanding.

`test_rider_pose.gd` passed with zero failures across 2,002 sampled poses, checking
bone lengths, grip and cuff attachment, bone frames, continuity, interval
containment, invalid inputs and atomic visual updates. Finite differences also
stayed within the analytic derivative bounds; those samples are sanity checks,
not the proof of continuous coverage. Rendered neutral and full lock poses were
inspected in `artifacts/rider-arms-neutral.png` and `artifacts/rider-arms-lock.png`.
The sleeves and gloves remain visually provisional, without reference equipment
scans or validated anatomy. The test log is `artifacts/rider-pose-check.log`.

## Leather surface detail

The playable sleeves and gloves now use the CC0 Leather Red 02 normal and
roughness maps from Poly Haven. Only surface detail is used; original black
and contrasting vertex colors remain. Object space triplanar mapping keeps
the grain attached to each articulated part, at the published 0.6 metre tile
width. Both 1K maps have mipmaps for distant and oblique sampling. Normal
intensity and roughness are artistic adjustments. Geometry, collision and
control behavior are unchanged. The standalone arm preview now renders the
actual integrated arms rather than building a second rig. Native visual
acceptance and equipment tailoring remain outstanding.

Verification: publisher and repository checksums passed, the native macOS export
completed, and the rendered camera checks passed with zero failures. Neutral
and full steering lock previews were visually inspected, along with the game
cockpit under track lighting. Evidence: `artifacts/leather-arms-neutral.png`,
`artifacts/leather-arms-lock.png`, `artifacts/leather-game.png`, and
`artifacts/leather-camera.log`. Native foreground performance remains unverified.
