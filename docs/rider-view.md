# Human rider camera

20 September 2026. The human rider camera now follows the motorcycle's rendered pose directly. Previously it used the chase camera's world position smoothing; steady motion therefore caused the viewpoint to trail the motorcycle, rather than stay at the rider.

The eye anchor is inside the original helmet mesh at local coordinates (0, 1.51, -0.30) metres. It follows pitch and lean. Gaze follows the road tangent with a 0.25 radian downward framing angle and 22 percent roll response. Vertical field of view is 90 degrees; switching to chase restores 64 degrees and its existing position smoothing. These are provisional presentation choices, not measurements of the reference camera or human head stabilization. The helmet and torso remain excluded to prevent occlusion; articulated limbs use a separate visible layer.

The dedicated agent observation camera remains at its existing versioned pose and intrinsics. Human presentation settings do not silently change policy observations. No physics or control inputs changed.

`tests/test_rider_camera.gd` checks fixed eye attachment after large position changes, sloped road samples, mirrored lean and different frame intervals, plus visibility and field of view when switching modes. Actual Vulkan renders were inspected upright and at 0.4 radians lean. The dashboard remains visible near the bottom of the frame, with more road in view. Artifact paths are `artifacts/helmet-view-framing.png` and `artifacts/rider-camera-lean.png`.

Camera motion and viewing comfort still need human riding review. This correction does not address the provisional motorcycle mesh, provisional rider arm materials, or the remaining track and dynamics realism work.

The clean exported build `7667d8f45deb-9b0c94c52cdf` was installed at `~/Applications/ThunderhillReview/7667d8f/Thunderhill.app`. The M3 Pro Metal renderer produced a 1920 by 1200 cockpit capture that was retrieved and visually inspected. Evidence is in `artifacts/mac-rider-preview.png` and `artifacts/mac-rider-preview.log`. The rendered human input diagnostic also passed on the local Vulkan renderer with zero failures. Native sustained riding comfort remains a human review item.

## Onboard framing view

The human camera cycle now offers chase, rider and onboard views. Press C or use Change camera in the pause menu; the HUD labels the selected view. The third view attaches to an approximate local anchor `(0, 1.14, -0.30)` metres, uses a 74 degree vertical field of view and a 0.40 radian downward angle. It exposes the handlebar controls and instrument cluster for comparison with the supplied Ken Moto footage. This is an appearance study, not a measured camera mount, calibrated lens or replacement for the helmet eye. The reference also supports a dark handlebar finish, now applied to the original bar geometry.

The existing rider view retains its eye anchor and framing. Both human close views use the existing partial roll stabilization, hide the helmet and torso, and show articulated arms and gloves on a separate visibility layer. Rider material realism, precise cockpit shapes and camera comfort remain unfinished. The separate versioned agent observation camera is unchanged.

The updated camera attachment and mode switching checks passed with zero failures. A rendered keyboard diagnostic cycled all three views and passed with zero failures; local frame intervals were median 17.242 ms and p95 17.746 ms at 1920 by 1200 on the RTX 2080 Ti. The final onboard view was inspected in `artifacts/onboard-framing-preview.png`. These checks establish the local presentation and control path, not exact reference camera calibration or native Mac visual acceptance.

The reservoir cylinders now use an amber finish with a visible fluid boundary,
guided by the supplied onboard frame at ten seconds. This is an opaque artistic
material approximation, not physical transmission, measured fluid capacity or
simulated slosh. Geometry, dimensions and collision remain unchanged. The local
Mobile renderer compiled the shader and the resulting cockpit capture was
visually inspected in `artifacts/reservoir-preview.png`. Native verification of
this material remains outstanding.

## Closer onboard framing

Comparison with the supplied video frame at ten seconds showed the previous
onboard anchor made the tank dominate and the display appear too small. The
optional onboard view now sits farther forward and slightly lower, with a
shared downward angle constant used by gameplay and the standalone preview.
The helmet and agent camera poses and intrinsics are unchanged. This is a
qualitative framing adjustment, not calibrated camera reconstruction.

The fluid pots now sit on raised supports, matching their elevated placement
relative to the handlebars in the footage. The 75 mm elevation change and mount
shape are artistic estimates, not dimensions inferred with measured confidence.
Both support meshes and the moved pots are included in the normal front joint
collision envelope. The instrument design, yoke, clamps and tank still need
shape and detail work to approach the reference quality.

The existing camera diagnostic now accepts `--screenshot-camera=2` alongside
`--screenshot=/absolute/path/image.png` to inspect the actual onboard camera
in the track scene at the existing leaned test pose, with the UI hidden.

Verification passed: 2,142 envelope checks, 115 continuous sweep checks, rendered
camera attachment and human controls, and the macOS export. Neutral, full steering
lock and leaned onboard captures were inspected in
`artifacts/onboard-raised-pots.png`, `artifacts/onboard-close-lock.png` and
`artifacts/onboard-close-lean.png`. These establish presentation and collision
integration, not measured mount dimensions or native viewing comfort.

## Clamp and yoke edge detail

The existing top yoke, risers and upper clamps now use a closed chamfered box
mesh within their previous outer dimensions. Bevel widths remain artistic
estimates. Flat faces, edge strips and triangular corner patches produce edge
highlights without changing the requested bounds. Upper clamps now share the
dark metal finish, with small dark hexagonal socket markings on their fasteners.
These markings depict socket appearance, not recessed mechanical geometry.

The mesh check covers closed manifold edges, clockwise winding, outward normals,
fixed extents and volume bounds. The actual motorcycle envelope includes the
new geometry, including socket markings. Development recording provenance also
includes the chamfer and rider geometry helper hashes; native packages retain
their existing complete content manifest.

Verification: the chamfer mesh test, 2,170 envelope checks, 115 sweep checks,
99 wall contact and recording checks, the rendered human control diagnostic
and macOS export pass. The neutral cockpit render was inspected in
`artifacts/chamfered-cockpit.png`. This improves original authored geometry
without claiming measured Ducati manufacturing dimensions.

## Molded reservoir silhouettes

The straight vessel cylinders and plain cap cylinders now use original revolved
profiles with rounded shoulder transitions, beveled lids and 24 molded grip
recesses per cap. These are visual estimates from the supplied footage. Vessel
height and maximum radius remain 45 mm and 28 mm; lid height and maximum radius
remain 8 mm and 30 mm. These values describe the authored assets, not Ducati
measurements. The existing opaque fluid shader remains an appearance estimate.

Closed ends have sharp normal boundaries and the side profiles use smooth
normals. The cockpit test checks finite unit normals, nondegenerate triangles,
clockwise winding, closed edges, positive volume and the previous radial and
height bounds. The same 304 collision components now contain 52,838 unique
input points. All 2,170 envelope checks and 115 sweep checks pass, as does the
rendered rider camera check and macOS export.

Neutral and leaned track captures were visually inspected in
`artifacts/reservoir-cockpit.png` and `artifacts/reservoir-on-track.png`.
This build has not yet been verified in a foreground native Mac session.

## Independent static reference pose controls

The existing lighting preview now accepts finite bounded `--lean-deg`,
`--view-roll-deg`, `--view-yaw-deg` and `--lateral-m` values. Lateral placement must
remain inside the selected track section. Camera roll preserves eye position
and gaze, then establishes optical roll relative to world up independently from
motorcycle lean. These are preview controls, not gameplay camera changes.
Metadata records the chosen pose and explicitly labels it an artistic study.

Frame 00:40 has an apparent image bank of roughly 12 to 17 degrees by visual
inspection, not a measurement of physical motorcycle lean or camera calibration.
The existing 0.22 camera follow factor produces a much smaller roll at the
previous 0.4 radian test lean. Static trials use a 14 degree image roll, with
negative sign for the reference's landscape descending toward the right.

`artifacts/turn2-inside-reference-pose/` uses rider camera 1 at station 950 m,
25 degrees of left lean, 4 m left lateral position and 8 degrees of gaze yaw.
`artifacts/turn2-deep-lean-reference-pose/` uses 45 degrees of left lean and
3.5 m lateral position. The first better matches the road edge entering the
bottom left; the second shifts the cockpit toward the lower right but worsens
road framing. Both retain the existing FOV. The low onboard view makes the
cockpit dominate and is unsuitable for this particular frame comparison.

Root and independent inspection find these views useful for coarse comparison,
not a registered 1:1 match. Road curvature in the image, distant terrain silhouette,
lens parameters, exposure, exact position and motion blur remain different.
Increasing lean to move the dashboard would confound bike pose with camera
placement, so no gameplay roll response or rider anchor is changed here.

## Filter the live instrument display at riding distance

The instrument source was already 1024 by 560 pixels. Its single bilinear
sample did not average the area covered by a screen pixel when the display was
minified, causing thin ticks and labels to fragment into bright isolated pixels.
The diagnostic readback of the live viewport image reports no mipmaps. The
correction averages a four by four grid over the projected UV derivatives in
the instrument shader. Samples are decoded to linear color before averaging.
The display remains live; no CPU image readback or additional viewport is added
to gameplay. Reflection, housing geometry and instrument readings are unchanged.

`artifacts/display-filter-baseline/` and `artifacts/display-filter-candidate/`
contain four matching camera variants. Root and independent inspection accepted
the candidate: small labels and ticks are more continuous without obvious blur
of the large timer or gear digits. Metadata confirms identical camera and
reservoir states across all four pairs. Small labels remain limited by their
screen size; stationary frames do not establish temporal stability. The shader
now defaults to the filter enabled. The cockpit preview follows that default;
`--display-unfiltered` selects the baseline and `--display-filtered` explicitly
selects the candidate. Contradictory flags are rejected. Readback is confined to
the diagnostic preview, which records filter state and shader hash.

`artifacts/filtered-display-rider-view/production.png` verifies the adopted
shader from the existing artistic station 950 rider pose. Motorcycle readouts,
geometry checks and rider visibility passed. Human controls passed with zero
failures; local Linux timing was median 17.326 ms and p95 19.231 ms over 267
frames. This is not an isolated GPU cost measurement or native Mac benchmark.
Formatting and whitespace checks passed. Tailscale reports the intended MacBook
offline, last seen 2026-09-21 09:00 UTC. The complete cockpit still needs further
shape and surface detail before it resembles the footage closely.
Mac export passed as `f8a72cc2c25f-1478320bf958`; native execution is unverified.

## Molded reservoir cap crown

The original cap had a flat featureless top disc. The supplied frame at 00:40
shows a dark molded top with rim detail and embossed markings. The model now
has a shallow recessed crown and concentric perimeter lip within the existing
cap bounds. Profile dimensions are original visual estimates, not manufacturer
measurements. The existing grip recesses and reservoir dimensions are retained.

`artifacts/reservoir-crown-detail/` exposed an incorrect bowl like highlight:
smooth normals from the rim propagated across broad planar annuli. A central
bump also looked unlike the source. Each cap profile band now has its own
smoothing group, preserving circumferential smoothness while keeping planar
rings planar. The central bump was removed. `artifacts/reservoir-crown-final/`
records that geometry correction. The cap uses a separate instance of the
existing polymer finish with roughness 0.72 instead of 0.52 to reduce broad glare.
Color, metalness and microtexture remain unchanged; roughness is an appearance
estimate, not measured material data.

Root and independent review accepted `artifacts/reservoir-crown-matte/` as a
modest improvement over `artifacts/display-filter-candidate/`. Raised views now
read as flat molded caps with shallow perimeter lips, without the speaker effect
or the earlier bright pale disc. The onboard view changes less because the caps
are nearly edge on. Embossed lettering and smaller surface irregularities remain
missing, so the asset is not finished or a photographic match.

The motorcycle test checks finite normals, closed edges, winding, positive
volume and existing cap bounds. It now additionally checks that horizontal cap
triangles have vertical normals, preventing bevel smoothing from inventing
curvature. Those checks, instrument readings and rider visibility passed.
Human controls passed with zero failures. Local Linux timing was median 17.244 ms
and p95 23.022 ms over 265 frames, not a native Mac benchmark. Formatting and
whitespace checks passed.
Mac export passed as `0dd8be9062df-6ab845257227`; native execution is unverified.
