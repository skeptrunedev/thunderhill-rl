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
