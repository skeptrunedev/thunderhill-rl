# Curb placement and contact audit

The original curvature rule was provisional and unsuitable as a reconstruction of
observed curb locations. On the track baseline it produced 112 disconnected runs,
many only one or two road segments long. This fragmentation follows local
curvature variation, not reviewed paint or construction boundaries.

`uv run tools/audit_curbs.py` now overlays the explicit placement manifest and its
existing 0.9 m wide footprint on the
pinned 2022 NAIP image. It reproduces the horizontal centered tangent frame from
`track.gd`, then applies the verified EPSG:6339 to EPSG:26910 datum operation.
The output report records source hashes, datum grid hashes, station bounds and
segment indices. Each run has an unmarked aerial crop beside its overlay.
The source resolution is 0.6 m. Enlarged pixels do not improve that accuracy;
edge paint and raised curbs can be ambiguous, and no height is inferred.

The overview and crops C49, C54 and C91 were inspected. C54 overlaps a visibly
striped area near the Cyclone; this does not establish its precise endpoints,
width or profile. C49 and C91 do not provide sufficiently clear evidence to
accept all of the provisional footprint as a raised curb. These are audit
observations, not replacement placement data. The remaining locations require
review against aerial imagery and onboard references.

## Shared rendering and contact geometry

The previous contact calculation used a nearest segment plane and radial offset,
while rendering used centered tangent frames at both endpoints. The baseline
sampled four interior points on each of 594 rendered triangles. Maximum height
disagreement was 0.074554 m, with a 95th percentile of 0.017738 m. Finite
difference normals could also cross curb boundaries.

`curb_surface.gd` now supplies the same triangle stream to rendering and contact.
It uses exact footprint tests, barycentric heights and face normals, with the
highest triangle winning overlaps. Invalid quads fail initialization. Where
terrain covers a curb, contact follows that higher visible terrain surface.
`sample_world` retains exposed curb identity and triangle ID. Recording
provenance now includes both the track script and the curb geometry helper.

The integrated test independently reads rendered mesh arrays and samples all
594 triangles at four interior points. All 2,376 samples pass, including 2,324
exposed curb points and 52 covered by terrain. Maximum height error is
0.000004564 m; maximum normal vector error is 0.000000180. Ground expectations
use the existing offroad sampler, whose mesh agreement is tested separately.
The standalone sampler suite passes 164 checks covering boundaries, overlap,
invalid input, winding, UVs and reset behavior.

Placement, width and profile remain provisional. This change preserves the
existing visible curb geometry and does not certify it against surveyed curbs.
Motorcycle friction now selects separate configurable curb coefficients for
exposed curbs, independently of `on_track`. Those coefficients remain uncalibrated
dry pavement estimates, as documented in `dynamics-implementation.md`. Reviewed placement across the remaining circuit is still required before claiming
an accurate reconstruction.

Evidence is in `artifacts/curb-audit/report.json`, `overview.png`, the numbered
paired crops, `contact-before.json`, and
`artifacts/curb-track-contact-check.log`.

## Turn 2 reference correction

Root and independent inspection of the pinned aerial crop
`[1100,1640,1720,2130]` and onboard frame 00:40 show a continuous white edge line
and earth shoulder on the inside of Turn 2, without the game's scattered blue
and white fragments. `data/reference/turn2-curb-review.json` records the source
hashes, observations and reviewed removals. C08 through C30 are removed: 23 runs,
36 road segments, or 72 rendered triangles. This is a correction to visible
construction, not evidence about buried concrete.

`godot/data/curb-placement.json` replaces runtime curvature driven spawning.
Its remaining 89 runs preserve the original 261 segment indices, sides and
profile exactly. All those retained runs are explicitly marked unreviewed and
provisional. The loader rejects mismatched track hashes, duplicate identifiers,
overlapping placements, invalid indices and nonfinite profile values. Rendering
and contact continue to share the same triangle stream. Development recordings
now include the placement data and helper hashes; both scenery bakes pin them.
The audit CLI reads this same manifest, writes a separate output directory and
refuses to overwrite an existing report.

The outside exit curb is visible in the aerial and frame 00:50, around stations
1268 to 1317 m. Root independently verified the pixel to track projection.
Its historical center lies approximately 4.7 to 5.3 m right of the centerline,
where the current road edge is 6 m right. It has not been added at an incorrect
road edge. This width alignment conflict remains unresolved. Neither the curb
height nor its width can be established reliably from 0.6 m pixels.

The placement test verifies every segment and side against the preserved baseline
with only the reviewed inner arc removed. Integrated contact checks pass for
54,520 pavement interior samples, 35,156 pavement boundary samples and 2,088
remaining curb samples. Maximum curb height discrepancy is 0.000003709 m.
All 36 removed segment footprints return no curb geometry or curb contact.
These are internal rendering/contact agreement checks, not survey accuracy.

Both scenery bakes and the rendered rider camera attachment check pass.
Root inspected `artifacts/turn2-curb-corrected/existing.png`; the unsupported
fragments are absent. `artifacts/turn2-curb-comparison/` compares it with the
previous view from the same station and camera. The track's remaining curb
inventory and overall visual fidelity are unfinished.

Human controls pass with zero failures. Local Linux timing was median 17.305 ms
and p95 19.698 ms over 264 frames, not native Mac performance certification.
Mac export passed as `18d1c645ab4f-524bc1616c78`. Tailscale reports the MacBook
offline; native execution of this build remains unverified.
