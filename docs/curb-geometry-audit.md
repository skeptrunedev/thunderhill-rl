# Curb placement and contact audit

The current curvature rule is provisional and unsuitable as a reconstruction of
observed curb locations. On the current track it produces 112 disconnected runs,
many only one or two road segments long. This fragmentation follows local
curvature variation, not reviewed paint or construction boundaries.

`uv run tools/audit_curbs.py` overlays the existing 0.9 m wide footprint on the
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
dry pavement estimates, as documented in `dynamics-implementation.md`. Reviewed station intervals
must replace the curvature rule before claiming an accurate reconstruction.

Evidence is in `artifacts/curb-audit/report.json`, `overview.png`, the numbered
paired crops, `contact-before.json`, and
`artifacts/curb-track-contact-check.log`.
