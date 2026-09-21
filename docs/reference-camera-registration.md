# Turn 2 reference camera registration

The repeated station 1150 material preview is a reproducible game view, not a
registered match to video frame 40. Comparing colors and surface character is
useful, but it does not establish exact terrain or scenery placement from pixels.

## Recovered source

The full supplied video was recovered from
https://www.youtube.com/watch?v=yVjzZqYbKuM using the H.264 video representation
(format 136). Local path: `artifacts/reference/ken-moto/onboard-avc.mp4`.
SHA256: `114e2abfce6d567ccd378b1f9c81d6dbbc7d37b164ccdb80a16db06841cd214b`.
Timing is 30 FPS and 320.733333 seconds. Footage stays in ignored artifacts;
it is not redistributed as a game asset. The first recovered AV1 representation
could not be decoded by the local OpenCV build, so the H.264 representation was
acquired directly rather than substituting fabricated or interpolated frames.

`tools/video_contact_sheet.py` now accepts finite bounded `--start` and `--end`
times and records the source hash and requested interval. The two second sample
sequence from 28 to 52 seconds is in `artifacts/reference/ken-moto-turn2-avc`.
The last sample is 50 seconds because the end is exclusive. This replaces the
previous ten second spacing for this investigation.

## Evidence and limits

At 30 seconds the view is nearly upright and the dashboard centered. Through
36 to 44 seconds the dashboard moves strongly toward and beyond the right edge.
At 48 to 50 seconds it returns toward the center as the motorcycle straightens.
This supports changing camera orientation relative to the motorcycle, rather
than a fixed cockpit camera. The exact mounting and head translation are unknown.
A rider shadow becomes visible in the closer sequence around 42 seconds; it is
a potential heading constraint, not proof of station without solar calibration.

The original preview at station 1150 uses yaw 8 degrees and bike lean minus 25
degrees. Separate tests at stations 1065 and 1150 used yaw 30 degrees and optical
roll minus 12 degrees. Those tests shift the cockpit toward the reference's side
of the image, but also move the road bend and do not solve the complete framing.
They are not accepted camera calibration and do not change playable defaults.
Evidence: `artifacts/reference-headturn-1065`, `reference-headturn-1150`, and
`reference-headturn-comparison`.

An orchard preview at station 1065 also remains mostly terrain occluded:
2050 of 2051 projected treetop samples are blocked in the approximate audit.
This supplements the station 1150 result; it does not identify the dark band in
the footage. Source: `artifacts/reference-alignment-1065/study.json`.

Next registration work should constrain upright camera framing first, then
compare a sequence of nearby stations with varying gaze and bank. Use pavement
boundaries and identifiable stationary landmarks together, not dashboard position
alone. Do not modify surveyed terrain or move orchard footprints to force one
unregistered screenshot to match.

## Verification

The dense extraction produced twelve frames with source frame indices 840 to
1500. A sample immediately before the end of the video decoded successfully,
with frame rounding bounded to the final valid frame. Negative start, inverted
range, nonfinite interval and end beyond duration each returned a usage error
without creating output. Ruff checks passed. The two new game camera poses
compiled and rendered with Vulkan Mobile. No runtime game behavior changed.

Independent sequence review adds a conditional turn phase constraint: both yaw
30 game tests already show substantial forward shadows, whereas the reference's
shadow enters only after frame 40. If the solar direction is approximately right,
frame 40 is earlier in heading progression than those tests, especially station
1150. The background includes both isolated dark roadside objects and a separate
continuous green vegetation band; classifying the entire band as a barrier would
be unsupported. Exact station and orchard identity remain unresolved.

### Solar direction correction

The conditional shadow timing inference above used an inherited light azimuth
of 145.81 degrees. Review of straight headings and reference frames 20, 30 and
32 supports an easterly working estimate of 110 degrees, now adopted. Earlier
shadow onset may therefore reflect the lighting error instead of later track
position. Do not treat the previous conditional station ordering as established.
See `docs/sky-lighting.md` for the controlled comparison and elevation limits.
