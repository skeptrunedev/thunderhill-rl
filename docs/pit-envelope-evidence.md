# Historical pit straight envelope

`uv run tools/build_pit_envelope.py` writes an experimental candidate envelope and diagnostic profiles under `artifacts/road-surface`. It never changes game data. Run `measure_pit_divider.py` first and provide the reviewed `pit-east-edge-measurements.json` artifact. Every input file, baseline track, road frame, original lidar tile manifest and builder is identified in the report.

The root cause of the current pit mismatch is the provisional width and center interpolation between eight aerial sections. The existing nominal edges are approximately minus 6 and plus 6 meters from the baseline. Registered aerial profiles put the east pavement transition at plus 8.025 to plus 9.075 meters. Raised lidar returns place the opposite structure face around minus 2.5 meters through most of the straight. These are corrections to both center and width, not just a narrower road.

The candidate builder uses the closest occupied raised lateral band to the pavement interior. The nominal face is the 95th lateral percentile of returns between 0.35 and 1.5 meters above a local interior pavement plane. Parallel calculations at lower height thresholds 0.15 and 0.25 meters expose threshold sensitivity. The face is not a surveyed wall base, and its spread does not capture absolute positioning error. In particular, short strips near the structure endpoints can have only two returns in the nearest band. Those candidates require review before adoption.

The 19 combined anchors cover baseline stations 50 through 410 meters. Candidate width ranges from 10.633 to 12.281 meters. The east uncertainty envelope includes threshold variation and one 0.6 meter image pixel on either side. It is not a statistical confidence interval, and georeferencing error remains unquantified.

Endpoint profiles are sampled every meter at stations 35 through 60 and 400 through 450. With a one meter profile half length, the first nominal raised candidate appears at station 43, after none at 42. The last is at 429, followed by no nominal raised candidate at 430. Lower height thresholds still detect a small band at 430. These bracket the observed raised structure, not a legal racing surface boundary. The current evidence does not establish how the racing pavement envelope continues across the adjacent pit entry or exit apron. The artifact therefore deliberately supplies no extrapolated joins. Inspect registered aerial imagery at both ends before adopting continuous geometry.

For integration, use the supplied explicit edge positions in local x,z coordinates, or project them into the baseline sample frames. The artifact station is the existing derived road station, not the original OSM `source_s` used by `reviewed_widths()`. Passing these stations directly into that function would introduce a coordinate error. Recompute center, tangents, width, station, surface mask and all dependent meshes together after adoption. The smooth experimental road frame is also slightly different from the renderer's piecewise baseline, which is why explicit geometric endpoints are retained.

Verification performed: both supplied measurement reports agree with the current track and road hashes; actual generated plot was inspected against the registered aerial overlay; dense endpoint counts and all combined anchor widths were examined. Formatting and lint pass. This is evidence preparation, not a claim that a physically exact wall, final track limits or runtime correction have been completed.

## Endpoint source review

The builder now produces `pit-endpoint-0-90.png` and `pit-endpoint-390-480.png`, each retaining an unobscured source panel alongside candidate overlays. The exact pinned NADCON grid pipeline registers lidar coordinates into aerial coordinates. It validates the source image hash and extent. Raw lidar archive columns must exactly match `x,z,height_local,classification,source_id`, with finite five column data and the original observation count.

The source image visibly ends the raised divider near station 429. A thin pale longitudinal marking continues through the apron toward station 480. Brightness contrast profiles, taking the median of three along profiles and comparing with flanks 1.2 meters away, put most candidate stripe centers between minus 2.45 and minus 2.9 meters; the local bend at stations 445 and 450 gives minus 3.8 and minus 3.35 meters. Contrast across stations 435 through 480 ranges from 0.063 to 0.183 on normalized RGB. Green crosses in the overlay mark these optical hypotheses. They are broadly aligned with the visible stripe, including its bend, rather than an arbitrary fade toward the old edge.

Before the divider, the faint marking is substantially less consistent. Station 5 selects a different feature at minus 5.9 meters with contrast only 0.012. Stations 10, 20, 25 and 35 have contrasts below 0.031. Stations 0, 15 and 30 produce stronger candidates near minus 3 meters, consistent with a possibly interrupted marking. No continuous visible boundary can be established over the whole 0 through 43 meter span from these pixels alone. All RGB profiles and peak contrasts are preserved, with a minimum 0.6 meter pixel interval and no claim of total positional accuracy.

These observations support a plausible stripe continuation after the wall, but do not establish whether the marking defines the legal main track limit or a pit blend line. The artifact keeps stripe hypotheses separate from adopted anchors. Use video or a track boundary reference to resolve that semantic distinction and the poorly observed northern apron before generating one continuous racing boundary. A rendered apron may remain fully paved while racing limits and the physical divider are represented separately.

## Reproducing the adopted interpretation

The historical reconstruction now has 42 adopted anchors in
`data/reference/pit-envelope.json`, with wrapped coverage from baseline station
4500 through station 700. The additional joins are explicit manual interpretation
of registered historical imagery, with at least 1.2 meters of interpretive
uncertainty. The western paved apron remains separately described; these anchors
are not a claim about surveyed wall bases or legal race limits.

After applying the correction, the live track is no longer the evidence baseline.
Recover the exact original baseline from git before reproducing the adoption or
its diagnostic panels:

```sh
git show e8a4056:godot/data/track.json > artifacts/road-surface/pit-baseline-track.json
uv run tools/adopt_pit_envelope.py --output artifacts/road-surface/pit-envelope-reproduced.json
uv run tools/review_pit_joins.py
```

Both tools default to that preserved baseline and accept `--baseline-track` for
another location. They verify its SHA against candidate evidence and the adopted
manifest, and verify the original road frame. Passing the corrected runtime
track fails instead of silently reinterpreting the historical station coordinates.
The diagnostic tool also verifies aerial image and datum grid hashes.

The adoption command above writes a review artifact without replacing the tracked
manifest or runtime data. Its anchors must be compared with the existing manifest
before adoption. A source only change to the builder updates
`adoption_builder_sha256`, so its resulting manifest SHA changes even when every
geometry field remains identical. Coordinate publication with the track builder
because runtime metadata pins the adopted manifest hash.
