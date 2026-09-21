# Ground treatment experiments

Three independent treatments were rendered at the same Turn 2 apex pose and
lighting. Their original standalone variants remain developer experiments.
The regional generated variant described below is now the default game treatment.
The motorcycle asset work is deferred while the owner obtains its download.

| Treatment | Implementation | Visible result |
| --- | --- | --- |
| Generated dry field | Original larger straw and soil texture, standalone material without production aerial or procedural bands | Strongest directional improvement. Continuous dry straw coverage removes the smooth brown apron. Too uniformly dense, with insufficient larger disturbed patterns. |
| Scanned turf | Poly Haven Grass Ground color, OpenGL normal and roughness maps | Natural small scale variation but wrong green vegetation state. Reject this particular source as a Thunderhill match. |
| Physical fragments | 9000 deterministic clumps, actual terrain placement, fine curved strips and sparse stems with shadows | Adds local depth but does not resolve the dominant underlying material composition. First bright oversized version was visibly poor; refined fragments are less conspicuous. |

Root and independent review agree on this ranking. Reference camera, exposure
and motion blur differ; no exact material or straw scale match is established.
The generated treatment also replaces distant ground and access areas, so it
needs regional composition before becoming a production material.

## Evidence

* `artifacts/radical-ground-comparison/00.png`: production versus generated.
* `artifacts/radical-ground-comparison/01.png`: production versus scan.
* `artifacts/radical-ground-comparison/02.png`: production versus refined geometry.
* `artifacts/radical-ground-generated.mp4`: generated treatment on the same
  compatible recorded Turn 2 trajectory, 2699 transitions completed.
* `artifacts/radical-ground-generated-review`: twelve sampled video frames.

The 700 frame video is 1280 by 800 at 30 FPS, including a short stationary tail.
Fixed rate movie capture is not a live performance benchmark. Sampled frames
show continuous ground coverage through the bend, but do not establish fine
motion stability. Native Mac performance of these experiments is unverified.

## Reproduction

Use a real renderer and the pinned Godot executable from the project setup.
From the repository root, replace `godot` below with that executable:

```sh
godot --path godot --script res://tools/preview_lighting.gd -- \
  --production-only --camera=2 --station-m=1150 --lateral-m=4 \
  --lean-deg=-25 --view-yaw-deg=8 --ground-study=generated \
  --output-dir=/absolute/new/output
```

Choose `generated`, `scan`, `geometry`, or `production`. Metadata records the
chosen treatment and source parameters. Production terrain metadata is retained
separately for comparison rather than incorrectly reading it from the replacement.
The scan helper loads pinned source maps in the developer workspace; its loose
image loading has not been adapted or verified for an exported game.

```sh
godot --path godot --script res://tools/preview_ground_replay.gd \
  --write-movie /absolute/new/movie.avi --fixed-fps 30 --quit-after 700 -- \
  --ground-study=generated --replay=/absolute/compatible/replay.jsonl \
  --preview-camera=2
```

The replay study rejects agent and control test sessions. It changes appearance
only and does not produce a learned policy or establish physical accuracy.

## Sources and validation

The generated image, exact prompt and limitations are in
`assets/source/ground-studies`. Built in image generation produced 1254 square
pixels; color mipmaps use linear light filtering. The source is original artwork,
not extracted footage or a measured scan.

The scanned material is [Grass Ground](https://polyhaven.com/a/grass_ground)
by Charlotte Baglioni, [CC0](https://polyhaven.com/license). Exact download URLs,
publisher MD5 and acquired SHA256 hashes are in
`godot/assets/materials/scan_study/source.json`.

All three treatments compiled and rendered with Vulkan Mobile. Full project
editor import completed without script errors. The geometry helper was checked
for finite upward normals, terrain placement and road clearance. It contains
1,008,000 total triangles, culled in spatial patches beyond 65 metres; that count
is not a performance guarantee. The original comparisons predate regional integration below.

## Regional integration

The generated material now replaces only the pale Turn 2 field annotation.
The darker western annotation, distant ground and explicit bare access corridors
keep their existing materials. A bounded five sample maximum extends the pale
mask toward the shoulder, using offsets of three metres in the two map axes.
This is a directional artistic expansion, not a uniform measured dilation.
Region strength and texture bounds are respected.

Albedo, roughness and independently differentiated relief gradients blend under
the same mask. Differentiating heights before blending avoids adding artificial
ridges at region boundaries. The extra source is loaded as a baked texture
resource, so the production path does not depend on loose developer image files.

Root and independent review prefer the regional apex over blanket replacement:
it retains the new foreground while preserving outer terrain differences. The
source still lacks some larger disturbed straw patterns from the footage.

`--ground-study=regional` explicitly selects this path; `production` uses the
current default and `legacy` disables it for static comparisons. The geometry
study explicitly uses the earlier underlying material to preserve its original
isolation. Both existing scenery resources were rebaked after the track setup
changed. Track, terrain, collision and friction data are unchanged.

The rendered human controls check passed with zero failures. Linux frame
intervals were 17.36 ms median and 17.92 ms p95 across 265 samples. These are
local short test results, not native Mac performance measurements.

Five real agent camera captures at 640 by 360 passed PNG and hash validation,
immutable artifacts, frozen tick and deterministic idle pose, changed pixels
after advance, queued advance/reset serialization, no privileged telemetry,
and headless rejection. Evidence: `artifacts/regional-ground-camera-qa`.

The actual default material is captured in
`artifacts/regional-ground-production-apex/production.png` and compared with
the previous material in `artifacts/regional-ground-comparison/00.png`.
`artifacts/regional-ground-production.mp4` records the final default over the
same 2699 replay transitions, replacing the earlier blanket material video as
the current game review clip. Mac export `aab7e3af3554-46126a98593b` succeeded;
the current laptop was offline at verification, so native review is pending.

## Direct scanned asphalt experiment

`preview_lighting.gd --asphalt-study=scan` isolates the existing CC0 ambientCG
Asphalt010 color, normal and roughness maps from the authored road shader.
It uses an estimated 0.5 metre repeat, color gain 0.45 and full normal strength.
These are artistic settings, not measured Thunderhill material properties.
The capture metadata records source paths, hashes, shader hash and settings.
Production material overrides are rejected in this mode to keep the trial clear.

Both entry (station 950) and apex (station 1150) rendered with Vulkan Mobile,
using camera 2, lateral offset 4 metres, lean minus 25 degrees and yaw 8 degrees.
Evidence is in `artifacts/asphalt-scan-direct-entry` and
`artifacts/asphalt-scan-direct-apex`, with controls in
`artifacts/asphalt-study-entry-control` and
`artifacts/regional-ground-production-apex`.

Do not promote this candidate. Root review at both angles and independent apex
review found conspicuous repeated cloudy patches and a cooler, matte appearance
that moves away from the supplied footage. The current material still has
artificial longitudinal shading, but its overall character is closer. A scanned
source alone does not establish a better match. Camera pose and motion blur
prevent a precise aggregate scale comparison with the video.

This developer preview changes no production material, geometry or simulation.

## Retained field structure control

The `structured` mode in both preview tools retains the generated regional
material while restoring the existing curved mowing and retained swath gains.
The shader's `photographic_structure_strength` defaults to zero, so this
experiment does not change the playable game's appearance. It changes albedo
only; coverage, relief, roughness and simulation remain as before.

Both static entry and apex captures compiled and rendered under Vulkan Mobile:
`artifacts/structured-field-entry` and `artifacts/structured-field-apex`.
Metadata records the mode, strength and shader/source hashes.
Root review of both angles and independent apex review found only a slight
change in middle distance mottling. This does not solve the uniform straw carpet,
so it is not promoted. The current source replaces larger field distinctions;
restoring modest color gains alone is insufficient. Future work needs distinct
sparse, flattened and disturbed material regions informed by multiple references.
The current footage does not establish exact mowing locations or strand scale.

## Sparse earth and straw composition

The `composition` preview mode combines the existing generated dense source
with original `sparse-field-v1` exposed earth and short cut grass. The second
source is color artwork, not a measured PBR scan. Both use four metre estimated
scale and the existing continuous four patch sampler. Sparse contribution is
60 percent across the pale region, rising toward 95 percent inside the existing
flattened corridor according to its authored point tint interruptions.
The corridor is an appearance reconstruction, not surveyed contemporary cover.

Each source supplies its own estimated roughness and relief. Gradients are
calculated before coverage blending, so changes in material weight do not
introduce fictitious geometric ridges. Bare access regions retain the prior
exclusion and the independent road mesh is unchanged.

The initial uniform corridor blend reduced the carpet appearance, but independent
review flagged a broad gray brown band. The subsequent variant uses the existing
point strength interruptions instead of treating the corridor as uniform.
Static evidence: `artifacts/composed-field-entry`, `composed-field-apex` (initial),
and `composed-field-interrupted-apex` (revised).
The revised apex comparison is `artifacts/composed-field-comparison/00.png`.
This remains an experiment. It improves sparse material identity but does not
establish accurate placement of individual disturbed patches in the footage.

The revised composition completed the same 2699 transition replay; the review
movie is `artifacts/composed-field-motion.mp4` (700 frames at fixed 30 FPS,
23.33 seconds including the ending hold). Twelve samples at two second intervals
showed no obvious material boundary seam. This sparse inspection does not prove
absence of temporal shimmer. The initial dark frame and ObjectDB exit warning
also occur in prior captures and remain unresolved. Fixed movie timing is not a
live performance benchmark. Editor import and formatter checks passed. The
production sparse material flag remains false pending stronger spatial matching.

## Playable mixed ground integration

The default game now uses the 60 percent sparse base mixture in the existing
pale region. The additional corridor disturbance is disabled. The `composition`
study explicitly enables that uncertain treatment; `regional` and `structured`
explicitly disable sparse material to preserve their historical comparisons.
Static capture metadata records the effective sparse and disturbance settings.

Root and independent visual review prefer the mixed base to the dense carpet.
Larger directional patterns and distant vegetation still need work; this is not
an assertion of exact reference fidelity. The revised default is captured in
`artifacts/mixed-ground-verified-apex/production.png`.
Both scenery resources were rebuilt for the changed track setup. Human controls
passed with zero failures. Local Linux frame intervals were 17.184 ms median and
19.744 ms p95 across 267 samples. This short check does not prove Mac performance.

Five real agent camera captures passed hash, PNG, immutable artifact, frozen
state and queued capture/advance checks in `artifacts/mixed-ground-camera-qa`.
Track, terrain, surface, pavement and field coverage data are byte identical to
the prior commit; both scenery geometry hashes are unchanged after rebaking.
Mac export `2e0b5539ca76-d95c96458ffe` completed. This export has not been tested
natively on the laptop. The default replay recording is
`artifacts/mixed-ground-production.mp4`; the static before/after comparison is
`artifacts/mixed-ground-comparison/00.png`. The replay completed 2699 transitions.
