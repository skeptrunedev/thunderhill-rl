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

## Interrupted pavement reflections

The reference frame 30 has broken longitudinal reflective structure. The prior
material mainly varied albedo, while its roughness remained narrowly bounded.
The first trial combined stronger color variation and a 0.39 to 0.72 roughness
range. It looked wet and was rejected. A second trial held albedo fixed with a
0.46 to 0.70 roughness target, blended at strength 0.6. It still made continuous
blue reflective ribbons in the turn and was also rejected.

The accepted candidate interrupts those regions with a filtered noise mask at
approximately four longitudinal metres and 1.2 transverse metres. Substantial
matte pavement remains between the reflective patches. The underlying two
directional scales are 18 by 0.32 metres and nine by 0.075 metres. Annular noise
domains close at the lap boundary; unresolved cells fade with pixel footprint.
These dimensions and roughness values are artistic estimates, not measured
friction, tire wear or track surface data. Albedo and normals are unchanged.

Root and independent review prefer this distribution at strength 0.8. It is now
the asphalt shader default. Some patch ends remain too rounded and the camera,
sky and solar elevation are still uncalibrated. This is a modest improvement,
not a claim of photographic equivalence.

Evidence is in `artifacts/patchy-sheen-straight`, `patchy-sheen-apex`, and
`patchy-sheen-comparison`. The default game capture in `patchy-sheen-default`
is byte identical to the explicit strength 0.8 straight study. The diagnostic
metadata resolves shader defaults through Godot's
[RenderingServer API](https://docs.godotengine.org/en/stable/classes/class_renderingserver.html#class-renderingserver-method-shader-get-parameter-default)
when the material has no explicit override.

The recorded candidate completed all 2699 replay transitions. Review movie:
`artifacts/patchy-sheen-motion.mp4`, 700 frames at fixed 30 FPS. Twelve samples
across the clip and consecutive samples from 8.0 to 8.4 seconds show no obvious
abrupt patch changes. This inspection does not prove absence of shimmer over a
full lap. The previously observed initial dark frame and ObjectDB exit warning
remain. Fixed movie timing is not a live performance measurement.

Human controls passed with zero failures, and all five rendered agent camera
observations passed the capture and serialization checks. Control frame timings
may overlap another render and are not a performance comparison. Native Mac
verification remains pending.

Both static and replay preview tools accept `--asphalt-directional-wear=0..1`.
Zero provides the previous material response. Replay studies accept
`--ground-study=production` to preserve the current terrain while testing the
pavement, and record the effective roughness treatment strength.

Mac export `020989d98dca-f022e46a99eb` completed successfully. It contains this
material change but has not yet been verified on the physical laptop.


## Alternative material directions, September 21

Four isolated experiments keep the current game defaults intact. They were
rendered with the real Vulkan Mobile renderer at the same Turn 2 camera pose
(station 1150 m, lateral 4 m, lean minus 25 degrees, view yaw 8 degrees).
The supplied video frame 40 is an appearance reference, not a registered camera
match. No fidelity percentage is justified by these images.

* `--asphalt-study=matte-aggregate` replaces the directional asphalt treatment
  with original filtered procedural aggregate and weak generic scan normals.
  The estimated stone scales are 6 and 18 mm; roughness is 0.88. It removes the
  smeared reflective bands, but also removes genuine directional variation.
  It looks too flat to replace the current material.
* `--photographic-height-blend=1` ranks overlapping source patches using
  luminance as an artistic height proxy. Continuous normalized weights preserve
  patch edges, and the treatment fades when projected texture structure becomes
  unresolved. This changes the blending method, not the source image resolution.
  Nearby straw remains more distinct, but the larger field structure is still
  missing. Estimated normals change with the blended color.
* `--photographic-contrast=1` compensates for contrast lost in patch averaging
  using inverse weight variance around the source mip mean. This is approximate:
  transformed samples correlate, and clamping can shift the mean. It produces
  modest grain changes and cannot recover detail removed by texture filtering.
* `data/reference/turn2-straw-regions-coverage-study.json` expands the original
  pale region from 12 to 30 m before clipping against the road clearance.
  Build it with `tools/build_field_coverage.py` and pass its output JSON to
  `--field-map`. This reaches the straight's previously untreated shoulder, but
  makes the field too uniformly pale and loses the darker margin in the video.
  The annotation remains a study, not surveyed vegetation or a production map.

Root and independent visual review reject wholesale promotion of the matte
asphalt and wider coverage. The remaining task is spatial composition: preserve
uneven dark shoulder areas, directional cut vegetation and interrupted pavement
variation. Greater sharpness alone is not equivalent to reference fidelity.

Local evidence lives in ignored `artifacts/`: `texture-directions-review`
contains four video versus game sheets with source hashes, `height-layered-apex`
contains the new blend, `ground-coverage-straight` and `ground-coverage-apex`
contain the map trial, and `ground-variance-apex` contains contrast compensation.
`matte-aggregate-reviewed-apex` is the final matte capture with corrected
metadata. Earlier matte captures record the unchanged production defaults for
wear/joint parameters even though this study bypasses their visual effects.
The final capture records their effective strengths as zero.

Real renderer captures compiled both shader branches. Human controls passed
with zero failures. Preview GDScript format and diff checks passed. These are
local static studies, not new Mac performance results or validated motion
quality. No physics, training behavior or production material default changed.


## Broad roughness hybrid followup

`--asphalt-study=hybrid-aggregate` retains 75 percent of the authored pavement
albedo, including its broad spatial variation and paving joint, while replacing
narrow wear reflections with broad binder roughness. The final estimate is
0.68 plus centered binder variation of amplitude 0.04, specular 0.42 and generic
scan normal strength 0.08. The remaining albedo comes from the matte aggregate
study. These values are original appearance estimates. Production defaults are
unchanged.

A first 0.75 roughness trial remained too dull. Final matched static captures
are `artifacts/hybrid68-straight` and `artifacts/hybrid68-apex`, with reference
sheets in `artifacts/hybrid68-comparison`. The earlier `hybrid-aggregate-*`
captures are the 0.75 trial; its apex metadata also predates correction of the
paving joint report. The hybrid retains the joint through the authored color
mixture, unlike the completely matte trial.

The hybrid removes the conspicuous narrow blue ribbons, but still loses too
much directional brightness compared with footage frame 30. At the corner,
the reference also has a lighter outer road region. Root review and independent
review do not support promotion. Scene illumination and the broad placement of
surface variation remain larger fidelity gaps than aggregate resolution.

The existing replay study accepts `--hybrid-aggregate` with
`--ground-study=production`; it rejects a simultaneous directional wear override
and records the shader hash. `artifacts/hybrid68-motion.mp4` contains 700 frames
at fixed 30 FPS and completed all 2699 recorded transitions. Eleven sampled
frames show a gradual broad reflection change, without an obvious abrupt switch.
Sparse frame inspection cannot establish absence of temporal shimmer. The
existing ObjectDB exit warning remains; fixed movie timing is not a performance
measurement. Human controls passed with zero failures. GDScript formatting,
real shader compilation and diff checks passed.


## Curved straw mapping

The accepted field improvement maps the dense and sparse photographic textures
along the existing curved mowing coordinates, with image x following travel.
The appearance scales are six metres along travel and two metres across it,
replacing the isotropic four metre world tile. This is original visual authoring
from footage, not a recovered mowing survey. The production photographic mask,
road shoulder and bare access exclusions remain in force. Texture sources,
physical terrain and friction data are unchanged.

Both root and independent image review prefer `artifacts/curved-only-apex` to
`ground-variance-control`: the field now has coherent flow into the bend instead
of an isotropic tangled carpet. `curved-only-exit` and `curved-default-straight`
check adjacent views. Nearby straw is visibly stretched, and some reference
streaking is camera motion, so this is an improvement rather than a fidelity
claim. `photographic_curved_uv` now defaults true. Static and replay previews
accept `--photographic-planar-uv` for the earlier mapping.

A separate `--photographic-directional-composition=0..1` experiment varies the
sparse fraction from 0.22 to 0.98 over interrupted curved passes. Unresolved
noise converges to the existing 0.60 proportion. It has little independent
visual benefit and remains zero by default. Its preliminary combined replay is
`curved-composition-motion.avi`; the default game replay is recorded separately
as `curved-default-motion.avi`. Derivatives of each source height remain blended
after differentiation, avoiding invented ridges at material boundaries.

Human controls passed with zero failures and five agent camera observations
passed the existing capture and serialization checks in
`artifacts/curved-default-camera-qa`. These local checks do not establish native
Mac performance. Shader compilation, preview formatting and diff checks passed.


## Isolated canopy angular response

The static and replay tools accept `--canopy-backscatter=0..4` with production
terrain. A dedicated helper copies its shader and uniform values, then appends
a direct light function from `canopy_light_study.gdshaderinc`. Production never
installs this function. Terrain exposes a coverage varying only; the production
straight capture is byte identical to `curved-default-straight` after that edit.

Both the zero control and candidate use Burley diffuse and omit direct specular.
The diffuse expression follows the engine source at revision ed1daf0bf, with
its MIT notice included in asset attribution. The original additional response
is `strength * coverage * max(dot(VIEW,LIGHT),0)^2`, multiplied into direct
diffuse before shadow attenuation. Ambient is unchanged. This is an uncalibrated,
non energy conserving appearance hypothesis, not a physical canopy model.
The photographic region is treated as canopy for this experiment, including
soil present in its source image; that approximation must be revisited before
any production adoption.

An initial fourth power lobe produced only modest change (`canopy3-exit`). The
broader squared lobe gives a stronger exit field response without changing the
forward field patch. Root and independent review found the initial response
moves the relative view brightness in the right direction. The broad motion
trial becomes quite golden late in the bend and remains experimental. It does
not establish that the complete custom material is preferable to production:
the control differs by its missing direct specular. A production implementation
must preserve the existing lighting terms and account for energy and coverage.

`artifacts/canopy-isolation-check.json` records per-channel difference extrema
for explicit image crops. Road [450,350,800,550] and sky [0,0,1280,200] are
identical between zero and broad strength three in both views. Field
[50,295,250,330] is identical on the straight and changes at the exit, with
maximum channel changes of 35,33,24. This checks selected pixels, not an entire
scene shadow invariance claim. `artifacts/canopy-comparison` contains the pairs.

The final replay `artifacts/canopy-broad3-motion.mp4` completed 2699 transitions
and 700 frames. Eleven sampled frames show gradual field brightening through
the bend, without an obvious discontinuous switch. This sparse inspection is
not a complete shimmer or native performance test. The known ObjectDB exit
warning remains. Real Vulkan compilation passed for control and candidate;
out-of-range input returned exit two. GDScript format and diff checks passed.
No default rendering, physics or agent behavior changed.
