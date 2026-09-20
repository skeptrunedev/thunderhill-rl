# Surface rendering refinement

## Grass alpha coverage comparison

Grass materials now enable alpha coverage, with edge threshold 0.30 below the
existing scissor threshold 0.35. The project retains its existing 4x MSAA setting.
[Godot documents](https://docs.godotengine.org/en/4.5/tutorials/3d/3d_antialiasing.html)
that ordinary MSAA does not smooth internal alpha scissor texture edges without
material alpha antialiasing. The scenery was regenerated and its serialized
fingerprint verified. Placement, meshes, atlas colors and collision are unchanged.

The visual effect is limited. Matching 1280 by 800 captures at station 1830 differ
in 177 pixels, confined to the grass region. A closer moving comparison captured
16 identical camera poses for both disabled and enabled coverage at 1920 by 1080.
Between 1,217 and 2,060 pixels change per pair, while bright stems remain clearly
visible. These results establish that the material setting is active, not that
the overall vegetation appearance or temporal stability is solved.

Run the reusable comparison with a real renderer and an empty output directory:

```
DISPLAY=:1 godot --path godot --script res://tools/preview_grass.gd -- --output-dir=/absolute/path/grass-comparison
```

It loads the actual game and baked grass, freezes simulation, and captures both
material modes at each pose. The report includes camera transforms, image and
window dimensions, MSAA, thresholds and source frame names. Failed or all black
captures fail the diagnostic. Existing outputs are preserved. This is a visual
comparison, not a frame timing benchmark or agent training run.

Evidence: `artifacts/grass-before.png`, `artifacts/grass-after.png` and
`artifacts/grass-motion/comparison.json` with its 32 images. The rendered human
control check passed with zero failures, as did macOS packaging. The short Linux
control check reported median 17.171 ms and p95 17.820 ms over 275 process
intervals. Native Mac performance and appearance remain unverified for this build.

20 September 2026. This is an incremental visual correction, not physical realism acceptance.

The original terrain sampler interpolated each elevation cell bilinearly while the mesh rendered two planar triangles. The sampler now evaluates those same triangles. A synthetic saddle fixture verifies both interiors, the diagonal, corners, edges, translated coordinates, adjacent cells and bounds. This corrects scenery and shoulder vertex placement relative to the rendered surface. Shoulder segments can still cross terrain triangles between their endpoints; a fully stitched road boundary remains necessary.

Terrain now blends the corresponding normal and roughness maps with grass and soil colors. The rotated secondary grass normal is transformed back into the original tangent frame. Three additional CC0 maps were copied only after SHA256 verification against the existing material manifest, and use mipmaps. Attribution remains in the asset directory.

Asphalt has less prominent aggregate relief, nonperiodic broad variation and lower roughness to approach the grazing reflections visible in the inspected onboard footage. These are artistic settings, not measured optical properties. The ambientCG Asphalt010 metadata reports no physical dimensions; the chosen aggregate scale remains an estimate. No reference video pixels are distributed.

Curb paint bands use continuous road distance and derivative smoothing rather than assigning one color to each approximately three metre mesh segment. The existing 2.5 metre band length and placement remain provisional. They are not surveyed curb dimensions.

An actual Vulkan Mobile cockpit render was inspected at station 1650, saved locally as `artifacts/terrain-surface-refinement.png`. Import and rendering reported no shader errors. These changes do not establish a photographic match; terrain boundary stitching, exact curbs and motorcycle detail remain open work.

The full recorded input regression completed 40,313 physics ticks and one legal lap with no crash, zero off track ticks and zero trajectory error against the prior on road fixture. This was accelerated headless verification, not a performance benchmark.

The exported clean Mac build `e2f3a8483b52-e52f5d26d6f2` was installed at `~/Applications/ThunderhillReview/e2f3a84/Thunderhill.app`. Its Metal 4 renderer on the M3 Pro successfully captured the same cockpit scene using the game viewport. The retrieved PNG was visually inspected and is actually 1920 by 1200 pixels. This verifies the native shader/assets and render output, not screen presentation or sustained performance of this revision. Raw local evidence: `artifacts/mac-surface-preview.png` and `artifacts/mac-surface-preview.log`.

## Historical terrain color variation

The terrain material now includes a small geographically aligned color gain map derived from the acquired July 15, 2022 USDA NAIP image. The USGS service identifies NAIP as public domain. `tools/build_terrain_color.py` pins the source image hash, validates its catalog identity and projection, and uses the actual server returned extent rather than the requested bounding box. The map is 282 by 363 pixels, approximately four metres per pixel.

A conservative RGB classifier selects tan terrain pixels and rejects neutral pavement, deep shadows, green vegetation and bright roofs. This is not semantic segmentation; tan objects and weak shadows may survive. Accepted linear RGB values enter a normalized Gaussian convolution with six metre sigma. A median prior makes unsupported areas converge to neutral variation. Source RGB from rejected pixels does not enter the smoothing. Channel gains relative to the accepted terrain median are limited to 0.6 through 1.4, then area downsampled and encoded as gain divided by two. Four synthetic tests check exclusions, rejected color contamination, smooth bounded variation and entirely rejected input failure. Rebuilding reproduced the same PNG hash.

The shader samples this as numerical data, without sRGB decoding, at the documented local coordinates. It fades to neutral over 30 metres at the image boundary and uses a clamped sampler so surrounding hills cannot repeat the image. Mipmaps are enabled. Asset metadata is included in both native export presets, and package manifests hash it with the texture and shader.

The map retains broad historical shoulder and field patterns. It is neither measured reflectance nor a surveyed material classification and does not change contact geometry or grip. The source display stretch and illumination are not physically inverted. Existing CC0 material maps retain close detail. Terrain tint and warmer pavement binder are artistic adjustments against the inspected onboard frames at 00:40 and 01:10, not calibrated color measurements.

Clean build `f80ee3ff2420-46b8e5407863` was exported and installed at `~/Applications/ThunderhillReview/f80ee3f/Thunderhill.app`. Its actual Metal 4 render at station 1830 was retrieved and visually inspected at 1920 by 1200 (`artifacts/Thunderhill-macro-preview.png`). Native logs contain no resource or shader errors. The shared human controls diagnostic passed with zero failures; it injects engine input events and is not a physical keyboard/controller test. Its 395 samples reported median 11.364 ms and p95 12.5 ms, which is a short diagnostic rather than sustained performance acceptance. Source geometry and dynamics were unchanged. Final scenery detail, calibrated handling and human review remain incomplete.

## Directional asphalt finish

The asphalt material now applies a small centered contribution from its existing broad and fine directional noise to roughness. The contribution has strength 0.14 before the previous roughness bounds. Albedo, normal strength, texture sampling and physical grip remain unchanged. This is an artistic response to the varying longitudinal reflectance in the inspected Ken Moto frame at 01:10, not a measured pavement BRDF.

Before and after Vulkan Mobile renders at stations 400, 1650 and 1830 were captured at 1920 by 1200 and inspected. The effect is subtle, most apparent in the grazing reflection on the straight. All six captures completed without shader or resource errors. Native Mac verification follows separately; these images do not establish sustained performance or photographic realism.

Clean build `2906f099b501-1374319f0e76` was exported, transferred with matching SHA256 and installed at `~/Applications/ThunderhillReview/2906f09/Thunderhill.app`. Its native Metal 4 render at station 400 was retrieved and visually inspected at 1920 by 1200 (`artifacts/mac-roughness-preview.png`). The preview completed successfully without shader errors.

A full recorded input lap was launched separately, but macOS reports `CGSSessionScreenIsLocked=Yes`. The process and its recording continue to advance, substantially slower than real time. This run cannot establish foreground playing performance. Unlocking and a foreground check remain necessary; no new steady frame rate acceptance is claimed from the locked session.

That run subsequently completed with a persisted successful benchmark result: 40,313 ticks, one valid lap, zero crashes, zero off track ticks and zero maximum position error against the input fixture. The recorded viewport was 1600 by 1000, despite the requested command line resolution. Its 335.94 simulated seconds took 1167.03 wall seconds; median frame interval was 220.466 ms. These are poor locked session timings, not foreground performance acceptance. The SSH session exited with status 255 and did not retain console output, so completion was verified from the flushed `benchmark_result` in the game recording. The full result was retrieved as `artifacts/mac-roughness-benchmark-result.json`. The same installed app was then opened normally for human review.

## Terrain datum registration

The runtime terrain and color builders previously treated EPSG:6339 road coordinates as EPSG:26910 source coordinates. They now use the reviewed horizontal datum operation with verified local NADCON grid hashes. The height sampler transforms each requested position before sampling the DEM and rejects requests outside its domain. NAVD88 elevation values are unchanged. The full track builder delegates to this same sampler, preventing a rebuild from reintroducing the mismatch.

The corrected DEM changes grid heights by at most 0.454 metres, with a 95th percentile absolute change of 0.088 metres. These changes result from horizontal relocation over sloping ground. The rebuilt shared terrain mesh retains its existing road footprint and matches the road boundary to floating point precision. Runtime contact checks pass for 475 offroad queries, 2,213 mesh queries and 31 road queries, with maximum far terrain discrepancy below 0.000002 metres. Source provenance rejection checks also pass.

The color builder reprojects the filtered gains into an axis aligned local road coordinate grid. Every quadrature sample uses the full datum operation; the classifier remains unchanged. Six color tests cover filtering, pixel centers, axis direction, an affine deformation and neutral exterior samples. Two additional datum tests cover round trip transformation, a synthetic DEM ramp and out of bounds rejection. A 1920 by 1200 Vulkan Mobile render at station 1830 was inspected without shader errors. This corrects registration, not the remaining provisional pit envelope, bike handling or scenery detail.

Clean Mac package `e8a4056c68c1-99fc09529720` was exported and installed under `~/Applications/ThunderhillReview/e8a4056`. Transfer SHA256 matched `c1f902b4c00e0b269c75dc0b3688396a14d9a2479b493982b075f14a92fa4eae`. Its native preview remains pending: a live process sample showed a wait inside Metal compilation during texture upload, and two compiler services were subsequently observed using CPU. The process was preserved for completion. The Linux shared human controls diagnostic completed with zero failures; its headless timings are not graphical performance evidence.

The same native preview subsequently completed successfully on Metal 4 and the M3 Pro. The actual 1920 by 1200 frame was retrieved and visually inspected (`artifacts/mac-datum-preview.png`), with no shader or resource errors in the completed log. This verifies the registered terrain revision's native render, not sustained foreground performance.

## Painted concrete detail

The pit divider and provisional curbs now use the diffuse, OpenGL normal and roughness maps from [Rough Concrete by Dimitrios Savva](https://polyhaven.com/a/rough_concrete), distributed under [CC0](https://polyhaven.com/license). The publisher reports a 1.2 metre tile width. Download URLs, publisher MD5s and acquired SHA256 hashes are pinned in `data/reference/material-candidates.json`; `tools/fetch_materials.py` verifies them and can refresh the publisher metadata explicitly. The three runtime images exactly match the pinned downloads.

The painted finish uses the source grain with reduced contrast and normal intensity. Bright paint tint and restrained broad variation replace the earlier strong cell sized patches on the wall. These adjustments are artistic approximations, not measured Thunderhill albedo or reflectance. Road edge paint keeps its existing material branch. The divider now generates tangents for its normal map; track ribbons already generated them. All new images have mipmaps and anisotropic sampling for distant and grazing views. No displacement, wall dimensions, collision geometry or curb placement changed.

Final Vulkan Mobile renders at 1920 by 1200 were inspected for the divider and the Cyclone approach (`artifacts/concrete-wall-close.png`, `artifacts/concrete-curb-preview.png`). Imports and completed renders reported no shader or resource errors. The wall geometry check passed, the rendered contact suite passed 92 checks and the shared human controls check passed with zero failures. The short local control run reported median frame interval 17.361 ms and p95 18.407 ms on the RTX 2080 Ti. Native Mac verification of this material revision remains pending.

The native Mac attempt for build `7258532317ea-fb7b4e5ea76a` completed on Metal 4 without logged shader or resource errors, but its 1920 by 1200 PNG was entirely black. Successful PNG writing is not rendering acceptance. The screen was locked and older review instances were open; the cause of the blank frame is not established. Startup instrumentation reached scene ready at 169.353 seconds and the first post draw signal at 250.645 seconds. Evidence is `artifacts/mac-concrete-preview.png`, `artifacts/mac-concrete-preview.log` and `artifacts/mac-concrete-startup.jsonl`. Native material appearance remains unverified.

### Capture validation and native diagnostics

Screenshot capture now writes a JSON sidecar with camera selection, window draw
availability and focus, frame counters, and viewport objects, primitives and draw
calls sampled before and after the draw signal. The CPU worker preserves the PNG,
rejects empty or entirely black RGB data, and reports file or sidecar write errors.
Preview mode exits unsuccessfully for a failed capture. A nonblack image alone is
not an appearance acceptance test.

The synthetic capture test covers RGB and RGBA black images, a colored pixel,
empty images and failed output writes. A Linux Mobile rendered preview at
1280 by 800 produced a visually inspected cockpit image and 183 draw calls.
The previous black Mac capture remains unexplained pending native diagnostics.

The instrumented native build `871573bdaea5-b8f4af621f89` produced a valid,
visually inspected 1280 by 800 cockpit capture on the M3 Pro Metal renderer.
Both diagnostic samples reported the intended camera, 183 objects and draw calls,
1,774,555 primitives, window_can_draw true and window_focused false. The original
blank capture was not reproduced; its cause remains unproven. Startup reached
scene ready at 68.139 seconds and first draw at 68.485 seconds. This is not a
foreground performance acceptance result, particularly with older instances open.
Evidence: `artifacts/Thunderhill-diagnostics-871573b.png`, its `.png.json` sidecar,
`.jsonl` startup trace and `.log` runtime output. The transferred archive SHA256
was `f50f3eec458fef3593b1f34ba76389a29168acf380cf4185845b7ac4041bb91e`;
native code signature verification passed.

### Static scenery bake

The 96,000 fixed decorative grass instances and provisional marker boards now
load from `assets/generated/scenery.scn`. Runtime no longer repeats the seeded
placement, exclusion and terrain sampling work. Regenerate with the pinned Godot
editor executable and a real renderer:

```
DISPLAY=:1 godot --path godot --script res://tools/bake_scenery.gd
```

The generator rejects headless rendering because its MultiMesh buffer behavior
did not preserve the authored instance data. It compares mesh arrays, transforms,
instance colors, material storage properties, visibility ranges and shadow flags
before saving and after an uncached reload. It verifies 96,000 instances and writes
source hashes to `data/scenery-bake.json`. Packaging rejects changed inputs or a
changed scene artifact; source runs verify all inputs, and exported runtime
verifies ground JSON inputs (export converts scripts and textures).

Local evidence: generation 6250 ms, scene resource reload and instantiation
44 ms. Complete runtime scenery stage, including source validation and attachment,
was 173 ms. These timings measure different scopes and are not a native benchmark.
The baked and generated cockpit captures were pixel identical at 1280 by 800.
Evidence: `artifacts/scenery-bake.log`, `artifacts/baked-scenery-startup.jsonl`,
`artifacts/baked-scenery-preview.png`, compared with `artifacts/reservoir-preview.png`.
The saved scene is approximately 5.4 MiB. Grass is still decorative and has no
collision; this does not change simulation or agent observations intentionally.

The native `93e71c38b2bf-023e62d45bc2` preview reduced the recorded scenery stage
from 25,433 ms in build 871573b to 586 ms. Landmarks took 27,865 ms, scene ready
56,688 ms and first draw 59,265 ms. These are individual runs with older instances
still open, not a controlled foreground benchmark. Archive SHA256 was
`ab6f2f5eeefa01150122ed2f4460eb645af1e54af75e81ca3d487e4731df6f52`;
transfer hash and native signature verification passed.

Native appearance validation FAILED: the saved PNG was entirely black and the
preview correctly exited with status 1. Both capture diagnostic samples reported
183 draw calls, 1,774,555 primitives, the intended camera, window_can_draw true and
window_focused false. Thus these counters do not establish valid texture pixels.
The earlier native black capture is reproducible intermittently; its cause
remains unproven. Do not infer that the scenery bake is visually accepted on Mac.
Evidence: `artifacts/Thunderhill-scenery-93e71c3.png`, its `.png.json` sidecar,
`.jsonl` startup trace and `.log`. Local rendered input checks passed with zero
failures, median frame interval 17.361 ms and p95 18.750 ms in 273 samples.

### Readback isolation probe

`godot/tools/probe_capture.gd` renders one unshaded box against a solid background
in both the root viewport and an offscreen SubViewport. It captures samples 1,
13 and 60, records readback durations, and compares CPU image hashes on the main
thread and a worker. It saves every result and fails if any sample is empty,
black, differs across threads, or cannot be saved. This is a diagnostic sequence,
not a retry policy for gameplay capture. Six local Vulkan samples passed and the
root image was visually inspected (`artifacts/capture-probe-linux*`).

`tools/package_game.py --debug` now uses the debug export template and a distinct
build identifier, exposing engine diagnostics omitted from release templates.
One hypothesis requiring runtime evidence is an ignored Metal fence timeout.
The pinned engine selects MTL3 even with a Metal 4 capability string:
[driver selection](https://github.com/godotengine/godot/blob/ed1daf0bf001b61586d9930840f2f1394092c079/drivers/metal/rendering_context_driver_metal.cpp#L90).
Its timeout log is conditional on DEBUG_ENABLED:
[fence wait](https://github.com/godotengine/godot/blob/ed1daf0bf001b61586d9930840f2f1394092c079/drivers/metal/rendering_device_driver_metal3.cpp#L50).
The rendering device does not check the wait return value:
[frame stall](https://github.com/godotengine/godot/blob/ed1daf0bf001b61586d9930840f2f1394092c079/servers/rendering/rendering_device.cpp#L8214).
These facts establish a diagnostic lead, not the cause of our black captures.
Game capture sidecars now also record GPU readback duration.

The exported app does not honor the editor's `--script` entry option. The first
native probe attempt therefore launched the normal game, as its THUNDERHILL_READY
log established, and supplies no probe evidence. That diagnostic process was
terminated explicitly. The app now has a dedicated `-- --capture-probe` mode
which disables gameplay processing and enters the probe before track construction.
Use `--probe-output=/absolute/path/prefix` after the argument separator. The
CAPTURE_PROBE_BEGIN marker confirms the correct path. The same app entry mode
passed all six local samples (`artifacts/capture-probe-linux-mode*`).

Fence construction now uses the existing exact point height cache for interpolated
endpoints. Consecutive segments share these coordinates; previously both sampled
the full terrain path again. Segment count and point interpolation are unchanged.
Rendered geometry hashes before and after match exactly:
`8ecc20f4594f5df2264cc7dcb85384270382cb46ab0126bc983fcad29d14a6e9`.
The local runs took 6337 ms and 6233 ms respectively, too close to establish a
meaningful overall startup improvement from these individual measurements.
Materials and pit wall collision code were not edited. Evidence is in
`artifacts/landmark-cache-before.log` and `artifacts/landmark-cache-after.log`.

The corrected native probe in debug build `5cd19a5b44e3-c425f467ddf8-debug`
entered CAPTURE_PROBE_BEGIN and completed all six samples with valid images.
Every root and offscreen RGB hash matched the corresponding local Vulkan sample;
main and worker hashes also matched. The root capture was visually inspected.
However, the debug engine logged `timeout waiting for fence` at metal3.cpp:54
between samples 1 and 13. Thus this Mac does experience the source identified
synchronization timeout, but this test does not establish that it causes the
intermittent black game image. Image checks passed; the engine log was not clean.
Native readback durations were 26.255, 7.166, 305.885, 5.564, 149.939 and 2.835 ms.
Evidence is in `artifacts/Thunderhill-probe-5cd19a5*`. Archive SHA256
`67b4aba3c41dcf7806584e6f013eab5258474e7a0222cc5182561409bafd5759`
matched after transfer and native signature verification passed.

The full game using the same debug build reproduced an all black image and exited
with status 1. The log contains 17 `timeout waiting for fence` errors. Two include
a GDScript backtrace to main.gd:720, verified in commit 5cd19a5 as the
`get_viewport().get_texture().get_image()` call. Readback took 2,015,201 microseconds,
consistent with those two one second fence waits. This directly establishes
failed GPU synchronization during the failed capture; the engine source above
shows that readback continues despite the fence errors. It does not yet establish
why this Mac's GPU work stalled, nor prove that every previous blank image had
the same cause. The screen remained locked with earlier game instances open.
No retry, backend change or timeout extension has been applied.

Scene ready was reached at 77,062 ms and first draw at 155,415 ms. The diagnostic
process completed and is no longer running. Evidence is preserved as
`artifacts/Thunderhill-debug-game-5cd19a5.log`, `.jsonl`, `.png` and `.png.json`.
The capture counters still showed 183 objects and draw calls with the intended
camera, demonstrating why these CPU counters cannot certify rendered pixels.
