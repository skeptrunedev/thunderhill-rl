# Surface rendering refinement

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
