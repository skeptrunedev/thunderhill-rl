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
