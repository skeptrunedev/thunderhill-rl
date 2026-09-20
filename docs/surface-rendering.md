# Surface rendering refinement

20 September 2026. This is an incremental visual correction, not physical realism acceptance.

The original terrain sampler interpolated each elevation cell bilinearly while the mesh rendered two planar triangles. The sampler now evaluates those same triangles. A synthetic saddle fixture verifies both interiors, the diagonal, corners, edges, translated coordinates, adjacent cells and bounds. This corrects scenery and shoulder vertex placement relative to the rendered surface. Shoulder segments can still cross terrain triangles between their endpoints; a fully stitched road boundary remains necessary.

Terrain now blends the corresponding normal and roughness maps with grass and soil colors. The rotated secondary grass normal is transformed back into the original tangent frame. Three additional CC0 maps were copied only after SHA256 verification against the existing material manifest, and use mipmaps. Attribution remains in the asset directory.

Asphalt has less prominent aggregate relief, nonperiodic broad variation and lower roughness to approach the grazing reflections visible in the inspected onboard footage. These are artistic settings, not measured optical properties. The ambientCG Asphalt010 metadata reports no physical dimensions; the chosen aggregate scale remains an estimate. No reference video pixels are distributed.

Curb paint bands use continuous road distance and derivative smoothing rather than assigning one color to each approximately three metre mesh segment. The existing 2.5 metre band length and placement remain provisional. They are not surveyed curb dimensions.

An actual Vulkan Mobile cockpit render was inspected at station 1650, saved locally as `artifacts/terrain-surface-refinement.png`. Import and rendering reported no shader errors. These changes do not establish a photographic match; terrain boundary stitching, exact curbs and motorcycle detail remain open work.

The full recorded input regression completed 40,313 physics ticks and one legal lap with no crash, zero off track ticks and zero trajectory error against the prior on road fixture. This was accelerated headless verification, not a performance benchmark.

The exported clean Mac build `e2f3a8483b52-e52f5d26d6f2` was installed at `~/Applications/ThunderhillReview/e2f3a84/Thunderhill.app`. Its Metal 4 renderer on the M3 Pro successfully captured the same cockpit scene using the game viewport. The retrieved PNG was visually inspected and is actually 1920 by 1200 pixels. This verifies the native shader/assets and render output, not screen presentation or sustained performance of this revision. Raw local evidence: `artifacts/mac-surface-preview.png` and `artifacts/mac-surface-preview.log`.
