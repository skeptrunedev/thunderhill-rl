# Historical pit straight boundary evidence

20 September 2026. These measurements inform a future geometry correction. They do not change the installed game or establish surveyed boundaries.

`tools/measure_pit_divider.py` verifies the pinned raw lidar tiles and extracts all nonwithheld returns around stations 0 through 500 metres. At each section it fits a plane to classified ground on the interior pavement, then identifies raised returns 0.35 through 1.5 metres above that plane. This distinguishes physical structure from an optical shadow. Candidate bands use 0.2 metre bins with at least three observations, and retain the underlying observations for review.

The run retained 234,643 observations and evaluated 51 sections. At stations 150, 200, 300 and 350, raised bands occupy approximately lateral offsets minus 3 to minus 2.2 metres. The provisional right margin is approximately minus 6 metres. The road mask therefore includes physical divider structure. Isolated roof exclusions cannot correct this continuous boundary error. The height threshold locates raised structure, not its exact ground footprint or a legal racing limit.

Independent inspection of the pinned 2022 NAIP image places the opposite pavement and soil transition around positive 8 to 9 metres at stations 50 through 430, while the provisional positive 6 metre margin remains inside pavement. The optical measurement uses the verified NADCON transformation between the lidar and aerial coordinate systems. Its 0.6 metre pixels, threshold sensitivity and unquantified registration error prevent treating these candidates as exact dimensions. Local evidence is saved in `artifacts/road-surface/pit-east-edge-measurements.json` with the inspected overlay and profiles.

Next geometry work must adopt a consistent pavement envelope, then rebuild the fit mask, road, shoulders and contact sampling together. The existing centerline and symmetric width cannot simply be widened through the divider.

## Road tessellation preparation

`tools/build_atlas_road_mesh.py` preserves the current planar road footprint and samples the selected height atlas across its full width. It produces shared triangle indices, analytic normals, metre UV coordinates, a coincident closing seam and edge rows for a matching shoulder rebuild. This remains an experimental artifact until the boundary correction and contact integration are complete.

The selected exclusion atlas produced 63,804 vertices and 117,768 triangles. Across 471,072 triangle centroid and edge midpoint comparisons, maximum vertical interpolation error was 0.0184591 metres and the 95th percentile was 0.00389037 metres. These compare mesh interpolation against its source atlas, not against independent survey truth. Three analytical tests verify planar footprints, normals, winding, the closing seam, known quadratic error and refinement.
