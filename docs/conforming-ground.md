# Road and terrain boundary

> Historical verification note: standalone unit tests referenced below were removed at the user's request. Their results remain historical evidence, but their old commands are no longer available. Use the actual game checks described in [training verification](../training/README.md#verification).

20 September 2026. The old terrain builder lowered ground toward 0.65 metres below the local road plane, extending that modification up to 16 metres beyond the road edge. On rising ground this could remove more than 0.65 metres. This was used to hide overlaps with separate road and shoulder ribbons. Despite old metadata calling it visual only, the lowered ground also supplied off road motorcycle contact.

The game now uses raw source DEM elevations on the original 8 metre grid, with no artificial recess. A generated 4 metre triangulation is clipped around the rendered pavement footprint. A provisional six metre shoulder transition joins its boundary heights to the measured ground. The outer transition boundary also splits the triangles, so the blend cannot bleed into the surrounding terrain. Its polygonal round joins approximate the six metre boundary with at most 1.808 mm chord inset.

The same indexed off road triangles supply the visible mesh and contact height/normal through a spatial index. Uncovered exterior points are errors, not silently substituted ground. The expected pavement cutout is identified separately. The existing analytic road and provisional curb contact remain unchanged. Close to corners, the analytic centerline based road classification can differ slightly from the visible polygon boundary. Curbs remain separate provisional geometry; this is not a guarantee that every curb and terrain join is watertight.

Regenerate all track data with `uv run tools/build_track.py`. To regenerate only the ground from existing track data, use `uv run tools/build_surface_mesh.py --output-dir godot/data`. Its default output is a review candidate in `artifacts/surface`. The builder uses constrained polygon triangulation to retain cutout boundaries and holes. Source DEM and track hashes accompany generated data. The derived surface retains OpenStreetMap and USGS attribution and ODbL licensing for road derived geometry.

New episodes record terrain and surface hashes. Replay and the input benchmark reject mismatched hashes when present. Legacy episodes without these fields retain their previous compatibility, so their ground provenance cannot be asserted from the older manifest alone.

## Verification

The final candidate has 115,170 vertices and 221,162 triangles. Generation checks road overlap, total coverage, duplicate area and boundary heights. Seven synthetic tests cover holes, split polygons, segment interpolation and preservation of raw terrain planes outside the transition. Runtime checks cover 2,213 visible triangle centroids, 475 far terrain samples and 31 road centerline samples. All sampled mesh normals face up; far terrain error was below 0.000002 metres, attributable to runtime floating point precision.

The agent API, recording and state replay checks pass, including 36 recorded transitions and 12 replayed transitions. The actual rendered human input diagnostic passes. A full on road recorded input run on the final geometry completed one valid lap with all 40,313 steps and zero trajectory error. Native Mac verification is recorded separately when available.

Road edge subdivisions match segment heights but retain topology T junctions against the separate road mesh. Historical source dates, provisional pavement widths, unsurveyed curbs and the artistic shoulder blend remain limitations. This change restores source terrain and removes overlapping shoulder meshes; it does not establish full track realism or calibrated motorcycle dynamics.

Destination contact validation runs inside the simulator step transaction. If the destination ground is missing or nonfinite, the complete pre step simulator state is restored and no successful transition or reward is recorded. Source hash checks use explicit runtime rejection and remain active in release exports. Regenerating both ground files from the pinned source produced byte identical output.

## Native Mac run

Clean build `90892d048231-37eb7b343bcf` completed the full recorded input circuit on the M3 Pro through Metal 4, using the human rider view. It completed 40,313 ticks and one lap, with no crash or off track tick, and zero trajectory error. Simulation time was 335.941667 seconds and wall time 336.415345 seconds.

Across 40,102 process frame intervals, median was 8.332 ms, p95 9.849 ms, p99 11.042 ms and maximum 582.252 ms. The maximum is a material stall, not hidden by the favorable percentiles. Its cause is not yet isolated. The run included an in game screenshot and PNG write; that is a possible contributor, not a demonstrated cause. These callback intervals are not GPU timings or proof of physical display presentation.

The captured native PNG was retrieved and visually inspected at 1920 by 1200 pixels; the logical viewport report remains 1600 by 1000. Raw evidence is `artifacts/mac-ground-benchmark.log` and `artifacts/mac-ground-benchmark.png`. The original review instance remained open during the run. Resident memory samples ranged from approximately 412 to 638 MiB; these are samples, not an instrumented peak.

The updated app is installed at `~/Applications/ThunderhillReview/90892d0/Thunderhill.app`. The scene, ground contact, model failure and provenance tests pass, along with the actual agent camera sequencing and image capture checks. The next performance investigation must account for the observed long frame rather than accepting median throughput alone.

## Shared pavement boundary and contact

The playable road now loads `godot/data/pavement.json`. Run
`uv run tools/build_pavement_mesh.py` to regenerate it from the pinned track and
terrain mesh. The loader verifies both source hashes. The exporter retains the
original road quad diagonals, subdivides triangles where terrain boundary
vertices interrupt their outside edge, and uses the exact terrain vertices for
those shared edges. It records generator provenance and source attribution.

This fixes a concrete topology defect: the old road edges and terrain boundary
were independently rounded and had different subdivisions. Removing the old
analytic contact fallback revealed 28 uncovered edge midpoint queries. Matching
edge arithmetic alone left 11. The final mesh shares all 7,896 terrain boundary
edges exactly. It contains 13,630 triangles and 10,765 UV indexed vertices.
Maximum alignment adjustment was 0.000054176 m horizontally and 0.000002424 m
vertically. The exporter verifies exact boundary edge identity, manifold edge
multiplicity, coverage and overlap, and reproduces identical output bytes.

`triangle_ribbon.gd` now supplies the same triangle stream to road rendering and
contact. `curb_surface.gd` adds curb identity to that shared implementation.
Terrain queries use scalar precision oriented edge tests without an extrapolation
halo. Contact selects the highest rendered pavement, curb or terrain triangle;
a missing surface is an explicit failure, never a fabricated plane.

Before the change, 12,288 rendered pavement samples showed a maximum height
mismatch of 0.053814 m. Final verification checks all 54,520 triangle interior
samples and 35,156 unique vertices and edge midpoints. Maximum interior height
error is 0.000005547 m; maximum boundary error is 0.000003152 m. All contacts are
finite. Existing terrain and curb contact checks also pass. Raw evidence is in
`artifacts/road-contact/watertight.log`; the reproducible test is
`godot/tests/test_track_contact.gd`.

This establishes agreement with the current historical mesh, not surveyed
pavement accuracy. Face normals remain piecewise constant, and no smooth
curvature, suspension or airborne dynamics is inferred from these triangles.
Track legality still uses its existing centerline rule. Physical pavement and
curb identities are now separate flags, so exposed pavement uses pavement grip
even where that rule and the mesh boundary differ.
