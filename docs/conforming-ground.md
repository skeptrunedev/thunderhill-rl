# Road and terrain boundary

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
