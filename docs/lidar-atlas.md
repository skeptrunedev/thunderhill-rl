# Continuous lidar surface

The new atlas is an experimental replacement for the road's centerline bank extrapolation. It is not yet loaded by the playable game. Historical class 2 ground observations constrain two dimensional height patches, allowing the road interior and edges to have different slopes and curvature.

## Construction

`tools/build_lidar_atlas.py` places patch centers on a 12 metre grid covering sampled road positions. Each patch fits a cubic tensor spline over a 32 metre square, using 2 metre knot spacing and bending regularization of 0.01 m⁴. These settings carry forward the local Cyclone experiment; they are not calibrated Ducati or surveyed pavement parameters. Only the interior 24 metre square contributes to the atlas, keeping support away from spline endpoints.

Each patch also receives a separate validation fit. Globally aligned 2 metre spatial cells determine the withheld fold, so a given observation has the same fold in overlapping patches. Validation includes all withheld points inside each fit square. Reported patch residuals can include soil and curb observations outside the pavement. They measure interpolation within the historical acquisition, not survey accuracy. The serialized atlas uses fits to all observations; its whole circuit audit is a coverage and differential check, not withheld validation of the blended result.

Patch weight along either axis is `(1 - t²)³` inside `abs(t) < 1`, and zero outside. The two axis weights multiply. Height is the weighted height sum divided by total weight. Gradient and Hessian differentiate that entire quotient, including changing weights. Weights and their first two derivatives vanish at support boundaries, so the blend preserves C2 continuity wherever total weight is positive. A support boundary with no other patches is outside the domain, rather than an extrapolation path.

The Godot evaluator uses a spatial index to visit nearby patches. When only one patch is active, it returns that patch exactly, avoiding cancellation as its weight approaches zero. Invalid configuration fails without replacing an existing valid atlas. Missing coverage and nonfinite results return an empty result rather than an invented height.

## Verification

Analytical Python tests check plane preservation, weight derivatives, changing patch weights, quotient derivatives and internal support seams. An independent Godot implementation is compared with Python at negative spatial cells and both sides of internal support boundaries. The analytical fixture uses different patch heights so omitted blend derivatives cannot pass accidentally.

Build the full experimental data with `OPENBLAS_NUM_THREADS=1 uv run tools/build_lidar_atlas.py`. The single BLAS thread avoids thread pool overhead on the small sparse patch fits. Output is `artifacts/road-surface/lidar-atlas.json` and its audit. `--limit` creates an explicitly marked partial diagnostic that the circuit audit rejects.

Run `uv run tools/audit_lidar_atlas.py` to check seven lateral positions at one metre station spacing and generate the full circuit Godot fixture. It checks source hashes, positive weight, first and second derivatives, distance to raw observations, differences from the previous ribbon and directional normal curvature. Run the Godot test with `godot --headless --path godot --script res://tests/test_lidar_height_atlas.gd -- --fixture=/absolute/path/to/artifacts/road-surface/lidar-atlas-circuit.parity.json`.

The Python test command is `uv run --with numpy==2.4.3 --with scipy==1.17.1 python -m unittest discover -s tools -p test_lidar_atlas.py`. Its `parity_fixture()` function returns the reproducible analytical fixture as a JSON serializable dictionary.

## Remaining integration

Road mesh vertices, road contact height and normals, local grade and bank metadata, curb offsets and the offroad boundary must use the same atlas. The current two triangles across the road width cannot represent the new cross section. Mesh subdivision needs an error audit against the continuous surface. Terrain datum conversion must also be explicit during the coordinated rebuild. No tire force change or physical realism acceptance follows from these surface tests alone.

## Full circuit results and acceptance limit

The first complete build contains 809 patches. The station grid audit checks 32,249 positions, covering seven lateral positions at one metre spacing. Every query has support, with total weight at least 0.376380 and one to four contributing patches. Maximum distance to a raw observation on this grid is 0.959429 m. A different grid at the original sample stations finds a maximum of 1.144369 m, so the one metre grid is not a global bound. Height changes relative to the old ribbon have 95th percentile absolute magnitude 0.106836 m and sampled maximum 0.284843 m.

Godot passes all 32,249 historical comparisons with maximum component discrepancy 0.00000002368567, in addition to the 307 analytical comparisons. These establish implementation agreement, not physical surface accuracy.

The strongest sampled negative directional curvature is negative 0.078614 per metre at station 392 m, lateral negative 4 m. Its raw local fit has only 17 points within 1.5 m, insufficient for the audit threshold. At radii 2, 3, 4 and 6 m, raw curvature varies from negative 0.062488 to negative 0.000752 per metre. This feature is strongly dependent on support scale.

More seriously, the strongest positive curvature is 0.048562 per metre at station 4502 m, lateral negative 6 m. Raw quadratic fits at radii 1.5 through 6 m remain between negative 0.000317 and positive 0.002591 per metre. At that location the four contributing spline patches individually produce approximately 0.041 to 0.049 per metre, so the unsupported peak originates within the patch fits, rather than being introduced by the blend. Global C2 continuity has therefore solved the seam problem but has not validated the fitting model at the pavement edge.

Per patch withheld RMSE reaches 0.068744 m; maximum individual withheld error reaches 0.724886 m. Those metrics include the surrounding square, not just road positions. They must not be presented as road surface survey accuracy.

This version is not accepted for playable tire contact. Next work must distinguish pavement support from nearby curb and soil, and check derivative sensitivity at the identified features. The full circuit audit now preserves raw local fits at every reported extremum, including explicit insufficient support results. Clamping curvature or claiming realism from the parity results would not resolve the source fitting issue.
