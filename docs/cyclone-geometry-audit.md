# Cyclone geometry evidence

The experimental smooth ribbon produced extreme normal curvature at its inside edge around station 1821. This audit compares that inferred shape with independent raw ground observations before curvature affects playable tire loads. The shipped road and Mac review build are unchanged.

## Aerial registration

`tools/audit_road_alignment.py` renders the unmodified historical aerial crop alongside the experimental center and edge curves. It verifies hashes for the source track, derived surface and July 2022 NAIP image. The curve broadly follows the Cyclone branch rather than the bypass; widths and boundaries remain provisional, and the overlay is not a surveyed pavement mask.

The game's source route and raw lidar use NAD83(2011), EPSG:6339. The aerial export uses generic NAD83, EPSG:26910. Initially PROJ selected a ballpark geographic offset with unknown accuracy and zero coordinate displacement. The preferred operation was unavailable because four NADCON5 grids were missing.

The audit now acquires openly licensed grids through pyproj's documented grid downloader into `artifacts/reference/datum`, validates their hashes against `data/reference/datum-grids.json`, and uses that directory only within its process. It rejects unavailable preferred operations or changed grid contents. At the track origin, the selected operation changes easting by positive 0.255285 m and northing by negative 0.446407 m. Its reported operation accuracy is 0.2 m; this is not the combined accuracy of imagery, lidar and inferred track edges. The operation pipeline, grids, hashes and actual shift accompany the plot.

Existing game terrain and macrocolor builders still directly reuse the local coordinate frame. They need the same explicit datum policy during the next coordinated geometry rebuild. Installing project reference grids does not silently change those other tools or move the existing game.

## Lidar comparison

The old builder fits a central strip four metres long and six metres wide, then extrapolates bank out to road edges about 5.76 metres from the center. Independent plane fits at provisional edges between stations 1790 and 1850 contain over one hundred raw points per local strip, but differ from the old surface by up to roughly fourteen centimetres. The previous small central fit residual does not validate those extrapolated edges.

The reproducible audit fits local quadratic height graphs at the two curvature extrema using radii 1.5, 2, 3, 4 and 6 metres. It computes directional normal curvature in the modeled planar travel direction, including the fitted slope metric. It repeats the fit using only points at least 0.25 m toward the assumed pavement interior. That second selection is a sensitivity check, not a surveyed road mask.

| Location | Ribbon curvature per metre | Raw fit at 3 m radius | Interior side fit at 3 m radius |
| :--- | ---: | ---: | ---: |
| Station 1820.392, lateral 5.76445 m | 0.179396 | negative 0.017564 | negative 0.005446 |
| Station 1821.946, lateral 5.76345 m | negative 0.235268 | negative 0.019751 | negative 0.010995 |

All twenty radius and selection combinations yield negative curvature at these points. The ribbon's inferred local positive dip is therefore unsupported by these raw fits. Large changes in its lateral frame and compressed inside station metric explain the extrapolation artifact. The replacement should fit height across the pavement in two dimensions rather than inherit all edge elevation from central bank.

These local derivatives vary with neighborhood radius. Full neighborhoods can include curb and soil; the interior selection depends on provisional geometry. Residuals measure fit agreement, not independent survey accuracy. No radius or replacement surface is accepted merely because it gives a desirable load.

## Local replacement experiment

`tools/fit_cyclone_surface.py` fits a cubic tensor spline height field directly to 5,715 raw ground points in a 24 m square. All 144 spatial cells contain observations. The height field has continuous second derivatives within the patch and provides analytical slope and curvature without extrapolating centerline bank. It verifies track and lidar provenance and rejects stale curvature audit locations.

Six candidates compare 2, 3 and 4 m knot spacing with either zero or 0.01 m⁴ bending regularization. Three validation folds withhold entire 2 m spatial blocks, including boundary blocks. No candidate is automatically accepted.

The regularized 2 m candidate has withheld height RMSE 0.01216 m, 95th percentile absolute error 0.02387 m and maximum error 0.08685 m. Its two directional curvatures are negative 0.023497 and negative 0.011517 per metre. All six candidates give negative curvature at both locations. The unregularized 2 m candidate has a slightly smaller training residual but a maximum withheld error of 0.95409 m, showing why training residual alone is insufficient.

These results support replacing the ruled elevation model. They do not certify pavement geometry or motorcycle dynamics. Observations include shoulders and potentially curbs, neighboring validation blocks remain correlated, and height errors do not validate second derivatives. The patch has no global seam handling and is not integrated into the playable game. The next implementation must use a consistent surface for road rendering and contact, then verify continuity and coverage around the complete circuit.

## Reproduce

Run `uv run tools/build_road_surface.py`, then `uv run tools/audit_road_alignment.py --download-datum-grids`. The latter command only downloads missing openly licensed datum grids and verifies them against the pinned manifest. Subsequent runs can omit the download option. Artifacts are `artifacts/road-surface/cyclone-alignment.png` and its JSON report, including all raw fit diagnostics and source hashes.

`uv run tools/test_road_alignment.py` verifies an analytical tilted quadratic, one sided fitting, zero plane curvature under multiple directions, and rejection of collinear observations. Both source and overlay panels were visually inspected after datum correction.

Run `uv run tools/fit_cyclone_surface.py` after the road surface audit to produce `artifacts/road-surface/cyclone-height-fit.json`. It records all candidate coefficients, validation folds, coverage, solver diagnostics and input hashes. Run `uv run --with numpy==2.4.3 --with scipy==1.17.1 python -m unittest discover -s tools -p test_cyclone_surface.py` for the three tests covering analytical derivatives and bending energy, plane preservation, extrapolation rejection, invalid parameters and stale provenance.
