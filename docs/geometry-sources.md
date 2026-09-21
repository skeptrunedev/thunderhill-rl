# Thunderhill East geometry acquisition

Checked 2026 September 19. We have acquired actual terrain and approximate course geometry. We do not yet have a survey of the new racing surface.

## Acquired artifacts

Run `uv run tools/fetch_geometry.py` from the repository root. The script has pinned Python dependencies and reads the source manifest at `data/reference/geometry-sources.json`. It downloads OSM geometry and reads a rectangular window from the USGS raster using HTTP range requests. All acquired artifacts are in ignored `artifacts/reference/geometry/`; third party geometry is not silently relicensed as MIT.

| Artifact | Verified result |
| --- | --- |
| `east-terrain-1m.tif` | 1451 rows by 1128 columns, 1 meter grid, all 1,636,728 samples valid |
| `east-osm.json` | OSM way 28825115 version 18 and its nodes |
| `east-centerline.geojson` | Approximate closed centerline, 260 coordinates including repeated closure |
| `acquisition-report.json` | Source URL, CRS, transform, elevation limits and SHA256 hashes |
| `terrain-preview.png` | Inspection preview with blue OSM line over shaded terrain, not a game texture |

The crop covers longitude minus 122.338 to minus 122.325 and latitude 39.534 to 39.547. Terrain elevation is 76.338 to 134.660 meters NAVD88 across this rectangle. Sampling the existing OSM nodes gives 84.559 to 117.908 meters, but this is only a centerline diagnostic. It is not a measurement of all course extremes or motorcycle surface roughness. The calculated spherical OSM polyline length is 4608.668 meters. Neither quantity should be represented as a surveyed course dimension.

## USGS terrain

Preferred [USGS product](https://www.sciencebase.gov/catalog/item/683e5a4fd4be0234870fdf99): `USGS 1 Meter 10 x55y438 CA_NorthCoastRanges_B23`, published 2025 May 30. [Direct GeoTIFF](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/CA_NorthCoastRanges_B23/TIFF/USGS_1M_10_x55y438_CA_NorthCoastRanges_B23.tif) is approximately 337 MB before cropping. Its actual raster CRS is EPSG:26910, NAD83 / UTM zone 10N. Elevations are meters referenced to NAVD88.

The [XML metadata](https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/1_meter/geotiff/CA_NorthCoastRanges_B23/USGS_1M_10_x55y438_CA_NorthCoastRanges_B23.xml) lists 2023 September 16 through December 15, although its `current` field says publication date. A verified [EPQS query](https://epqs.nationalmap.gov/v1/json?x=-122.33146&y=39.5404&units=Meters&wkid=4326&includeDate=true) independently returned acquisition date November 5, 2023, resolution 1 meter, and elevation 91.258209229 meters at that point. Thus this is demonstrably before the repave.

Four newer lidar point cloud products intersect the search rectangle: `556376`, `556377`, `557376`, `557377`, all under `CA_NorthCoastRanges_B23`. Their precise URLs and sizes are in the manifest. They total approximately 203 MB. These LAZ files are catalog verified but not downloaded. They are the next source if pavement edges or terrain features cannot be resolved from the DEM. The generic metadata does not establish a local surveyed vertical accuracy, and 1 meter grid spacing does not mean 1 centimeter accuracy.

The [USGS product description](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services) and product metadata identify 3DEP data as public domain. Preserve attribution, source dates, and a record of cropping or resampling.

## Plan geometry and variants

The [OSM main loop](https://www.openstreetmap.org/way/28825115) was last edited 2025 May 11. It has no width tag and is a centerline, not a pavement boundary. Its geometry is useful for the first scaled blockout. It is not sufficient to set legal track limits.

Its turn 5 segment follows the same nodes as [way 892530996](https://www.openstreetmap.org/way/892530996), which makes an angular turn west before curving north. A separate [Hill Bypass way 1538451376](https://www.openstreetmap.org/way/1538451376) takes a smoother direct northwest route between the same endpoints. This is consistent with the main loop representing the Cyclone route, but the intended rider video must decide the configuration. Do not infer route from the nominal 3 mile name. The official [track map page](https://www.thunderhill.com/track-info/track-maps) provides 3 mile, 5 mile, 2 mile and Middle Hill configurations.

OSM is under [ODbL 1.0](https://www.openstreetmap.org/copyright), requiring attribution and applicable database share alike obligations. The code remains MIT. Acquired raw and derived OSM geometry is kept outside the committed source tree, and the generated GeoJSON explicitly records its license. Publishing a game with OSM derived data requires carrying that attribution and satisfying those terms.

## What the repave changes

The circuit's [engineering report](https://www.thunderhill.com/news-from-the-hill-1/inside-the-thunderhill-repave-the-engineering-behind-the-new-surface) says the project aimed to preserve existing width, grade and camber. It also reports new curb geometry and more usable width around turns 3 to 4, 6 to 7, and 10 to 11. Therefore older lidar is a credible starting point for the hills and broad grade, but cannot certify current curb profiles, asphalt edges, runoff transitions, or grip. Apex Circuit Design worked on CAD for the project; no public CAD or post construction survey has been acquired.

## Implementation decision

Use the real DEM for terrain and initial grade, with a local meter origin so Godot never simulates at million meter UTM coordinates. Store the original CRS, origin, vertical datum and import transform. Generate a separately controllable road surface over that terrain. Preserve both course branches until video confirmation. Derive road width, camber and curb placement from stronger evidence before labeling the game geometrically validated. Smooth rendering and simplified collision meshes must not silently erase the elevation profile.

For the playable section, this evidence is enough to build recognizably correct terrain rather than invented hills. Hyperrealistic tire contact, curb response, surface grip and current track limits remain unvalidated and need explicit calibration.

## Turn 2 polygonal source path audit

The current rider view at station 950 shows a sharp left edge corner. The
production centerline changes heading by 6.838 degrees at station 976.173.
Width is exactly 12 m throughout stations 850 to 1150, so a width interpolation
change cannot explain this corner. Independent inspection connects this to
a 10.045 degree raw OSM vertex near source station 975.204. The current
1 m Gaussian smoothing sigma leaves long source chords with short transitions.
Subdividing those straight segments would preserve the visible kink.

`tools/audit_road_alignment.py --production` now overlays the actual shipped
piecewise linear center and edge frame, using the same horizontal normal
construction as `track.gd`, rather than requiring an experimental differentiable
surface. It retains a source only aerial panel and datum provenance. The CLI
refuses to overwrite existing image or report evidence.

The audit at stations 900 to 1320 is in
`artifacts/turn2-production-alignment.png`. Root inspected the historical
image and overlay: the path is visibly polygonal and its edges do not precisely
follow the historical pavement envelope. The outside exit width discrepancy
previously identified by the curb review remains unresolved.

`--plan-smoothing-m=9` creates a horizontal candidate without editing game data.
It filters uniformly spaced station samples and tapers displacement to zero
with a quintic weight over 30 m at both interval ends. The reviewed candidate
in `artifacts/turn2-plan-smoothing9.png` moves the center at most 0.645 m and
reduces the interval's maximum heading jump from 7.711 to 2.035 degrees. This
exceeds the original source smoothing tolerance, so it is explicitly a new
reconstruction study, not a silent adjustment to the source constraints.

`data/reference/turn2-plan-study.json` preserves the candidate coordinates,
original station labels, source hashes and integration requirements. It is
not runtime geometry. A visually smooth curve is not proof of correct road
boundaries. Any adoption must refit lidar heights and bank and rebuild shared
pavement, terrain interface, contact and dependent placements together.
Five analytical tests pass, including local smoothing on a nonuniform sampled
closed path, exact preservation outside its interval, source immutability,
invalid parameter rejection and the existing lidar curvature checks.
