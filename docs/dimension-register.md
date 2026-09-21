# Measured dimensions for the first game

Checked 19 September 2026. These are measured historical geometry and published motorcycle dimensions, with explicit limits. They are sufficient to constrain the first reconstruction in meters. They are not an exact survey of the 2026 surface.

The project owner explicitly approved historical geometry as the construction baseline on September 19, 2026. Current videos guide appearance; exact repave measurements are not a prerequisite for building or reviewing the playable game. Historical source dates and estimated details remain recorded.

## Requirements retained

1. Thunderhill East using historical measured geometry, with the Cyclone reference sequence and current repave appearance.
2. Ducati Streetfighter V4S, selected as model year 2026.
3. Original or reusable assets in a public Godot repository.
4. A playable, attractive game verified on the M3 Pro Mac before any RL training.
5. Physical geometry and future telemetry must remain independent of graphics quality.

## Dimensions we can use now

| Quantity | Evidence and implementation value | Limit |
| --- | --- | --- |
| Overall route footprint | Approximately 840 m east to west by 1,022 m north to south from the projected OSM route | Approximate centerline, not pavement footprint |
| Mapped route length | Approximately 4,607 m along the acquired OSM polyline | Not an official lap measurement; do not stretch geometry to the promotional three mile label |
| Terrain working area | 1,128 m by 1,451 m, sampled on a one meter USGS elevation grid | Grid spacing is not vertical accuracy |
| Historical pavement envelope | Eight reviewed aerial sections span 10.8 to 12.0 m | At least ±1.2 m interpretive uncertainty per width; does not describe all corners or current runoff |
| Cyclone approach grade | Approximately +14.3% in a measured local strip | Historical 2023 lidar, local tangent and approximate route location |
| Cyclone descent grade | Approximately −13.3% in a measured local strip | Historical 2023 lidar, not a maximum for the whole descent |
| Local transverse slopes around Cyclone | Approximately −2.0°, −4.0°, and −1.8° at the three documented approach, crest, and descent samples | Negative means falling toward the left of OSM node order; not automatically inside or outside banking |
| Ducati wheelbase | 1,496 mm | Current official S generation; exact 2026 manual pending |
| Ducati rake and trail | 24.5° and 99 mm | Manufacturer baseline, not a handling validation |
| Ducati suspension travel | 125 mm front, 130 mm rear | Does not supply damping or spring curves |

The [lidar measurements](lidar-measurements.md) record the raw point processing, precise sample coordinates, fit sizes, source accuracy, and measured elevations. The [motorcycle reference](motorcycle-reference.md) contains all published baseline dimensions and dynamics gaps.

## Aerial width measurement

The [USGS NAIP ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer) supplies a calibrated overhead source, rather than perspective estimates from a motorcycle camera. Catalog item 23435 is `m_3912230_sw_10_060_20220715`, acquired July 15, 2022, with a stated native resolution of 0.6 m. The export is locked to this source image in NAD83 UTM zone 10 (EPSG:26910). The source is public domain USDA imagery distributed by USGS.

The reproduction tool samples color transitions perpendicular to approximate route tangents. It compares three thresholds and generates rotated strips for visual review. It does not automatically determine asphalt edges. Pit lanes, curbs, shadows, gravel, and road junctions create false boundaries. Of 23 candidate sections, eight were retained after visually reviewing both contact sheets and the overview:

| Section | Distance from first OSM node | Preliminary envelope width |
| --- | --- | --- |
| S08 | 1,500 m | 12.0 m |
| S10 | 1,900 m | 11.4 m |
| S11 | 2,100 m | 11.7 m |
| S13 | 2,500 m | 11.4 m |
| S15 | 2,900 m | 10.8 m |
| S17 | 3,300 m | 11.4 m |
| S18 | 3,500 m | 11.7 m |
| S22 | 4,300 m | 12.0 m |

Reported decimals preserve the detector output, not decimeter accuracy. Use at least ±1.2 m around each estimate as an initial interpretive envelope, representing two source pixels across the full width. This is not a calibrated confidence interval. Registration, tangent direction, and ambiguous paint can add further error. Sampling every 0.3 m does not improve the underlying 0.6 m resolution. These widths include unresolved painted edge pixels; they do not define legal racing limits or curb width.

The rendered edge stripe uses a provisional 0.20 m width with its outer edge
0.06 m inside the pavement boundary. The 0.20 m value is an artistic estimate
adopted after comparing 0.12 m and 0.20 m renders against onboard footage,
especially frame 00:40. Neither value is a measured Thunderhill specification.
The aerial resolution cannot determine stripe width. Paint geometry does not
define collision, legal racing limits or the agent's track boundary.

S01 and S02 illustrate why review matters: an automatic threshold includes adjoining pit pavement and reports roughly 30 m. Those results are excluded. The broad junction at S20 and several curb dominated sections are also excluded. We must trace both actual edges and recenter the road before generating its collision mesh. Do not apply one width symmetrically around every OSM point.

Run `uv run tools/fetch_geometry.py`, followed by `uv run tools/measure_ortho.py`. Generated imagery and contact sheets stay in ignored `artifacts/reference/ortho`. The committed [measurement register](../data/reference/ortho-measurements.json) preserves the inspected results, coordinates, source metadata, hashes, acceptance decisions, and uncertainty. It is a reviewed snapshot; rerunning the detector creates new unreviewed candidates, not automatic replacements for the accepted record.

## What the repave changes

The [operator's engineering account](https://www.thunderhill.com/news-from-the-hill-1/inside-the-thunderhill-repave-the-engineering-behind-the-new-surface) says the intended broad width, grade, and camber were preserved. It also describes changes to usable space around turns 3 and 4, 6 and 7, and 10 and 11, and new exit curbs based on an FIM design. Therefore historical lidar is a defensible base for large scale shape, but historical edges cannot certify current corner limits.

No public dimensioned drawing of those installed curbs was verified. A generic FIM drawing would not prove the actual installed profile. The supplied videos establish appearance and configuration cues, but do not supply a calibrated camera or known scale at every edge. Current curb height, width, detailed serration, exact widened boundaries, and surface grip remain unresolved. Preserve those as configurable estimates until they can be checked against an actual construction drawing or measured current reference.

## Existing games

The [simulator audit](simulator-track-references.md) found an Assetto Corsa author who documents using lidar, but no verified redistribution permission for its mesh. The inspectable CARLA Thunderhill source is flat, has zero banking, and constant 10 m lanes. It would erase precisely the features needed here. Use our acquired USGS data for physical reconstruction; existing game footage can support visual comparison without importing unlicensed assets.

## Modeling decision

Use meters throughout, with a local origin subtracted from projected coordinates before generating Godot meshes. Preserve the source coordinate system, vertical datum, origin, and conversion in generated metadata. Fit the pavement from raw lidar and reviewed edges, preserve the crest, and blend surrounding terrain into it. Do not add invented bumps from texture noise to the collision surface. Keep any estimated curb profiles separate and versioned so that later measurements can replace them without rescaling the circuit.

These results remove the need to guess overall scale, slope, or a generic track width. They do not yet justify calling every current dimension exact or the motorcycle physics validated.
