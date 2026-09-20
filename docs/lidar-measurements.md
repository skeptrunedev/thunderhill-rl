# Historical lidar measurements of Thunderhill East

We acquired and parsed all four original 2023 USGS LAZ tiles covering the circuit: 52,718,349 points, with 2,749,744 nonwithheld ground points retained within 30 meters of the approximate route. These support actual local slope measurements rather than invented terrain. They do not constitute an exact survey of the 2026 repave.

## Measurements

The script measures 52 sections. The table shows selected examples; full coordinates, sample counts and strip width sensitivity are in `data/reference/lidar-measurements.json`. Elevation is NAVD88 in meters. Negative cross slope falls to the left of the OSM route direction. These are local road strip slopes, not a claim that the entire pavement has a single camber.

| Station (m) | Longitude, latitude | Along grade | Cross slope | Fit residual RMSE | Ground points |
| --- | --- | --- | --- | --- | --- |
| 300.0 | minus 122.3311922, 39.5401476 | minus 0.16% | minus 1.15 degrees | 0.008 m | 215 |
| 1747.6 | minus 122.3290314, 39.5404320 | 9.89% | minus 2.15 degrees | 0.009 m | 254 |
| 1790.2 | minus 122.3291870, 39.5407951 | 14.33% | minus 2.01 degrees | 0.010 m | 241 |
| 1817.3 | minus 122.3292314, 39.5410324 | 3.83% | minus 3.95 degrees | 0.011 m | 236 |
| 1837.6 | minus 122.3294606, 39.5410664 | minus 13.30% | minus 1.84 degrees | 0.010 m | 215 |
| 1885.5 | minus 122.3298879, 39.5413248 | minus 12.73% | minus 1.87 degrees | 0.013 m | 235 |

Stations around 1747 to 1886 follow the mapped angular hill branch. Coordinates, rather than presumed turn names, identify the measurements. Station zero is OSM way 28825115 version 18, node 570035219 at longitude minus 122.3312159 and latitude 39.5428506. Distance follows node order, initially south, in the lidar projected coordinate system. This is our measurement index, not official surveyed stationing.

## Method and quality

`uv run tools/measure_lidar.py` acquires the original four files from pinned source URLs, checks file lengths, hashes their contents and decodes their LAS records. Only class 2 ground records without the withheld flag enter the analysis. A 4 meter longitudinal by 6 meter transverse strip around each approximate route station supplies an ordinary least squares plane. Heading uses the route positions 5 meters ahead and behind. Along grade is the plane's longitudinal derivative; cross slope is the transverse derivative expressed as an angle.

We also repeat the fit with transverse widths of 4 and 8 meters. At station 1790.2, these give approximately minus 1.98 and minus 2.02 degrees around the 6 meter fit of minus 2.01. At 1837.6 they give minus 1.87 and minus 1.82 around minus 1.84. This checks sensitivity to the selected strip, not absolute accuracy or a statistical confidence interval.

The [original USGS project report](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/CA_NorthCoastRanges_B23/USGS_CA_NorthCoastRanges_B23_Project_Report.pdf) reports nonvegetated point cloud vertical RMSE of 3.63 centimeters and 7.11 centimeters at 95% confidence across the project. Those are project results, not independent control checks at Thunderhill. The report identifies horizontal EPSG:6339 and vertical EPSG:5703 with GEOID18. The actual LAS coordinate quantization is 1 centimeter. Quantization and our local plane residuals are not absolute survey accuracy. The [NOAA collection metadata](https://www.fisheries.noaa.gov/inport/item/79080) identifies fall 2023 acquisition and nominal pulse spacing of 0.35 meters.

## What the profiles show about width

Each section has a plot of actual point heights and median return intensity in half meter transverse bins. At station 1790.2, visual inspection shows a low reflectance band roughly between offsets minus 7 and plus 3.5 meters from the OSM route. At station 1837.6 it is roughly minus 6.5 to plus 4.5 meters. These are approximate historical reflectance band observations, not accepted asphalt widths. Paint, curb material, scan angle, old patches and adjacent pavement can affect intensity. The crest section has a diffuse transition that cannot support a reliable width reading from intensity alone.

The important direct finding is that the OSM line is visibly offset from the center of the reflectance band. A road mesh that blindly places both edges at a fixed distance from that line would be unsupported. Cross section edge interpretations should be checked against the independently georeferenced aerial imagery. The central 6 meter fitted strips in the inspected examples remain inside the observed road band.

## Artifacts and use in the game

Original LAZ files, their hashes, all 52 plotted profiles, full binned profiles and the retained ground corridor are in ignored `artifacts/reference/lidar/`. `all-sections.json` contains detailed results; `data/reference/lidar-measurements.json` is the compact reproducible measurement table. Plot filenames use station distance, for example `profile-01790.2.png`.

Use these measured slopes as historical geometry constraints when constructing the road surface. Keep source dates and uncertainty visible in the asset manifest. Do not copy the local residual as a physics accuracy claim. Do not infer current curb height, post repave edge position, asphalt texture, friction or tire response from these points.

The source USGS point clouds are public domain. The measurement database uses OpenStreetMap station geometry and retains [ODbL 1.0 attribution and terms](https://www.openstreetmap.org/copyright), separately from MIT licensed analysis code. Local plot overlays and derived route data must carry the same source attribution when published.
