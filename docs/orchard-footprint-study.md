# Historical orchard visibility study

The existing landmark database contains 27 trees concentrated around the paddock.
The personally inspected July 2022 NAIP aerial also contains extensive orchard
rows north and northeast of the circuit. Existing terrain color processing
suppresses orchard interiors, so aerial input does not preserve their green tone.

`preview_lighting.gd --orchard-study` adds an isolated footprint experiment.
Three polygons were manually interpreted in the 1391 by 1792 inspection view
of `artifacts/reference/ortho/east-2022.png`. Their coordinates are explicit in
`godot/scripts/orchard_study.gd`. The source SHA256 and date are recorded in each
capture. Extent metadata is `data/reference/ortho-measurements.json`.

This is a visibility study, not a finished foliage asset. There are 2089 simple
ellipsoidal crowns. The 11 pixel row spacing, stagger, radii of 2.7 to 3.3 metres,
heights of 4.0 to 5.2 metres and palette are estimates. Individual trees have not
been traced. Positions use the existing extent-to-local conversion; differences
between NAD83 datum realizations remain uncorrected in this preview. The public
domain aerial is reference only and no aerial pixels are redistributed here.
Geometry is original code under the project MIT license.

At station 1150, camera 2, lateral 4 metres, lean minus 25 degrees and yaw 8
degrees, the orchard adds almost no visible change. The approximate visibility
audit projects all 2089 treetops inside the view and finds 2088 blocked by terrain.
It samples each sightline every eight metres using the terrain grid, so it is
not exact raster visibility and does not test entire canopies. The result is
consistent with the rendered image: adding this orchard does not solve the
missing distant dark band at this view. The correspondence between the orchard
and frame 40's band remains unproven. Do not reposition or inflate the orchard
merely to force a visual match.

Evidence is `artifacts/orchard-audited-apex/study.json` and `production.png`.
The preview compiled and rendered under Vulkan Mobile. It is not loaded by the
playable game. The next reference investigation should distinguish nearer
vegetation, barriers, and camera/terrain alignment before adding scenery.
