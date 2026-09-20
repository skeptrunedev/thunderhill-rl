# Track reference readiness and first playable section

Reviewed September 19, 2026. The first motorcycle is the 2026 Ducati Streetfighter V4S, explicitly selected by the project owner. Game playability, appearance, and Mac performance must be reviewed before any RL training begins.

The [motorcycle reference](motorcycle-reference.md) records verified manufacturer dimensions and market differences. Ducati's current page describes the MY25 generation; exact 2026 manual confirmation is still outstanding. Use its US specification only as a provisional baseline for the selected bike.

## What was actually acquired and inspected

The supplied [Speed Secrets East map](https://speedsecrets.com/wp-content/uploads/2018/12/Thunderhill-East.pdf) was rendered and visually inspected. It identifies the course, turn numbers, and the split at the Cyclone, but provides no surveyed dimensions or elevations. It is a layout reference, not a mesh or texture license.

Two recent videos were downloaded for local reference inspection. Their frames are not included in the public repository or licensed as game textures.

| Footage | Metadata and inspection | Useful evidence |
| --- | --- | --- |
| [Ken Moto, First Laps on Thunderhill East's New Surface](https://www.youtube.com/watch?v=yVjzZqYbKuM) | Published September 13, 2026; 321 seconds in publisher metadata. Description identifies the 2026 Ducati Streetfighter V4S. Inspected 33 frames at ten second intervals plus 12 detailed crest views at two second intervals. | Rider view, cockpit proportions, pavement and curb appearance, changing horizons, paddock and trackside landmarks. |
| [Matt, Thunderhill East Repave Aprilia RS660 2:03 Lap](https://www.youtube.com/watch?v=LjmP603avMQ) | Published September 15, 2026; 139 seconds in publisher metadata. Inspected 14 frames at ten second intervals. The lap time is the author's title claim, not a timing measurement made here. | Independent camera and motorcycle, same fresh pavement and curb palette, broad terrain and runoff appearance. |

This is a timestamped sampled visual review, not a frame by frame photogrammetric reconstruction. Both cameras move with the rider and have substantial perspective distortion. Do not estimate road width, camber, or speed directly from apparent image angles.

A [USGS terrain crop and approximate OSM route](geometry-sources.md) have been acquired. The crop covers the circuit with a one meter elevation grid and no missing pixels. Data is from before the repave. [Reusable material candidates](asset-source-plan.md) have also been downloaded and visually inspected, with pinned checksums.

## Visual observations to build against

| Source time | Observation | Implementation consequence |
| --- | --- | --- |
| Ken Moto [00:20](https://www.youtube.com/watch?v=yVjzZqYbKuM&t=20s) | Dark pavement, straight white edge markings, pit structures and trees concentrated to one side, sparse open terrain elsewhere. | Spend detail on nearby riding cues and distinctive structures rather than dense generic scenery. |
| Ken Moto [00:40](https://www.youtube.com/watch?v=yVjzZqYbKuM&t=40s) | Golden brown mown hills, thin road edge line, broad unpaved shoulder. | Blend dry grass and exposed earth; avoid uniformly green vegetation. |
| Ken Moto [01:04](https://www.youtube.com/watch?v=yVjzZqYbKuM&t=64s) through [01:22](https://www.youtube.com/watch?v=yVjzZqYbKuM&t=82s) | Approach rises to a blind crest, turns left, descends, and bends right. Blue and white curb bands, cones, and a broad distant view become visible as the horizon opens. | Strong candidate for the first playable section. Need real elevation and road collision geometry, not a flat map extrusion. |
| Ken Moto [02:30](https://www.youtube.com/watch?v=yVjzZqYbKuM&t=150s) through [02:50](https://www.youtube.com/watch?v=yVjzZqYbKuM&t=170s) | Large curbs beside dry ground, barriers and fencing near the paddock, braking boards near the edge. | Author curb profiles, barriers, fencing, and legible distance boards as distinct assets. Exact dimensions remain unmeasured. |
| Matt [00:30](https://www.youtube.com/watch?v=LjmP603avMQ&t=30s) through [01:10](https://www.youtube.com/watch?v=LjmP603avMQ&t=70s) | Independent view confirms smooth dark surface, blue and white curb bands, open brown hills, and strong elevation transitions. | Cross check the primary view without copying its camera roll into terrain geometry. |
| Matt [01:40](https://www.youtube.com/watch?v=LjmP603avMQ&t=100s) | Green painted runoff adjoins blue and white bands at a corner exit. | Preserve painted runoff as a separate surface from loose soil. Its friction is not established by its color. |

The crest sequence and the OSM angular branch are consistent with the Cyclone rather than the smooth Hill Bypass. Treat that as a working interpretation; preserve both route branches until their exact correspondence is checked during reconstruction.

The [track operator's repave account](https://www.thunderhill.com/news-from-the-hill-1/inside-the-thunderhill-repave-the-engineering-behind-the-new-surface) says broad width, grade, and camber were intended to be preserved while curbs and usable corner space changed. That supports using older terrain as a base. It does not establish that every road edge or cross section is unchanged.

## First playable section

Build the approach to the Cyclone, the crest and descent, and enough exit road to brake, turn, accelerate, and restart safely. Final endpoints come from alignment with the geodata. Include surrounding hills far enough to reproduce the skyline. Use the full route and terrain as context, but do not delay section review for every paddock building.

The section must support manual riding, smooth throttle and brake inputs, steering, camera selection, pause, restart, and a clear return to the starting position. Support a gamepad and keyboard on macOS. Keyboard input needs explicit input smoothing; it must not secretly change grip or balance physics. Expose any rider assistance in settings and telemetry.

Visual priorities are correct road silhouette and elevation, convincing lean and camera motion, fresh asphalt response, curb geometry, dry terrain blending, and stable shadows. Cockpit controls and bike silhouette should identify the selected Ducati, using original or appropriately licensed meshes. The video shows cockpit references but is insufficient to reconstruct the whole motorcycle accurately.

Compare screenshots from matched approximate viewpoints against the timestamp index, then test an actual ride. A screenshot alone cannot validate steering response, contact behavior, frame pacing, or motion sickness. The full circuit and polish follow this first section.

## Mac target and acceptance

The active target laptop was inspected using system_profiler: Apple M3 Pro, 12 CPU cores, 18 GPU cores, 36 GB unified memory. Do not publish device identifiers or network addresses. This is a hardware inventory, not a game benchmark.

Start with Godot 4.7.2 using native Metal. Profile both Mobile and Forward+ on the representative section before committing to renderer specific effects. Godot documents both as Metal capable; the [renderer comparison](https://docs.godotengine.org/en/4.7/tutorials/rendering/renderers.html) explains their feature differences. Begin with a sun light, economical shadows, simple sky, mipmapped materials, restrained vegetation, and distant geometry simplification. Terrain fidelity and physics must remain the same across graphics quality levels.

Initial performance acceptance target, not a measured result: sustained 60 FPS at a 1920 by 1200 internal render resolution during manual riding, with frame time percentiles and memory recorded over a ten minute session. Also test resizing, full screen, resume from pause, and gamepad input. Compare quality modes at the laptop's native display resolution separately rather than claiming native resolution performance in advance. Measure on battery and external power and record the power mode.

## What is sufficient and what is missing

| Area | Ready for first section? | Remaining work before a hyperrealistic claim |
| --- | --- | --- |
| Overall terrain and recognizable course layout | Yes, with explicit source uncertainty | Align road edges and height samples; validate against images. OSM is approximate and the terrain predates repaving. |
| Current visual palette and broad scenery | Yes | Match materials in engine; author near field geometry; correct camera distortion during comparisons. |
| New curb profiles and exact usable widths | Preliminary estimates only | Obtain current survey or design dimensions, or measured reference photography. Public footage is not sufficient to certify dimensions. |
| Pavement camber and local bumps | Broad shape only | Inspect original lidar points and compare repeated cross sections; use current survey for fine surface validation. One meter grid spacing is not one meter accuracy and is not curb scale detail. |
| Motorcycle appearance | Cockpit reference available | Original full bike mesh and views from other angles; do not infer hidden geometry from the onboard video. |
| Motorcycle dynamics and current grip | No validated model yet | Verify bike specifications, tire behavior, mass distribution, rider model, and handling tests. Video or asphalt appearance cannot supply friction coefficients. |

Decision: enough material exists to start a credible, attractive playable section grounded in real terrain. It is not enough to certify survey accurate track geometry or physically validated Ducati handling. Keep those gaps visible, and do not label visual plausibility as validation. No outreach requesting private survey files has been sent.

## Reproducing the local reference artifacts

Run `uv run tools/fetch_geometry.py` and `uv run tools/fetch_materials.py` from the repository. Both store source assets and reports under ignored `artifacts/reference/` and validate pinned source information. See their manifests for licensing and hashes.

For footage available locally, run `uv run tools/video_contact_sheet.py VIDEO --output artifacts/reference/VIDEO_ID --interval 10`. The script records sample timestamps and produces inspection sheets. Keep footage and derived frames out of shipped game assets unless reuse rights are established. Local inspection artifacts are separate from the public source pack.
