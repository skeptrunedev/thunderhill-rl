# Video material comparison

The acceptance target is the supplied Thunderhill FPV footage, especially the
pavement, dry grass and visible motorcycle. Generic source resolution alone is
not acceptance. Current renders still fall substantially short of that target.

## Reference and comparison method

Ken Moto's supplied video is inspected at 00:10 for cockpit materials, 00:20 for
the straight and asphalt, and 00:40 for cut grass and pavement at a lean. Original
inspection frames stay in `artifacts/reference/ken-moto/`. They are not shipped
as texture assets. `tools/compare_material_views.py` creates reference and game
sheets from explicit crop rectangles, records input SHA256 hashes, preserves
colors and avoids upsampling. It does not calculate misleading pixel similarity
scores for unmatched camera poses, lighting, motion blur or exposure.

Current local evidence: `artifacts/material-comparison-before/` and
`artifacts/material-comparison-final/`. Straight and cockpit game views are
1280 by 720. The final straight view is at station 520 and rider camera 1; the
cockpit view is at station 400 and camera 2. These are approximate viewpoint
comparisons, not registered images. A matched moving corner remains necessary.

## Findings and first changes

The existing grass and asphalt source images contain fine photographic detail.
Their Godot imports have no size cap and use lossless mode, mipmaps and
anisotropic sampling. The road's roughness and normal combination produced
conspicuous cloudy foreground grain compared with the reference. The shader now
uses a narrower source roughness range and weaker normal relief, with restrained
longitudinal variation. An earlier stronger variant visibly resembled painted
bands and was reduced before this checkpoint. The reference's exact wear and
repaving marks are not reconstructed.

The grass material previously mixed 35 percent of a rotated secondary image into
its primary color, normals and roughness. This is now 10 percent to retain more
coherent detail. An original generated cut grass albedo is saved separately in
`assets/source/material-studies/`, with its exact prompt and limitations. A real
renderer comparison swaps only albedo under identical camera and lighting via
`godot/tools/preview_grass_material.gd`. The candidate looks more like continuous
straw nearby, but does not solve distant field structure or sparse standing
grass. It remains a study because matching relief and roughness are absent.

The bike's solid red coating previously had metallic set to 0.48. It now uses a
dielectric base with clearer topcoat response. This small correction does not
solve the plainly visible cockpit gap: uniform materials, missing fine component
detail, approximate shapes and sparse instrument graphics remain unfinished.

## Clubhouse work completed before material priority correction

The official [facility photograph](https://www.thunderhill.com/track-info/facility)
shows a low open south canopy, low wings and a taller tower with two glazing
bands. `clubhouse-profile.json` replaces the previous uniform extrusion with
those broad volumes, preserving the mapped footprint by polygon intersection.
Section positions, heights, window spacing and overhangs remain estimates.
The source image is inspection only. `preview_building.gd` captures the actual
baked geometry from paddock and track sides; both were rendered. Static scene
serialization verification passed. Additional architecture work was stopped
when the user identified material quality as the priority.

## Verification and remaining acceptance

Actual Vulkan Mobile renders were inspected, including the paired grass study.
The existing cockpit readout and reservoir geometry check passes. Human controls
pass with zero failures. The macOS development export also succeeds. These tests establish function, not photographic fidelity.
No training was run. The game remains under visual development and has not met
the requested near one to one match.

## Live instrument surface and cockpit finishes

The cockpit screen now uses an original 1024 by 560 live texture with a rising
RPM scale, lap timer, speed and gear. The supplied 00:10 frame guides visual
density and hierarchy. Values come from simulation state and the current lap
timer. No Ducati firmware image or fabricated sensor values are displayed.
The viewport redraws only when a displayed value changes. The screen combines
emission with dielectric glass response in `instrument_screen.gdshader`, without
a reflection painted into the image.

The reflective material exposed front face normals being smoothed together with
the case sides. The panel now has separate flat normals and clockwise front
winding. The regression checks that all front triangle normals face the rider,
allowing for the packed normal quantization observed in the actual ArrayMesh.
Godot's pinned SurfaceTool implementation generates normals by smoothing group:
[engine source](https://github.com/godotengine/godot/blob/ed1daf0bf001b61586d9930840f2f1394092c079/scene/resources/surface_tool.cpp#L1105).

`cockpit_finish.gdshader` adds original object space microtexture to molded
plastic and the darker metal fittings. Separate roughness, metalness, grain scale
and relief distinguish the finishes. Derivative filtering fades unresolved
microtexture rather than letting it sparkle at a distance. These are estimated
surface properties, not scans or measured Ducati materials. Development
provenance includes both new shaders and the display script.

The cockpit readout test verifies speed conversion, gear and neutral, RPM,
lap time and unchanged value caching. The reflective panel normal check and
rendered human control check pass. A game cockpit screenshot at station 400,
camera 2, 1280 by 720 was inspected against the same reference frame in
`artifacts/instrument-comparison/00.png`. The planar face has no previous
triangular highlight artifact. This does not establish a complete cockpit match;
reservoir transmission, fitting geometry, housing edge profiles and many fine
features remain unfinished.

## Custom grass shader and texture integration

The original 1254 pixel cut grass study now supplies the terrain albedo. The
shader derives restrained relief and roughness from its filtered luminance,
replacing unrelated normal and roughness maps from the previous grass source.
This is an artistic approximation. Brightness cannot establish measured height
or reflectance. Geographic color gains and the road shoulder soil blend remain.

Manual mirrored UVs use a clamped sampler and explicit gradients from the
unfolded UV coordinates. This avoids opposite edge filtering and mip derivative
cancellation at mirror folds. The source still repeats with reflected symmetry
every four metres, so this does not solve medium distance repetition.

The real renderer produced twelve paired camera positions, 0.30 metres apart,
in `artifacts/grass-relief-motion/`. Production uses the new source (`existing`
in that tool); the comparison candidate is the previous Withered Grass albedo.
These sampled positions are not a continuous motion or temporal shimmer test.
Near straw detail is more continuous, but distant fields remain too smooth and
standing grass too sparse. A rider view at station 520 was also inspected.
Human controls pass with zero failures. Near one to one visual acceptance remains
unmet. No training was run.

## Breaking up grass repetition

Each shared vertex of a square world space lattice now supplies a deterministic
rotation and offset into the grass source. Four clamped, mirrored samples blend
with quintic interpolation and concentrated overlap. Shared transforms agree at
cell boundaries, and blend weight first and second derivatives vanish there.
This is necessary because the relief differentiates the blended albedo. Source
texture gradients still use the rotated, unfolded UV footprint.

The same twelve camera positions were rendered in
`artifacts/grass-smooth-patch-motion/` and compared with
`artifacts/grass-relief-motion/`. Long repeated parallel bands are less obvious;
the source still contributes broad swirls and the middle distance remains too
smooth. Sampled stills do not prove temporal stability. The surface requires
four grass texture reads instead of one. The rendered human control check passes
with zero failures on Linux. This does not establish Mac performance.

## Short cut grass geometry

The former clump combined one tall source stalk with five small source stems.
It now combines two small CC0 stems with 96 original bent ribbon blades. The
96,000 instance placements still use sampled terrain heights and normals.
Building and paving exclusions expand by 0.9 metres to cover the wider clumps.
Each mesh uses 525 triangles instead of 634, but has two material surfaces
instead of one, so triangle reduction alone does not establish lower GPU cost.

The ribbons use the generated straw texture. Their unscaled heights range from
0.020 to 0.050 metres and widths from 0.0014 to 0.0036 metres. Placement scaling
changes those dimensions. These are visual estimates, not field measurements.
Bent ribbon normals follow the surface slope. Thin material transmission uses
Godot's [backlight material feature](https://docs.godotengine.org/en/stable/classes/class_basematerial3d.html#class-basematerial3d-property-backlight),
with an estimated straw tint. There is no claim of measured plant scattering.

The bake verifies all instances and serialization of transforms and geometry.
The generated grass texture is now included in its source provenance. Because
landmark baking shares the scenery fingerprint helper, its bake was refreshed
as well.

The final riding height diagnostic is
`artifacts/cut-grass-fine/existing_00.png`. Short ground vegetation replaces the
previous isolated tall weeds; distant terrain still lacks field structure.
The rendered human control test passes with zero failures. Local frame samples
are not a Mac performance certification. The macOS development export is
validated separately, and the game remains below the requested realism target.

## Reservoir transmission

The reservoir shader now samples opaque scene radiance and attenuates it through
an analytic bounded cylinder representing the fluid. Ray intersections with the
cylinder sides and fluid top/bottom determine the path length. Estimated RGB
absorption coefficients produce the amber transmission using exponential
attenuation. A thin wall tint and grazing reflection term describe the vessel.
Fluid dimensions and absorption are appearance estimates, not manufacturer
measurements. Added outlets and feed hoses sit separately from the supports.

This is screen based transmission without refractive displacement, multiple
internal reflections or sloshing. Godot copies the screen before transparent
objects, so they are absent from the transmitted image. This limitation follows
the [engine documentation](https://docs.godotengine.org/en/stable/tutorials/shaders/screen-reading_shaders.html).
The fluid top is fixed in vessel coordinates. It is a visual material, not a
fluid simulation or an RL observation of brake fluid condition.

The real Vulkan Mobile cockpit render is
`artifacts/reservoir-absorption-final.png`, compared with video frame 00:10.
Transmission is visibly amber instead of an opaque brown surface. This improves
one material but does not resolve approximate cockpit proportions and details.

## Cockpit mounting hardware

Handlebar mounts now use slender risers and rounded collars with bolt ears and
visible sockets. Collar centres follow the existing handlebar centreline. A
transverse steering damper adds a satin cylinder, polished shaft, end collars,
mounting bridge and pivot fittings. Its final position clears the risers and
keeps the cylinder visible between handlebar and tank in the onboard view.
Dimensions and mounting shapes are estimates guided by frame 00:10.

The entire damper is currently decorative geometry attached to the steering
assembly. It does not telescope between independently moving mounts and does
not alter steering physics. Clamp split seams and exact manufacturer profiles
remain unmodeled. This is an improvement in visual structure, not validated
Ducati hardware reconstruction.

`artifacts/cockpit-hardware-proportions.png` was rendered with Vulkan Mobile at
station 400 using onboard camera 2. The comparison with the supplied footage
still shows substantial differences in tank shape, surrounding components and
lighting. Cockpit readout and human control checks pass.

## Pavement edge material

The asphalt shader now darkens a narrow irregular band at the pavement edge,
guided by the strip outside the white line in frame 00:40. It uses 0.08 to
0.15 metre estimated material widths, not surveyed seam dimensions. Noise
coordinates wrap around a circle so this new band is continuous at the lap
seam. Existing longitudinal pavement noise is unchanged.

The pavement mesh already stores station and lateral distance in its primary
UVs. An optional secondary UV carries full road width interpolated from those
stations. The shader therefore follows both sides through widening sections.
No road vertices, contact triangles, paint positions or grip values changed.
The triangle helper regression verifies secondary attributes stay attached
through winding changes while rendered geometry and primary UVs remain equal.
The contact regression passes 202 checks; rendered human controls pass with
zero failures. Both affected scenery manifests were rebuilt and verified.

The real renderer close view is `artifacts/pavement-edge-close/existing.png`,
produced with the new `--road-edge` inspection option. The strip is visible
between paint and shoulder. The comparison still shows an overly coarse rocky
shoulder and simplified distant field structure, so this is not a complete
pavement and grass appearance match.


## Original fine shoulder material

The generic rocky soil is replaced by an original 1254 pixel square fine dust
and straw albedo, guided by frame 00:40. `terrain.gdshader` now shares rotated,
smoothly blended world space patches between grass and soil. Both use explicit
filtered gradients and clamped mirrored source coordinates. Soil roughness and
1.5 mm relief are artistic luminance estimates, not measured PBR data. The
unrelated generic soil normal and roughness maps are no longer sampled.

`artifacts/original-shoulder/existing_00.png` uses the same camera as the earlier
`artifacts/pavement-edge-close/existing.png`. The coarse stones are gone and the
shoulder reads as fine earth. The reference sheet is
`artifacts/original-shoulder/reference-comparison/00.png`. This compares material
character, not identical framing: the reference is leaned and motion blurred.
Broad field structure, shoulder wear and the smooth road boundary still fall
short of the video. Three sampled camera positions do not establish temporal
stability during a full lap.

Shader review found no correctness defect. The generalized soil sampler adds
one net texture lookup (10 versus 9) and four patch transforms per fragment.
Rendered human controls passed with zero failures, median 17.116 ms and p95
17.361 ms across 274 Linux frames. This is a local smoke measurement, not a Mac
performance claim. Gameplay geometry, collision and agent observations are
unchanged.


## Pavement binder and paint edge response

The seam now varies binder strength along its length and adds fine aggregate
variation in world space. The broad coordinates remain periodic at the lap join.
Fine contrast fades toward its centered mean when its screen footprint is too
large, rather than changing the mean through a thresholded noise signal.

Edge paint has an independent opt in material flag for estimated 0.2 to 2.8 mm
boundary chips. World space pixel footprint and transverse marking footprint
both suppress unresolved detail. Discard occurs after all shader derivatives and
texture sampling. Curbs and other painted objects retain the default disabled
flag. Geometry, paint mesh position and width, and collision are unchanged.

The fixed camera capture is `artifacts/edge-grain/existing_00.png`. This is a
subtle close surface refinement. It does not solve the much larger distant
terrain and scene realism gap. Three captures are not a full motion stability
assessment. Contact regression passed 202 checks, and rendered human controls
passed with zero failures. Local frame timing was median 17.299 ms and p95
21.040 ms across 271 frames. No Mac runtime performance inference is made.


## Field structure diagnostic

The existing historical detail map contains field tracks, but its sampler used
isotropic mip filtering. Both terrain gain maps now opt into the project's
existing anisotropic filtering. Comparison of the same 1830 metre camera in
`artifacts/edge-grain/existing_00.png` and
`artifacts/terrain-anisotropy/existing_00.png` shows only a small distant change.
This corrects the sampler choice but does not explain or fix the uniform field.

A new `detail_gain_exponent` appearance control defaults to 1.0, preserving
source gain values. The inspection CLI accepts `--detail-gain-exponent` between
0.5 and 3 and `--station` within the lap, records both requested and sampled
station, and rejects invalid numeric inputs. This makes material studies
reproducible at different track positions without editing production settings.
The exponent 2 study in `artifacts/terrain-detail-contrast/existing.png`
exaggerated broad streaks rather than adding convincing grass structure, so
it was not adopted as the default. Source images and terrain data are unchanged.

Additional default contrast captures at stations 400 and 3000 are in
`artifacts/terrain-station-400/existing.png` and
`artifacts/terrain-station-3000/existing.png`. Both reveal a repeated smooth
shoulder and sparse dark stems against the ground. This points to vegetation
coverage and material blending as the next investigation, rather than a global
contrast increase. These are inspection views, not matched video poses.


## Short stubble and broken shoulder blend

The grass clump now contains 132 original bent ribbons, replacing 96 ribbons
and two taller atlas stems. That is 528 triangles rather than 525, with one
material surface instead of two. Placement still uses the separate world RNG;
all 96000 positions remain authored by the existing placement procedure. Roots
are embedded 3 mm rather than 15 mm. The old depth could fully bury the shortest
scaled blades (minimum vertical tip height 9.1 mm on level ground).

Terrain material coverage varies within the existing grass and soil transition
using smooth world space noise at estimated 2.4 metre and 0.35 metre scales.
The original fully grass and fully soil endpoints are preserved. Noise contrast
fades with pixel footprint. Material height derivatives are calculated before
blending their gradients, so the footprint dependent coverage is not itself
differentiated into a ridge. This is original decorative variation, not a
surveyed vegetation map. Surface friction and track contact are unchanged.

The intermediate capture is `artifacts/stubble-transition/existing.png`; the
final station 400 view with corrected burial is
`artifacts/stubble-final/existing.png`. The isolated long stems are reduced,
but this is still a modest refinement of the overall scene. Sparse stubble and
large scale terrain structure remain visibly unlike the onboard footage.

The rendered human controls check passed with zero failures. Local Linux timing
was median 17.289 ms and p95 18.231 ms over 274 frames. Bake serialization checks
preserved all 96000 grass instances. This does not establish native Mac runtime
performance or a complete photographic appearance match.

## Field tint and coverage studies

The preview CLI now accepts `--ground-tint=R,G,B` (linear channels greater than
zero and at most one) and `--field-soil-strength=0..1`. It records their values
and hashes of the terrain shader and both aerial gain maps. Invalid component
counts, nonnumeric channels, zero channels and out of range coverage were
confirmed to exit with status 2.

The neutral tint trial `(0.59, 0.57, 0.51)` is preserved in
`artifacts/neutral-field-study/existing.png`; it was not adopted. Median encoded
RGB from a field crop in frame 00:40 (180,250 to 400,350) was (142,114,83), while
the station 400 game crop (180,420 to 400,520) was (191,152,110). Their normalized
RGB ratios are similar. These unmatched patches do not establish calibrated
albedo, but do not support blaming a global hue shift for the realism gap.

An optional shader trial blends fine soil into darker broad aerial regions.
Its mask is `1 - smoothstep(0.65, 1.0, macro_luminance)` and its strength is
bounded by the CLI. Missing neutral aerial gain adds no soil. This is an artistic
brightness to coverage hypothesis, not a vegetation classification, and changes
neither contact geometry nor friction. It adds no texture samples.

At strength 0.65, captures in `artifacts/field-coverage-study/` and
`artifacts/field-coverage-3000/` showed almost no improvement. Changes were mainly
confined to distant patches; the uniform foreground remained. **Production
strength is zero.** The default was rendered again in
`artifacts/field-study-default/`, and its metadata confirms zero strength.
The study is retained for reproducible comparisons, not presented as a visual
upgrade. Broad field structure, foreground texture character and matching the
moving reference camera remain unresolved.

## Flattened straw texture revision

The original generated v2 albedo replaces v1 for both terrain and bent stubble
ribbons. The exact prompt and reference roles are recorded in
`assets/source/material-studies/README.md`. Its 1254 square source contains
larger flattened straw bundles and connected fine earth openings rather than
the previous nearly uniform tangled mat. Runtime scale remains two metres and
the existing filtered luminance relief shader is retained. These are original
appearance estimates, not scans or surveyed straw dimensions.

Matched existing and candidate views at stations 400 and 3000 are in
`artifacts/grass-v2-400/` and `artifacts/grass-v2-3000/`. The candidate has more
readable straw patches in the foreground at both locations. It does not resolve
the distant uniform terrain or the overall scene realism gap. Stronger bundles
may expose mirrored patterns, so sampled stills are not proof of full lap
temporal stability. Final production captures are in `artifacts/grass-v2-final/`.

The source and runtime copies have SHA256
`d9799490c3e2fe24a52905a80ccf2713292b354d92cd4cd4eb116d92ffd20c02`.
Import is lossless with mipmaps and no size cap. Both scenery manifests were
rebuilt and serialization checks preserved all 96000 grass instances. Human
controls passed with zero failures; local Linux frame times were median
17.049 ms and p95 17.361 ms over 275 frames. This is not native Mac performance
certification. No contact geometry, physics parameters or agent interface changed.

## Historical pavement tone study

`tools/build_pavement_tone.py` creates a 64 by 2048 numerical gain map from
the pinned July 2022 NAIP source. Sampling follows the pavement's original
triangle UV mapping and centered edge tangents, including the closed lap.
Source pixels are classified before bilinear resampling so rejected paint or
soil colors cannot bleed into accepted pavement. Normalized smoothing uses a
neutral median prior. RGB rejection is not semantic segmentation; shadows can
survive. Source, export metadata, track, generator and output hashes are recorded.

The asphalt shader can apply this map without changing road geometry, contact
or friction. The outer strip fades to neutral. Matched captures at stations
400, 900 and 3000 are in `artifacts/pavement-tone-study/`, made with
`godot/tools/preview_pavement.gd`. Each pair preserves the same camera and
lighting and compares strength zero with 0.65.

The visual change is small: station 400 is nearly indistinguishable, station
900 has a mild broad tone shift, and station 3000 has a soft patch near the
bend. Independent visual review agreed that this does not establish improved
reference fidelity. No obvious seams or colored contamination were visible.
**Production strength remains zero.** The historical layer is retained as a
reproducible optional study, not presented as a realism upgrade. It predates
the repave and does not reproduce current longitudinal wear or grazing response.

Four Python regression tests cover rejected colors, rejection before sampling,
periodic filtering and actual triangle mapping. All passed. The 202 curb
surface checks passed. Remaining material work needs footage matched camera
views and authored color, relief and roughness, with temporal checks at speed.
The overall scene is still visibly unlike the onboard reference.

Final human controls passed with zero failures. Linux frame times were median
17.202 ms and p95 23.344 ms over 263 frames. The Mac export completed as build
`291bd93c36e8-a6956c4bf8e1`; this is a packaging check, not a new native Mac
runtime or visual acceptance test.

## Original asphalt surface study

The original `racing-asphalt-v1.png` albedo replaces the source character of
the generic aggregate. Its source prompt and limits are documented in
`assets/source/material-studies/README.md`. Shader controls separate fine
aggregate color, shallow estimated relief, broad longitudinal variation and
roughness. Initial gain 1.0 was too bright; matching the previous material's
average linear color required gain 0.30 with restrained contrast. The first
roughness trial at 0.56 produced excessive broad glare; 0.62 reduced it.

Paired studies are in `artifacts/authored-asphalt-v1/`,
`artifacts/authored-asphalt-v1-tuned/` and
`artifacts/authored-asphalt-v1-roughness/`. The last samples twelve positions
at 0.5 metre intervals at each of three stations. Independent inspection found
better foreground grain but also a curved filtering boundary near station
3005.5. Disabling relief left the boundary unchanged, as did unlit rendering.
The decisive isolation was variation only: it reproduced the boundary without
the texture, while raw albedo did not. The floating point sine hash in the
procedural variation was responsible. Integer lattice hashing removed the
boundary at stations 400 and 3000 while retaining the foreground grain.
`artifacts/authored-asphalt-integer/` contains the corrected comparisons.
The exact compiler arithmetic behind the sine hash discrepancy is not proven.
An earlier fixed mip trial masked the boundary; it did not establish a texture
filtering cause and is not used in production.

The preview tool supports `--candidate`, `--frames`, `--offset-m`,
`--flat-relief`, `--unlit`, `--fixed-mip`, `--linear-mips`, `--world-uv`,
`--raw-albedo` and `--variation-only`, recording those
settings, source hashes and each camera pose. Diagnostic shaders are copies;
they do not change production render modes. Sampled stationary frames are not
proof of full lap temporal stability.

Separately, the material's mipmaps are baked in linear light, then encoded
back to sRGB. This preserves average color at distance. The original mean
linear RGB is approximately (0.1052, 0.1007, 0.0956); averaging encoded sRGB
first yields (0.0886, 0.0841, 0.0810) after decoding. The baked ImageTexture
contains the corrected chain, with source, builder and output hashes recorded
in its JSON manifest. The source PNG remains unchanged. Regression tests verify
black/white averaging, alpha, source preservation and image dimensions. This
brightness correction is independent of the procedural seam fix.

The final production capture is `artifacts/authored-asphalt-production.png`,
at station 3005.5 with rider camera one. It loads the baked runtime resource,
not the preview's external image. The curved boundary is absent and fine grain
remains visible. Both scenery manifests were rebuilt with unchanged geometry
fingerprints. Human controls passed with zero failures; Linux median frame
time was 17.242 ms and p95 21.801 ms over 267 frames. Mac packaging passed as
`7434768d3780-a7890d635c6e`. No fresh native Mac performance or full lap motion
acceptance is claimed. No collision, physics or agent protocol was changed.

## Ground brightness, scale and sparse coverage

The controlled station 400 studies in `artifacts/grass-value-baseline/` and
`artifacts/grass-value-dark/` isolate a 0.65 multiplier on the existing linear
ground tint. This changes brightness, not the hue ratios tested earlier. The
darkened field reads more brown and less uniformly golden, closer in character
to the inspected footage. Because the viewpoints and lighting are not precisely
matched, this is an artistic value adjustment rather than calibrated albedo.

`artifacts/grass-value-soil-3000/` and `artifacts/grass-value-scale/` compare the
same surface at two metre and one metre grass scales. The smaller scale reduces
oversized wiry foreground strands and is adopted. The source texture is unchanged.
Independent visual review found brightness and scale the strongest improvements.

The terrain shader now supports interrupted elongated sparse straw patches,
with appearance strength 0.5, and a separate soil albedo value multiplier 0.7.
These add restrained soil contrast but do not reconstruct measured mowing paths.
The two noise scales and their orientation are original appearance estimates.
Neither changes collision, friction or the preexisting shoulder footprint.
Coverage remains within zero to one, and existing bare shoulders remain bare.
Relief parameters and fine source textures are unchanged.

The shared `DRY_GROUND_TINT` also colors the standing stubble. Both scenery
bakes were regenerated so it does not retain the earlier brighter tint. The
production rider view is `artifacts/grass-field-production.png`. Human controls
passed with zero failures. Local Linux frame time was median 17.349 ms and
p95 22.565 ms over 265 frames. These are not new native Mac results: the user's
MacBook was offline in the Tailscale check during this pass.

The preview tool records and validates `--grass-tile-m`, `--soil-value` and
`--field-patch-strength`, in addition to its existing controls. The updated
scene is still visibly simpler than the footage; these material improvements
do not establish photographic realism or a complete track appearance match.

Final close views are `artifacts/grass-field-final-close/existing_00.png`
through `existing_05.png`, sampled 0.30 metres apart. The first and last were
inspected with the rebuilt stubble tint. Thin dark stubble remains conspicuous
in places, so this does not resolve all vegetation appearance issues or certify
motion stability. Invalid scale, nonfinite soil value and out of range patch
strength exited with status 2. Mac export passed as
`16d5c95a5f56-631e58d2fc0b`.

## Custom stubble canopy shader

The controlled pair in `artifacts/stubble-canopy-study/` compares the prior
StandardMaterial3D ribbons with the original `dry_stubble.gdshader` at station
3000. Texture, geometry, camera and lighting are held fixed. Both root and
independent visual review found fewer isolated dark stems near the camera and
less distant stippling. This supports changing ribbon lighting, but does not
establish a measured physical model of dry stalks.

The shader orients the ribbon normal toward the local canopy hemisphere and
blends it 0.65 toward the transformed tuft up direction. It retains the original
grass texture, shared ground tint, instance color, roughness and backlighting.
The blend is an artistic approximation for unresolved short vegetation. The
tradeoff is flatter individual blade shading. It does not add geometric detail
or reconstruct the field variation visible in the video.

Scenery serialization now verifies ShaderMaterial source, effective uniforms
and external texture hashes. Unsupported material dependencies fail explicitly.
A focused rendered test verifies uniform, default, source and texture changes,
and preservation across saved scene reload. Both scenery bakes passed. Human
controls passed with zero failures; local Linux frame times were median
17.327 ms and p95 25.0 ms over 269 frames. These short runs do not certify full
lap stability or native Mac performance. The MacBook was offline in this pass.

The final baked shader was inspected at station 400 in
`artifacts/stubble-final-close/existing_00.png` and `existing_05.png`, with
camera positions 1.5 metres apart. Grass remains visible without the previous
concentration of dark sticks, though some distant stippling remains. The local
comparison sheet `artifacts/stubble-video-comparison/00.png` places the supplied
video frame beside this render without color correction. Its different camera
and position are labeled explicitly. The sheet makes the remaining excessive
uniformity, simplified architecture and lighting mismatch clear; this pass is
not a 1:1 result. Video frames and comparison sheets remain local artifacts.
Mac export passed as `3cd028be5f4b-70c27e42dfa9`; it has not been run on the Mac.

## Broad pavement material variation

The supplied video frames at 00:20 and 00:40 show organized longitudinal tone
and grazing reflections, while the current game emphasizes evenly distributed
fine aggregate. The controlled `artifacts/pavement-sheen-study/` varied only
roughness by up to 0.16. That produced bright blue and white strips resembling
wet pavement and was rejected.

`artifacts/pavement-band-study/` instead tested a smooth broad diffuse band with
maximum albedo reduction 0.32. Its effect was slight. The stronger 0.50 study in
`artifacts/pavement-band-strong/` was inspected at stations 400, 900 and 3000,
with three camera positions at each station. Root and independent visual review
found a modest improvement in the dark interior region, clearest at 900 and
3000, without the wet strips. The stronger diffuse band is adopted. Correlated
roughness changes remain small: at most 0.025 downward and 0.018 upward.

The band uses normalized road width and a periodic station domain. Width,
lateral center, drift and amplitude are artistic estimates, not surveyed rubber
deposits, an optimal racing line, or measured binder reflectance. One similarly
placed continuous band is still simpler than real pavement wear; increasing
its contrast cannot solve that structural limitation. Aggregate texture and
relief remain unchanged. Collision, friction and the agent protocol are unchanged.

`preview_pavement.gd --surface-variation=1` renders matched baseline and candidate
views using the actual baked production texture. It rejects incompatible image
or lighting diagnostics, validates strength, and records camera transforms,
shader hash and runtime texture hash. Other study modes explicitly disable this
new variation to preserve their earlier comparison meaning. Nonfinite strength
was checked and exits with status 2. These sampled stills are not proof of full
lap motion stability or a calibrated reference match.

The actual production rider capture is `artifacts/pavement-band-production.png`.
It was inspected after changing the default strength to one. Human controls
passed with zero failures; the local Linux run measured median 17.361 ms and
p95 18.545 ms over 267 frames. No native Mac or full lap acceptance is implied.
Mac export passed as `ac6e4cfc9fe5-cf123d9972de`. The final controlled pair in
`artifacts/pavement-band-final/` records the production texture checksum and
correct zero/one variation strengths at all three stations; these metadata
values were verified against the actual file. Native Mac testing remains pending.

## Grass direction diagnostic

`artifacts/grass-direction-study/` holds a matched station 400 pair using the
same source image with full random patch rotation versus no rotation. The
unrotated candidate looks flatter and does not restore the field structure
seen in the footage. Production retains full rotation. The new
`grass_rotation_spread` control changes only grass sampling; soil retains its
previous transforms. `--candidate-grass-rotation=0..1` applies only to the
candidate capture and records both values. This isolates texture direction
from another source image, broad aerial contrast, or a global color change.

The built in imagegen v3 source tests shorter straight stems and basal clumps.
`artifacts/grass-v3-study/` compares it with v2 at the same station 400 view.
It removes some sweeping wiry strands, but the field remains equally uniform
and looks flatter overall. It is retained as a source study, not installed as
the game material. The prompt and source checksum are recorded beside it.
Human controls passed after the direction diagnostic change with zero failures.

Inspection of the aerial and builder identifies a separate structural omission:
terrain detail alpha represents distance to the racing surface, not field cover.
Pale access corridors in the aerial can therefore receive full grass and stubble.
Explicit historical material regions are the next investigation; RGB contrast
or additional random texture variants cannot supply those missing regions.

## Explicit historical field corridor

The 2022 aerial shows a pale connected strip west of the circuit near station
4264.549 m. `data/reference/field-corridor-study.json` records six manually
traced pixel centers, the original image checksum and extent, crop bounds,
datum operation, local coordinates and uncertainty. The source crop was
visually reviewed. All local coordinates were independently recomputed from
pixel centers through the inverse datum transform and agreed within 0.00001 m.
This verifies the conversion, not survey accuracy. The estimated width is six
metres with 1.8 metres of interpretive uncertainty; source georegistration error
is additional. Compacted soil, gravel and sparse vegetation cannot be reliably
distinguished in this image. The junction and western merge are not reconstructed.

`godot/data/field-coverage.json` carries the reviewed trace with its source
reference checksum. `field_coverage.gd` validates the data, supplies the shader
segments and excludes grass clumps from the same corridor core. The terrain
shader uses smooth capsule distances with a 1.2 metre artistic feather. It
replaces grass with the existing fine soil material. The corridor soil uses a
value multiplier of one, while ordinary shoulder soil retains 0.7. No collision,
contact height, friction, road boundary or training interface changes.

`artifacts/field-corridor-study/` isolates the ground material at rider height.
Both captures share the newly baked stubble exclusion, so they do not establish
its visual contribution. The candidate exposes earth instead of straw, though
this close view reads as a broad clearing more than a connected access route.
A wider overview is used to inspect the corridor boundaries and alignment.
The preview supports explicit side and eye height, recording those camera
settings; these diagnostic viewpoints are not claimed to match the onboard video.

Validation checked capsule widths and endpoint caps, malformed data, shared
shader parameters and every baked grass instance. All 96000 instances were
inspected with zero corridor intrusions. Both scenery bakes passed. Human
controls passed with zero failures; local Linux timing was median 17.241 ms and
p95 17.646 ms over 271 frames. The 202 curb and surface checks passed. Native
Mac testing remains pending because the MacBook is offline. The scene remains
visibly simpler than the footage; this is one mapped material region, not a
complete land cover reconstruction.
The 20 metre eye height pair in `artifacts/field-corridor-overview/` reveals
both boundaries and the rounded traced endpoint near the circuit. The connected
strip is distinguishable at this wider scale. Its uniform soil treatment and
soft edge are still simpler than the aerial, and its deliberately untraced
junction remains visible. The overview is a diagnostic view, not a riding camera.
Mac export passed as `a620ac1f0d56-223165bf3cd1`; native execution is unverified.

## Smoked reservoir walls and cap profile

The current cockpit audit (`artifacts/cockpit-current-audit/production.png`)
shows clear pale reservoir walls compared with the dark smoked vessels in the
supplied 00:10 frame. The existing shader used absorption (3, 4, 7) over a
nominal 1.5 mm wall path, so it transmitted almost all background light at
normal incidence. The fluid tint alone could not darken the empty upper vessel.
The closed mesh winding and outward normals were checked; reversing geometry
would not address this material discrepancy.

Wall absorption is now independent of fluid absorption. Matched camera studies
used (300, 400, 700) in `artifacts/cockpit-smoked-wall/` and (600, 800, 1400) in
`artifacts/cockpit-smoked-wall-dense/`. Root and independent visual review found
the latter closer to the reference's smoked appearance, particularly in the
upper chamber. It is adopted as an appearance estimate, not measured polymer
absorption. Fluid absorption, fill level and lighting are unchanged.

`artifacts/cockpit-cap-profile/` isolates a taller grip sidewall with the first
tint. The cap height increases from 8 to 14 mm while retaining its lower mating
face and radius. It reads more like the prominent molded screw lid in the
reference. This dimension is an artistic estimate, not a manufacturer part
measurement. Mesh bounds, closed edges, signed volume and outward normals pass
the existing geometry checks with the updated height bounds.

The combined final result is `artifacts/cockpit-reservoir-final/production.png`.
It was visually inspected after adopting both changes. The scene still lacks
refraction: background paint lines pass through the vessel without bending.
The mirrored mounts, hardware detail and overall cockpit fidelity remain
incomplete. These changes do not establish photographic realism.

The cockpit tool accepts `--reservoir-wall-density` as a multiplier on the
shader default. Captures record the effective coefficients and shader checksum
for each camera pose. The value is validated as finite and within 0 to 200.
Cockpit geometry/readout checks and human controls passed. Local Linux frame
times were median 17.361 ms and p95 21.750 ms over 269 frames, not native Mac
performance measurements. No physics or control interface changed.
Nonfinite wall density exited with status 2. Mac export passed as
`39c83c9a0021-9fa84dbcfac3`; it has not been run natively on the MacBook.

## Cockpit hardware edge curvature

The yoke, four clamp ears and central damper bridge now use the existing
rounded box builder in place of planar chamfers. Outer dimensions, positions,
materials and corner radius values are retained. The production camera render
in `artifacts/cockpit-hardware-fillets/production.png` shows continuous highlights
around these corners, closer to manufactured hardware in the supplied 00:10
frame. Root and independent visual review both found this a local improvement.
The bridge remains bulky and the overall cockpit silhouette is not a measured
reconstruction. Geometry changes cannot substitute for correcting those shapes
or matching the reference materials.

Existing rounded mesh checks passed for bounds, closed edges, normals and
volume. Cockpit readouts, rider visibility and human controls passed. Local
Linux frame timing was median 17.361 ms and p95 22.277 ms over 266 frames.
These are not native Mac measurements.
Mac export passed as `a869083b5357-ca4c44fda88b`; native execution remains
unverified for this build.

## Longitudinal pavement mottling and band breakup

The 00:20 reference contains irregular longitudinal tone and a narrow wandering
dark mark. The 00:40 reference also shows elongated variation, but does not
establish that the same mark continues around the circuit. The production
shader's dominant feature was an uninterrupted broad band, with only weak
smaller variation. Root and independent image inspection identified that
uniformity as a mismatch. Neither the marks nor their physical cause can be
measured from these compressed moving frames.

The custom asphalt shader now adds two scales of longitudinal albedo variation
and varies the existing band's center, width and opacity over shorter distances.
Both new fine domains are annular, closing at the lap boundary; their radial
offset follows cross track distance. Derivative filtering fades unresolved
noise cells toward their mean. No texture allocation or draw call is added.
The scales and strength are original artistic estimates, not a tire history,
surveyed construction map, recovered reflectance or optimal racing line.

`artifacts/pavement-mottling-study/` tried albedo variation alone at strength
0.65. `artifacts/pavement-broken-band-study/` adds band breakup at 0.85.
Root and independent review prefer the latter to the unchanged baseline,
especially at stations 900 and 3000. It reduces the smooth ribbon appearance
without an obvious cloudy or wet effect in these captures. The improvement at
400 is subtler. The narrow mark in the video remains absent and the broad
lighting gradient still dominates. Roughness constants are retained, but the
existing small roughness term follows the altered band weight, so this is not
strictly an albedo only experiment.

`artifacts/pavement-mottling-filtered/` contains baseline/candidate captures at
six positions spaced 0.5 metres apart at each of three stations. Root inspected
selected filtered captures after adding derivative filtering. These sampled
stationary views are not a continuous motion or full lap flicker test. The
candidate explicitly sets 0.85; the subsequent default change from 0.65 to
0.85 adopts that captured value without changing the shader equations.

The preview tool validates `--binder-mottling` and excludes simultaneous
texture/lighting diagnostics or a second material comparison. Metadata records
actual material parameters rather than inferring them from labels. A nonfinite
value exits with status 2. Human controls passed with zero failures. Linux
timing was median 17.361 ms and p95 22.440 ms over 265 frames, not a native Mac
measurement. Physics, grip and the agent interface are unchanged.
Mac export passed as `3e5adf905695-22da0d631573`; native execution of this build
is unverified.

## Edge marking proportion

The previous paint mesh was 0.12 m wide. The footage at 00:40 shows a broader
looking cream white strip, although camera lean and proximity prevent deriving
its metric width. The aerial source does not resolve paint width either.
`artifacts/edge-paint-width-study/` compares 0.12 and 0.20 m meshes with identical
production pavement, lighting and cameras at stations 400, 900 and 3000.
Root inspected the 900 pair and 3000 candidate; independent review inspected
all three pairs. Both reviews prefer 0.20 m as a modest improvement: the near
markings read as painted strips rather than thin outlines without appearing
excessively wide. This is an artistic estimate, not a dimensional correction.

The outer paint edge remains 0.06 m inside the pavement boundary, so the added
width extends inward. The reference's dark outer margin is still more apparent
than in these centered game views. Paint wear now uses the actual mesh width
as a shader parameter. Curbs, contact surfaces, grip and agent boundaries are
unchanged. The preview records paint width and the track and paint shader hashes,
and isolates marking comparisons from other material diagnostics.

Both scenery bakes passed after refreshing their track source fingerprints.
The 202 curb surface checks and human controls passed with zero failures.
Local Linux timing was median 17.146 ms and p95 26.314 ms over 261 frames.
This is not a native Mac benchmark. Tailscale reports the MacBook offline,
so native visual and performance verification remains pending.
Mac export passed as `ba2c565290ba-cb2489b1bbc0`. Nonfinite preview width
correctly exits with status 2.

## Field relief experiment

A custom field normal layer was compared at 0.06 m amplitude near station 900
and 0.10 m at station 400. These are artistic shading amplitudes, not measured
terrain displacement. The shader differentiates continuous height before applying
coverage and pixel footprint weights to avoid artificial material boundary ridges.

The captures in `artifacts/field-relief-study/` and
`artifacts/field-relief-close/` show almost no useful improvement at riding height.
The production default therefore remains zero. Raising relief does not address
the field's uniform land cover. Reference frame 00:40 instead shows pale straw
areas separated by darker flattened strips, including a band following the curve.
Those spatial patterns should guide the next material study. Motion blur and
camera exposure prevent treating the frame as a calibrated albedo or height map.

The preview now supports an isolated `--candidate-field-relief` comparison using
the production texture without loading an unrelated candidate image. The cleaned
command was rendered successfully in `artifacts/field-relief-final-check/` and
visually inspected. Human controls passed with zero failures; local Linux frame
timing was median 17.361 ms and p95 20.1 ms over 265 frames. This diagnostic does
not establish native Mac performance or a visual match to the footage.
