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

## Turn 2 flattened straw pattern

The 00:40 footage shows darker curved streaks in tan vegetation inside Turn 2.
The historical aerial crop `[1000,1720,1750,2160]` shows curved mowing patterns,
but does not independently establish this same dark strip. The new pattern is
an original visual reconstruction from the footage. It is not an aerial trace,
surveyed feature or recovered material composition.

`data/reference/turn2-field-band-study.json` records image hashes, construction,
coordinates and all artistic parameters. Eleven segments follow actual track
samples near stations 990 through 1265, offset four metres from the left pavement
edge. Width varies from zero at the ends to at most 1.6 m, with 0.8 m feather.
The shader retains ground straw and modulates its linear color along the curve.
Pale gaps interrupt dark stretches. Both the tint and dimensions are estimates.
The original pale historical access corridor retains its previous material.

The field coverage helper now supports per point widths, linear color, ground
grass retention and per point tint strength. Terrain shading and baked upright
grass exclusion share the same tapered footprint. Tint variation does not alter
that footprint: the pale gaps remain flattened ground. No collision, contact
height, tire grip, reward or observation contract is changed.

`artifacts/turn2-band-first/` showed a broad continuous dark path and was rejected.
`artifacts/turn2-band-tapered/` improved width but still resembled a dirt path.
`artifacts/turn2-band-final/` adds intermittent color and rebuilt stubble. Root
and independent inspection accepted the final pattern as a modest improvement.
Three paired captures spaced 0.30 m apart isolate the material; both sides use
the same rebuilt vegetation. They are not a complete vegetation before/after
comparison or a continuous motion test.

`artifacts/turn2-band-comparison/` includes a footage comparison with unaltered
colors. Camera position, lean, sun, exposure and motion blur are not registered.
The field still lacks the reference's wider pale directional straw patterns,
and the complete scene remains substantially below the realism target.

Both scenery bakes passed. Field coverage tests passed, including defaults,
taper interpolation, material parameters and malformed data rejection. The
rendered baked scene check inspected 96,000 grass instances and found zero
intrusions under the shared exclusion rule. Human controls passed with zero
failures. Linux timing was median 17.314 ms and p95 25.0 ms over 264 frames;
this was a local functional run, not an isolated native Mac benchmark.
Tailscale reports the MacBook offline, so native verification remains pending.
Mac export passed as `39c6c2f4157a-8f429689d9e2`; native execution is unverified.

## Broad pale straw studies and placement diagnosis

The next grass study tested broad color and coverage separately from fine
source detail. `artifacts/pale-straw-swaths/` added directional brightness noise;
root and independent review rejected it because it resembled lighting blotches
and striped the distant hill. `artifacts/pale-straw-coverage/` combined a paler
color with retained grass coverage but was superseded by the bounded test.

`artifacts/pale-straw-bounded/` limits the treatment to an artistic Turn 2 ellipse
(center X/Z 200/690 m, radii 210/100 m, smooth fade from radius 0.8 to 1.0).
It uses a modest linear RGB multiplier (1.10, 1.13, 1.18) and restores at most
half the grass coverage removed by the existing sparse patch layer. The existing
bare shoulder remains bare. This fixed the distant hill artifact, but the visual
improvement was too weak to justify adoption. `straw_swath_strength` remains zero;
the disabled branch skips its noise work. These parameters are not measured
vegetation, reflectance or mowing data.

An original imagegen source, `assets/source/material-studies/dry-straw-swaths-v4.png`,
then tested broader connected straw and earth patterns. Its exact prompt and
provenance are recorded beside it. `artifacts/straw-v4-four-metres/` made patches
readable at riding distance, but had oversized foreground stems. The two metre
trial in `artifacts/straw-v4-two-metres/` reduced stem size but showed a repeated
woodgrain pattern across three camera positions spaced 0.30 m apart. Neither
source variant was adopted.

`artifacts/straw-v4-no-relief/` retains this pattern with grass relief set to zero.
`artifacts/straw-v4-aligned/` instead removes random rotation while retaining
normal relief and the two metre scale. The curved pattern becomes parallel rows.
Both root and independent review therefore identify orientation of this strongly
directional source as a major contributor, not the derived normal layer. This
does not establish a mipmap defect. The aligned result remains too regularly
striped compared with the interrupted unequal swaths in frames 00:30 and 00:40.
The next correction must address source repetition and placement rather than
increasing relief or brightness. Production retains v2 at its existing scale.

The preview now accepts candidate specific tile size and grass relief, keeping
baseline settings unchanged and recording both in metadata. The swath diagnostic
uses the production texture and refuses simultaneous candidate changes. All
new numeric controls validate finite bounded inputs.

The aligned straight check at station 400 is in
`artifacts/straw-v4-aligned-straight/`; root inspected this additional view.
Human controls passed with zero failures. Local Linux timing was median
17.361 ms and p95 21.326 ms over 267 frames, not native Mac certification.
Mac export passed as `34d12c446b1f-6dea2d7e8791`; native execution is unverified.

## Scale variation and separate fine straw component

The scale diagnostic adds `grass_scale_spread`, disabled by default. Each shared
lattice vertex owns a deterministic source scale ranging from 0.55 to 1.65 at
full strength. The scale applies around the patch anchor, and the same multiplier
is passed into both texture gradients for mip selection. Neighboring cells share
these transforms. The preview exposes `--candidate-scale-spread`, validates finite
values from zero to one, and records both baseline and candidate values.

`artifacts/straw-scale-control/` and `artifacts/straw-scale-jitter/` isolate the
change. Compare candidate images across the two directories: both use v4 at two
metres, zero rotation, station 900 and identical camera settings. Each contains
three positions spaced 0.30 m apart. Scale variation disrupts individual marks
but the field still resembles ridged sand or woodgrain. Root and independent
review rejected adoption. Two or three still images do not establish temporal
stability, although no gross popping was apparent in the inspected frames.

`assets/source/material-studies/fine-straw-v5.png` removes broad soil holes from
the authored fine layer. `artifacts/fine-straw-v5-turn2/` and
`artifacts/fine-straw-v5-straight/` compare v5 against production v2 at one metre
with otherwise identical shader settings and geometry. Root inspected both
locations; independent review also inspected Turn 2. Shorter straighter fragments
reduce obvious patterns, but the middle distance becomes too uniformly brown.
V5 remains a candidate component. Production retains v2. The next material work
must author coherent broad straw coverage separately, not restore the broad
holes to every small texture tile. Foreground streaks in the video include
motion blur and cannot be treated as a static texture measurement.

The real Vulkan Mobile renderer compiled and captured all four studies without
reported shader errors. The human control test passed with zero failures; local
Linux timing was median 17.261 ms and p95 18.790 ms over 267 frames. Invalid scale
inputs `nan`, `inf`, negative values and values above one were rejected with exit
code 2. These are local functional checks, not native Mac performance evidence.
Mac export passed as `d706061342e3-4e93ab2a4107`; this export has not been
executed on the native Mac. The scene still falls substantially short of the
requested photographic realism.

## Curved field coverage adopted at Turn 2

The next material iteration separates fine straw from broader mowing appearance.
The existing retained straw corridor guides coordinates along the corner and
inward from its curved boundary. Bare access corridors do not seed the pattern.
The current dataset contains one retained straw chain. This is an original visual
construction, not a trace of individual mowing passes or a general field survey.
The historical aerial crop shows curved passes near the corner edge, with a
different pattern farther into the field.

The first v5 layered trial, `artifacts/fine-straw-guided-field/`, was rejected:
continuous high contrast bands resembled sand dunes. The muted v5 trial in
`artifacts/fine-straw-guided-muted/` still made the underlying field too smooth.
The final version keeps production v2 and separates broad color gain from the
fine luminance used to estimate normals and roughness. Color boundaries are not
interpreted as height changes. Two noise scales, 1.1 by 14 m and 0.4 by 6 m,
plus a smooth lateral warp interrupt the pattern. These scales are artistic.
Linear straw gains range from 0.96 to (1.25, 1.23, 1.18); coverage retention
ranges from 0.70 to one. The treatment fades across the first two metres inward,
from 24 to 40 m away from the guide, and across eight metres at each chain end.
No pavement, terrain height, contact, friction or reward data changed.

`artifacts/guided-field-v2-final/` compares the treatment off and on at station
1065 using the same v2 material. Root and independent review accepted a modest
improvement in flattened straw appearance. The far field remains somewhat
regular. Root additionally inspected the paired station 1190 view in
`artifacts/guided-field-v2-exit/`. Both studies include three positions spaced
0.30 m apart; these do not prove full lap temporal stability. The default
`straw_swath_strength` is now one, replacing the earlier disabled ellipse study.
The material preview can combine a candidate source with this broad layer and
records the source hash and treatment strength.

`artifacts/guided-field-rider-view/production.png` shows the adopted material
from the prior artistic station 950 pose (camera 1, lean minus 25 degrees,
view roll minus 14, lateral four metres, yaw eight degrees). It is not registered
to the source footage. The complete image still has substantial deficiencies
in distant scenery, motorcycle detail and photographic realism.

Human controls passed with zero failures with the adopted shader. Linux timing
was median 17.361 ms and p95 25.0 ms over 263 frames. This is a functional check,
not an isolated performance comparison or native Mac benchmark. Formatting and
whitespace checks passed. Tailscale reports the intended MacBook offline, with
last seen time 2026-09-21 08:50 UTC; native execution remains pending.
Mac export passed as `2e4aac19fd9b-8e72d567d16c`; native execution is unverified.

## Restrain the broad pavement band with current mottling

The broad band was previously judged against a simpler pavement material. With
the later binder mottling enabled, the combined result has overly strong broad
ribbons. `artifacts/asphalt-restrained-band/` compares the actual production
combination (surface variation 1.0, binder 0.85) against surface variation 0.35
with binder unchanged. Root and independent inspection accepted the reduced
band as a modest improvement at stations 400, 900 and 3000, clearest at the two
corner views. Production now uses 0.35. Source texture, aggregate relief, tone
map strength and binder mottling are unchanged. This is an appearance adjustment,
not measured asphalt reflectance or a surveyed racing line.

The preview adds `--production-baseline` for a single surface variation or binder
comparison. It preserves both current shader defaults for the baseline and
changes only the chosen value in the candidate. Current track construction sets
no overrides for these two defaults. Legacy studies retain their previous
isolated comparison behavior. The report records the mode and each effective
parameter. All twelve captures were checked: authored surface was enabled,
historical tone was zero, binder stayed 0.85, and only surface variation differed.
Missing candidate control, nonfinite strength and combined independent controls
were rejected with exit code 2.

The reference frames at 00:20 and 00:40 also show narrower irregular marks that
remain absent. Those need location and shape evidence; stronger generic bands
cannot substitute for them. Foreground motion blur limits aggregate comparison.
The new production rider capture is
`artifacts/restrained-asphalt-rider-view/production.png`, using the prior artistic
station 950 camera pose. It remains far from a registered photographic match.

Human controls passed with zero failures. Linux timing was median 17.361 ms and
p95 20.555 ms over 266 frames, a local functional check rather than a controlled
performance comparison. Formatting and whitespace checks passed.
Mac export passed as `e3e8bfdaad35-462e4875e072`; this export has not been
executed on the native Mac.

## Asphalt warmth and curb blue comparison

Reference frames 20, 40 and 50 consistently show warm charcoal pavement; the
rendered road often had a blue gray cast while its white markings remained
plausibly warm. This supports an isolated material adjustment rather than a
scene wide white balance correction. `asphalt.gdshader` now applies a linear
RGB factor derived from `(1.18, 1.04, 0.83)`, normalized by its luminance, to the
authored pavement albedo. `authored_warmth` blends from identity to that factor
and defaults to 1. This is a footage guided artistic estimate, not recovered
spectral reflectance. The normalization avoids an intentional general gain;
it does not guarantee identical rendered luminance for every colored texel.
Normals, roughness, texture scale, light direction and geometry are unchanged.

`preview_pavement.gd --asphalt-warmth=1 --frames=2` captures the zero warmth
baseline against the candidate with production binder and band settings.
`artifacts/asphalt-warm-charcoal/` contains six pairs at stations 400, 900 and
3000, with adjacent views separated by 0.5 m. Metadata matches in every field
except the image name and warmth setting. Root reviewed representative views
at all three stations; independent review inspected all six pairs and accepted
the reduced cool cast without conspicuous brown coloration. The grazing
reflection remains. Foreground grain cannot be calibrated directly from the
motion blurred footage, so grain scale was not changed in this pass.
Nonfinite and out of range warmth controls are rejected; formatting passes.

The reference blue curb at frame 50 is darker and less cyan than the current
render. A second isolated comparison changes `stripe_blue` from the shader's
sRGB values `(0.0902, 0.3765, 0.6353)` to `(0.1059, 0.2549, 0.4902)`.
White, wear, concrete texture and geometry remain unchanged. The station 2000
captures in `artifacts/curb-pigment-baseline/` and `curb-pigment-muted/` have
identical camera, lighting variants and sky metadata, and both use the accepted
asphalt warmth. Root and independent review accepted the darker blue against
frame 50. These are different track views, not a registered pixel match.
The curb still appears too flat and clean; pigment alone does not solve that.
No physics, agent actions, contact or reward data changed.
Human controls passed with zero failures. Local Linux timing was median
17.361 ms and p95 23.181 ms over 266 frames, not a controlled performance
comparison. Mac export passed as `a6a29e112206-e0ed16a32984`; this build has
not been verified running natively on the Mac.

## Preserve resolved straw in soil blends

The production grass source already contains pale straw above brown soil.
Crossfading that entire image into a second earth texture suppresses the straw
along with the source's underlying soil. An isolated shader comparison replaces
that uniform mixture with a luminance ranked material blend where detail resolves.
It does not infer actual height from the source or change terrain geometry.

`grass_height_blend` defaults to 1. The source's linear luminance is ranked with
`smoothstep(0.12, 0.55, luminance)`. Layered coverage is
`smoothstep(0, 1, 2 * coverage - 1 + rank)`, retaining exactly bare and fully
covered endpoints. The effect fades over projected footprints of 0.008 through
0.05 times the grass tile size, returning to the prior mixture for unresolved
straw. Albedo, roughness and material gradient weights share the resulting
coverage. Coverage derivatives are not introduced into relief. Palette, source
textures, texture scales, aerial gains and lighting remain unchanged.

`preview_grass_material.gd --candidate-height-blend=1` compares zero strength
against the candidate. It rejects nonfinite and out of range strengths and
other simultaneous candidate controls. Captures in
`artifacts/straw-height-blend-turn2/`, `straw-height-blend-straight/` and
`straw-height-blend-crest/` use stations 1065, 400 and 3000, each with three poses
0.30 m apart. Root inspected each location. Independent review accepted all
six pairs from Turn 2 and the straight: pale foreground strands survive more
clearly, bare shoulders remain bare, and no obvious material edge seam appears.
Nearby stills cannot establish shimmer free motion.

`artifacts/straw-layered-rider-view/production.png` is the final runtime view at
station 950 with the existing artistic reference pose. The source's tangled
straw is more apparent, and distant brown uniformity remains. This is a modest
material composition improvement, not a resolution upgrade or a photographic
match. Broad pale field patterns still need work. Physics and observation
protocols are unchanged.
Human controls passed with zero failures (Linux median 17.313 ms, p95 18.863 ms,
270 frames). Formatting and diff checks pass. Mac export passed as
`53b4b529b5bf-3c5361a25ede`; native Mac execution remains unverified for this build.

## Next field material pass

A fresh comparison of frame 40 and
`artifacts/straw-layered-rider-view/production.png` identifies the broad field
composition as the largest remaining surface mismatch in this view. Pale straw
regions and darker curved flattened strips in the footage become a mostly
uniform brown field in the render. Resolved foreground fibers do not correct
the distant material mixture. Historical aerial evidence also shows distinct
mowing regions with a change of orientation inside Turn 2.

The next candidate should be a spatially registered, nonrepeating Turn 2 straw
coverage map. Use the aerial for placement and footage for appearance. Blend
pale flattened straw against exposed earth while keeping fine texture scale
and normals independent of those broad transitions. This is a proposed authored
appearance map, not a measured contemporary coverage or friction map. Do not
repeat the rejected global contrast, enlarged tile, or generic stripe changes.

## Registered Turn 2 straw coverage

The first authored region is now implemented as an editable source pixel polygon
in `data/reference/turn2-straw-regions.json`. It follows the broad pale central
and eastern field in the historical aerial. The inspected crop was verified
pixel for pixel against source bounds `[1000, 1720, 1750, 2160]`. This outline
and its appearance weights are estimates, not a contemporary vegetation survey.

`tools/build_field_coverage.py` verifies the source hash, transforms densified
polygon edges through the pinned inverse datum operation, checks that the region
avoids road geometry, and rasterizes at local pixel centers. The 1128 by 1452
linear data texture shares the existing terrain map bounds. Its red channel
requests 0.95 grass coverage, green requests 0.8 pale straw response, blue is
unused, and alpha feathers inside the annotation over 3 metres. Outside the
annotation all channels are zero. Minimum clearance from road geometry is
approximately 7.15 metres. The importer disables alpha border modification and
premultiplication and enables mipmaps without lossy compression.

The shader blends this mapped coverage with the existing field appearance,
retaining the shoulder alpha and later access corridor handling. Pale straw
uses an artistic linear multiplier `(1.40, 1.48, 1.55)`. An earlier warmer
multiplier made the patch too golden. Fine texture scale and the independently
differentiated material relief remain unchanged; the map does not create
raised borders or affect friction, collisions, agent actions or rewards.

Matched views are in `artifacts/field-map-baseline-rider/` versus
`artifacts/field-map-muted-rider/`, and `artifacts/field-map-baseline-apex/`
versus `artifacts/field-map-muted-apex/`. Root and independent visual review
accepted the clearer pale interior against the darker roadside strip. Neither
view showed an obvious polygon corner or isolated bright island. The broad
region remains simpler than the footage and the foreground still reads as
textured earth. These stills do not establish a complete moving lap match.

The candidate can be reproduced with `preview_lighting.gd --field-map` and an
absolute map metadata path. `--field-map-strength=0` disables the adopted map
for baseline comparisons; strengths must be finite and within zero to one.
Three focused tests cover pixel center registration and axis direction,
interior feathering and outside neutrality, and invalid inputs. The production
map retains the candidate SHA256
`e08578da50238c5ee6e4ad163e405c34ad20f5c48b20db369e11b22a67a032d1`.

The final imported production texture was checked against its source MD5 and
rendered in `artifacts/field-map-production-rider/production.png`. Both scenery
bakes passed with unchanged geometry fingerprints. Human controls passed with
zero failures; local Linux frame timing was median 17.290 ms and p95 18.926 ms
over 269 frames. This is a runtime check, not a controlled performance benchmark.
Python checks, formatting and diff checks pass. The MacBook was offline during
this pass, so native Mac verification remains pending.
Mac export passed as `1ae73280c01d-1b3b7582c5d2`.

## Local paving pass joint

Reference frame 50 shows a narrow longitudinal construction boundary between
slightly darker left pavement and lighter right pavement. The game previously
contained broad mottling but no such local feature. An original appearance
estimate now adds a restrained joint from station 1265 to 1510, with 16 metre
fades at both ends. These station limits, the 18 mm line width, and its position
at 0.46 of pavement UV width are artistic choices, not surveyed repave seams.
The actual pavement mesh was checked: UV zero is the right edge and full width
is the left edge. The first candidate had the orientation reversed; the adopted
candidate corrects both the line position and the side tone response.

`paving_joint_strength` defaults to 1. The right pass multiplier is 1.04 and the
left is 0.92, with at most 30 percent local gain for the narrow line. The line's
coverage is its interval overlap with the projected pixel footprint, retaining
subpixel energy without a hard threshold. Roughness, texture scale, normals,
track geometry and all contact or reward data remain unchanged. This feature
must not be interpreted as an optimal racing line or friction boundary.

`preview_lighting.gd --paving-joint-strength=0` produces the baseline;
strength 1 produces the candidate. The command validates a finite unit range
and records the selected strength and asphalt shader hash. Root and independent
review accepted `artifacts/paving-joint-oriented/production.png` against
`artifacts/paving-joint-baseline/production.png` as a modest improvement. It
reads as a subtle construction seam, not a painted stripe. Root also inspected
stations 1270, 1400.5 and 1498 for endpoint transitions and adjacent position
continuity. These sampled stills do not establish stability over a moving lap.
The reference's irregular crossing marks and more varied surface wear remain
absent, and no exact video camera registration is claimed.

Human controls passed with zero failures. Local Linux timing was median
17.361 ms and p95 19.463 ms over 266 frames. Formatting and diff checks pass.
Nonfinite and out of range preview strengths were rejected. The final production
capture at station 1400 matches the reviewed candidate PNG exactly, and its
metadata records the adopted strength of 1.
Mac export passed as `ec35bbe82000-2f4c203e8c2f`; this build has not been verified
running natively on the Mac.

## Grass color mipmap correction

The imported grass texture mip chain exactly matched Godot averaging encoded
sRGB bytes. Its base pixels matched the source PNG exactly. At mip levels 4
and 8, rebuilding in linear light increased mean decoded texture luminance
by 8.40 and 9.01 percent respectively. These are texture measurements, not
claims about final rendered brightness. `test_color_mipmaps.gd` now supports
`--audit-texture` to reproduce the source, import and rebuilt chain comparison.

The original v2 base pixels remain unchanged. Terrain and baked stubble now
load `dry_cut_grass_v2.res`, built with the existing color mipmap builder.
Its sibling JSON records source, builder and output hashes. Scenery source
validation includes both new files. Both scenery bakes were regenerated.

The filtering study is in `artifacts/grass-linear-filter-turn2`. The v5
texture was also reconsidered with the new broad field coverage map in
`artifacts/fine-straw-mapped-turn2`; it still produces overly uniform brown
bands. The new generated v6 study was compared against corrected production
v2 in `artifacts/short-straw-v6-linear-turn2`, using two adjacent camera
positions and linear light mipmaps for the candidate. Root inspected the
first pair. Its foreground pattern differs slightly but does not establish
a convincing realism improvement, so v6 remains outside runtime assets.
`preview_grass_material.gd --linear-mips` enables this comparison.

The correction is modest. Field coverage, repeated straw patterns, lighting
and the overall scene still differ visibly from the onboard reference.
No exact video registration or photorealistic match is claimed.

Color mipmap tests and human controls passed with zero failures. Local Linux
frame timing was median 17.251 ms and p95 18.097 ms over 271 frames, a runtime
check rather than a controlled benchmark. Formatting and diff checks passed.
Mac export passed as `63331edbaf79-74ab066d9e53`. This export has not been
verified running natively on the Mac.

## Cockpit surface orientation and shadow diagnostic

The red tank material has no bitmap texture. Matched captures in
`artifacts/cockpit-shadow-baseline` and `artifacts/cockpit-shadow-disabled`
establish that its regular dark pattern comes from shadow rendering.
Increasing constant bias from 0.1 to 0.2 leaves stipple; 0.4 removes the
pattern but also loses close shadows in the controlled light comparison.
Neither change was adopted. Doubling normal bias, changing the near split,
disabling blur, and using 32 bit shadow depth did not produce an acceptable
replacement. Blur zero exposed triangular shadow boundaries, initially
misinterpreted as self shadowing. The caster isolation and filter diagnosis
below supersede that interpretation.

`preview_cockpit.gd` now records the light direction and shadow settings and
accepts bounded shadow diagnostic overrides. `--diagnostic-sun` uses an
explicit bike relative light direction to compare shadow retention. This
light is a diagnostic, not reconstructed reference illumination.

An independent mesh audit identified a separate defect in `_lathe()`:
its triangle order generated inward facing normals on tires, rings and
grip details. The tank and ignition primitives do not have this defect.
The lathe triangle order is corrected. A closed analytic annulus regression
checks the direction of inner, outer and end surfaces and its signed volume.
The old implementation fails; the corrected implementation passes. Smoothed
corner normals are checked for outward direction, not exact face alignment.

`artifacts/cockpit-lathe-corrected` records the four matched camera views.
The visible difference is small and the tank shadow artifact remains.
Human controls pass with zero failures. Local Linux timing was median
17.361 ms and p95 25.0 ms over 262 frames, not a controlled performance
comparison. Geometry and dashboard validation pass. This change does not
establish a photographic match to the Ducati footage.
Mac export passed as `10783bb35ddc-d9dc16c3ab73`. It has not been verified
running natively on the Mac.

## Cockpit shadow filter diagnosis

Further caster isolation supersedes the initial tentative self shadow diagnosis.
Disabling only fuel tank casting, changing its material to back face culling,
and reversing sunlight shadow culling with that tank override produced no
meaningful tank pixel change. Disabling the front assembly's shadow casting
removed the pattern. Removing rider casting did not. Both standard and custom
materials in the front assembly contribute cast shadows on the tank.
These are diagnostic overrides only. No caster removal was adopted.

The dominant regular stipple comes from sampling these received shadows with
Godot's default Soft Low directional filter. The
[Godot 4.7.2 PCF implementation](https://github.com/godotengine/godot/blob/4.7.2-stable/servers/rendering/renderer_rd/shaders/scene_forward_lights_inc.glsl#L305-L336)
rotates its sample disk at every screen pixel using interleaved gradient noise.
The [quality settings implementation](https://github.com/godotengine/godot/blob/4.7.2-stable/servers/rendering/renderer_rd/renderer_scene_render_rd.cpp#L1220-L1260)
uses four samples for Soft Low, eight for Medium and sixteen for High. High
also increases the kernel radius from two to three. Therefore the candidate
uses High and `shadow_blur = 2.0 / 3.0` to preserve the previous effective
filter radius. Bias, normal bias, cascade distances and shadow depth remain
unchanged. This improves sampling rather than removing nearby cast shadows.
Godot's [shadow filter guidance](https://docs.godotengine.org/en/stable/tutorials/3d/lights_and_shadows.html#shadow-filter-mode)
also describes this dithering on smooth surfaces.

The matched candidate is `artifacts/cockpit-shadow-high-matched-blur`, with
`artifacts/cockpit-shadow-medium` as the intermediate comparison. Root
independently reviewed the High wider view and confirmed a clear reduction
in the pattern with the shadow retained. The controlled light study at
`artifacts/cockpit-shadow-high-diagnostic` also retains hardware and tank
shadows. The earlier bias 0.4 study had lost nearby shadows.

A local noise proxy uses the smooth tank rectangle `(590, 610, 750, 640)` in
the 1280 by 720 wider view. Encoded luminance residual RMS after a one pixel
Gaussian blur is 2.945 for Low, 1.282 for Medium and 0.663 for High with matched
radius. The calculation, image hashes and limits are recorded in
`artifacts/cockpit-shadow-filter-metrics.json`. This is not a physical
radiometric error or a realism score. Glove materials changed independently
during the study; the measured rectangle contains only the tank.

`preview_cockpit.gd` records the effective directional filter quality from
project defaults or the explicit study override, separately from override
booleans. It rejects simultaneous Medium and High overrides. Caster group,
tank culling and tank casting controls preserve the isolated diagnostic.
This study does not establish an exact match to the reference illumination,
prove that every shadow artifact is gone, or measure native Mac performance.


## Custom glove finish and accepted shadow filtering

The glove now uses an original object space shader, `glove_finish.gdshader`.
Explicit UV2 part tags separate the smooth molded protector from leather.
The shader supplies restrained leather grain, a tailored dorsal panel,
filtered seam relief and a limited warm cuff accent. The 0.8 mm grain pitch,
12 micrometre grain relief, 30 micrometre seam relief and roughness values
are artistic estimates, not measurements recovered from compressed footage.
Vertex positions and articulation remain unchanged. The existing standalone
glove preview now uses the production material.

The first red hand shell candidate was rejected. A close crop identified the
red and black object beside the right grip in frame 10 as a phone mounting
bracket, not a glove. Actual gloves in frames 20 and 30 are predominantly
black with limited warm orange red and white cuff accents. Those frames guide
the adopted palette. They do not resolve microscopic grain or exact seams.
The image comparison in `artifacts/glove-material-final-comparison` preserves
original pixels without enlarging the small reference crop. Differing hand
pose and framing are explicitly labeled. The modeled glove remains too
smooth and simple to reproduce the reference folds and articulated padding.

Production now uses directional shadow filter High with blur 2/3. The final
four cockpit views are in `artifacts/cockpit-black-glove-production`; metadata
records quality enum 4 and unchanged biases. `artifacts/rider-shadow-high-filter`
confirms that rider ground shadows remain visible. Root visually inspected
the wider cockpit view and rider shadow capture. Human controls pass with
zero failures after the filter change; local Linux median was 17.361 ms and
p95 24.125 ms over 261 frames. This is a runtime check, not an isolated GPU
benchmark or proof of Mac performance. Dashboard and glove geometry checks
also pass. The Mac peer reported online, but SSH timed out during this pass,
so native Mac verification remains outstanding.
Mac export passed as `b95fc47a2bbe-f2b3e8086617`; native execution is unverified.

## Terrain structure and source sampling studies

Two matched appearance trials were rejected for production. Lowering asphalt
roughness by 0.12 at station 700 changed the highlight but did not establish
closer footage similarity. `roughness_offset` remains zero. The bounded preview
argument records the selected value independently from contact and friction.

The authored pale field region overwrote earlier swath color variation. A
centered retained swath trial first affected only a small distant patch because
its coordinates were gated by the narrow Turn 2 corridor. Expanding it with
the existing global elongated field pattern increased its extent, but still
produced generic mottling rather than the reference's coherent mowing marks.
Root and independent visual review rejected both variants. The new
`retained_swath_strength` remains zero. Captures are in `artifacts/field-swath-control`,
`artifacts/field-swath-candidate`, `artifacts/field-swath-expanded` and
`artifacts/field-swath-expanded-apex`. Coverage, corridors and contact data
were not changed by these shader trials.

`build_terrain_color.py` now supports bounded detail smoothing and output
spacing studies while preserving existing defaults. The finer candidate uses
0.2 metre Gaussian sigma and 0.6 metre output pixels, compared with the original
0.6 metre sigma and 1 metre pixels. These are processing parameters, not newly
measured vegetation dimensions. The input remains the pinned historical
public domain NAIP image. The broad macro image is independently unchanged.
The preview tool can load a candidate detail map after verifying its image
hash, dimensions, local mapping and geometry source hashes. A synthetic narrow
stripe test verifies contrast retention through output area integration, while
rejected source colors must remain neutral under both filtering settings.

The source sampling candidate was rendered at the same Turn 2 entry and apex
poses in `artifacts/terrain-detail-fine-turn2` and `artifacts/terrain-detail-fine-apex`.
Root found no clear improvement in the recognizable field structures against
the footage, so its larger detail map was not adopted. The runtime map and
builder defaults remain unchanged. Finer sampling preserves more historical
source signal, but that alone does not establish better rendered fidelity.
Eleven terrain color tests, Ruff, Godot preview compilation and formatting pass.
Nonfinite asphalt roughness input is rejected before rendering.
Human controls pass with zero failures (Linux median 17.217 ms, p95 24.864 ms,
265 frames). Mac export passes as `7880a095bd5b-246d85cdc600`, without native
Mac verification. These checks validate the tooling integration and unchanged
production material defaults, not a new visual fidelity claim.

### Aggregate scale correction (2026-09-21)

The controlled Turn 2 study isolates a coarse foreground aggregate pattern in
`artifacts/pavement-aggregate-control/production.png`. Removing inferred normal
relief alone (`pavement-aggregate-flat`) leaves the visible color speckle. Reducing
the authored texture repeat from 0.8 m to 0.25 m produces smaller, less prominent
grain (`pavement-aggregate-fine`). Relief changes proportionally from 0.00035 m to
0.000109375 m, preserving the relief to tile ratio. These defaults are adopted.
The broad binder variation, roughness, collision mesh and friction are unchanged.

This is an appearance estimate, not measured aggregate size. Frame 40 has motion
blur and different framing, so smoothness in that frame alone cannot establish
physical stone dimensions. The result improves the static foreground appearance;
it does not establish a calibrated match or solve the field and cockpit gaps.

Reproduce the controlled pose with `preview_lighting.gd`, using
`--production-only --camera=1 --station-m=950 --lateral-m=4 --lean-deg=-25
--view-roll-deg=-14 --view-yaw-deg=8`. The old material can be captured with
`--asphalt-tile-m=0.8 --asphalt-relief-m=0.00035`. The diagnostic tool validates
finite tile values in [0.05, 2] metres and relief in [0, 0.002] metres and records
explicit overrides. Shader hashes record the default values when not overridden.

The comparison specification is `artifacts/pavement-aggregate-comparison.json`;
`tools/compare_material_views.py` generated the reference and controlled pairs in
`artifacts/pavement-aggregate-comparison/`, without color grading or upsampling.
The Linux human control check passed with zero failures. Its short render sample
is not a Mac performance benchmark.

### Continuous Turn 2 mowing shader (2026-09-21)

The short noise patches guided by flattened straw corridors did not reproduce
long curved mowing passes. A new `mowing_gain` uses distance to a westward ray:
straight passes join curved passes continuously around the east bend. The
center `(306.70854, 686.54039)` comes from a least squares circle fit to current
track centerline samples with `980 < s < 1280`. Solve
`[2*x, 2*z, 1] * [cx, cz, c] = x*x + z*z`; the fitted radius is 96.4531 m and
maximum radial residual is 0.8506 m. This is a placement guide, not mowing survey
data. Source track SHA256 is
`a9ad305331e90dc90b7aed703b3ff7f4d850dabd580ed6a84d826a8ce4a93c04`.

Nominal 2.6 m spacing, phase warping, continuity, contrast and fade boundaries are
artistic estimates. The effect fades westward between x=140 and 175 m, inward
between radial distance 14 and 30 m, and outward between 83 and 94 m. It changes
only straw color before grass/soil blending. It does not restore grass over the
bare shoulder or change friction, collision, terrain elevation, or stubble.
Screen derivatives fade unresolved bands. A zero strength skips the calculation.

The first trial (`mowing-bend-candidate`) was too regular and was largely hidden
by the field map footprint. The expanded, warped trial (`mowing-bend-varied`)
removes that dependency and varies pass spacing and continuity. It gives a modest
improvement in coherent field structure; default strength one is adopted. It
still falls short of frame 40, especially in foreground detail and lighting.
The existing 0.5 to 3.5 m soil/grass shoulder blend suppresses the straw effect
near pavement; that is not evidence that every unchanged foreground pixel is
inside the shoulder.

Comparison sheets and image hashes are in `artifacts/mowing-bend-comparison/`.
The apex pair uses identical upright camera 1 at station 1135, strength zero
versus one. The entry pair uses the documented leaned station 950 pose. The
reference pair is explicitly not calibrated for camera or motion blur.
`preview_lighting.gd --mowing-band-strength=0|1` reproduces either state and
records the effective parameter. Linux human controls passed with zero failures;
short frame timing samples are not a sustained native performance test.

### Steering damper silver finish (2026-09-21)

The close cockpit frame at 00:10 shows a bright silver transverse housing. The
procedural part used a grey StandardMaterial and an unshaped cylinder. The new
material reuses the original `cockpit_finish` shader with a brighter silver color,
metalness one, roughness 0.22, and shallow directional grain. `grain_aspect`
defaults to one, preserving all other existing cockpit finishes. For this part,
local x is the lathe axis: aspect (1, 0.04, 0.04) stretches grain around the tube.
Pitch 0.00015 m and relief 0.000001 m are artistic estimates, not measured machining
specifications. The shader filters unresolved grain rather than showing it as
coarse scratches.

A closed revolved housing replaces the body cylinder. Its outer shoulder tapers
from 9 mm radius toward 7 mm at each end, preserving the 160 mm length and the
existing mounting position. A 4.6 mm bore surrounds the existing shaft. The
existing shaft finish is also brighter. These dimensions remain original visual
estimates and do not alter simulation geometry or motorcycle dynamics.

Controlled captures are in `artifacts/damper-finish-control/` and
`artifacts/damper-finish-candidate/`; comparison sheets and source hashes are in
`artifacts/damper-finish-comparison/`. The same production camera and lighting show
a modest improvement in the silver surface, not a solution to overall bike
fidelity. The simplified geometry and missing component detail remain visible.
`preview_cockpit.gd` now includes the finish shader hash in new study metadata.
Headless motorcycle geometry/dashboard checks and rendered human control checks
passed. Native Mac execution and sustained performance of this revision remain
unverified.


## Layered field placement diagnostic

The v7 material remains production. New candidates are retained as studies, not
adopted solely because their pixels differ. `mowing_detail_strength` defaults to
zero. Its optional continuous arc length streaks use the existing Turn 2 mowing
coordinates and filter unresolved noise. The first render at
`artifacts/mowing-streaks-v1` adds fine directional variation but does not resolve
the field's broad material organization.

`build_field_coverage.py` now supports explicit `overlap_policy: source_over`.
The default still rejects overlapping regions. Ordered source over blending
preserves the underlying material through an overlaid polygon's feather, rather
than replacing its alpha and accidentally revealing unannotated ground. Four
focused tests pass, including overlap feather endpoints. Existing disjoint
regions retain the same rasterization behavior.

`data/reference/turn2-straw-regions-layered.json` is a candidate with an expanded
pale eastern field and a broad western darker region traced against the existing
historical aerial crop. The map in `artifacts/turn2-straw-expanded-map` has SHA256
`2b3f7a20f30ba39f71bb2c21ff1be875a3cba754f7d689955980f5db6572afe6`.
Its minimum road clearance is 2.579 metres. That minimum does not describe the
whole boundary: the remaining pale field setback is materially wider along the
eastern curve. Independent visual inspection supports the polygon's historical
placement but cannot establish the current straw boundary or feather width.

The western dark overlay alone changes no channel by more than two levels in
the station 950 capture, so it does not address entry foreground uniformity.
Expanded outline entry and apex comparisons are saved in
`artifacts/turn2-straw-expanded-comparison`. Their effect is modest. The additional
1.5 pale gain trial in `artifacts/turn2-straw-expanded-pale-verified` makes the
mapped island too conspicuous without fixing the brown foreground. It is not
adopted. The preview CLI records this isolated gain control; shader defaults must
be read through RenderingServer when no material override exists.

For locating future edits, stations 950, 1065 and 1150 project to approximately
(421.9,349.4), (600.9,315.4) and (648.6,186.4) within the reviewed 750 by 440
crop. These positions are computed from current track samples and the pinned
datum transform, not inferred from a screenshot. Correct the field transition
along that actual road boundary before increasing the whole pale region's gain.
The scene is still visibly synthetic. Native Mac testing was unavailable because
the intended Tailscale peer reported offline during this pass.


## Road constrained field transition

The production map now uses `data/reference/turn2-straw-regions-roadside.json`.
The historical pale outline expands by an artistic 12 metres, clipped at one
metre from current road geometry. Disconnected pieces beyond the road are
removed. Its interior feather remains 1.5 metres. This is a visual placement
rule, not a measured vegetation edge or an instruction to change physics.
The output SHA256 is
`9c5f706f71974a2691a0ba291d723ce0cf8b8d85dae81ec6a371108b60fe96dc`.
The builder defaults to the adopted annotation and records its own source hash.

This correction narrows the broad brown apron visible through Turn 2. Entry and
apex comparisons use identical camera and lighting with unchanged material gains:
`artifacts/roadside-field-comparison/00.png` and `01.png`. Both root and independent
review found a modest improvement without a new visible hard polygon edge.
The imported production scene was also inspected at station 1280 in
`artifacts/turn2-roadside-production-exit`. The near shoulder still appears too
uniform, and scattered stubble does not yet convey the reference's flattened
field. These stills do not establish temporal stability or photographic fidelity.
Five field annotation tests cover datum mapping, blending, road clearance and
exclusion of disconnected regions. No track geometry, friction or action schema
changes are involved.

The framing audit also found that recent comparisons used the 1.51 metre rider
anchor. The existing 1.14 metre onboard anchor reveals too much cockpit for frame
40. Preview overrides now independently control camera height (0.9 to 1.8 m),
downward pitch (0 to 40 degrees), and FOV (50 to 110 degrees), reject nonfinite
inputs and chase camera use, and save their values in capture metadata.
The 1.37 metre study makes more cockpit visible but is not a recovered camera
calibration. Human camera defaults and the policy camera remain unchanged.


## Reservoir transmission softness

The current vessel multiplied a sharp screen copy by absorption. White road
stripes and support stalks remained distinctly readable through the molded
plastic. Comparing frame 10 with `artifacts/reservoir-haze-control` isolated this
as a material problem rather than missing mesh segments or flipped normals.
The new shader samples mipmapped screen color and adds a restrained light
responsive wall contribution while retaining existing amber absorption.

The initial fixed LOD 2 study softened the line, but haze 0.15 made the forward
pose too gray. The adopted haze is 0.07. Production instead projects an artistic
5 mm scattering footprint into screen pixels using view depth, projection scale
and viewport height, then chooses the matching mip level. This avoids applying
the same pixel blur at human and policy camera resolutions. The footprint and
haze are appearance estimates, not measured plastic scattering properties.
The transmitted term is reduced by the haze fraction, while the wall's diffuse
response is lit by the scene. This is not emission painted onto the wall.

[Godot's screen reading documentation](https://docs.godotengine.org/en/stable/tutorials/shaders/screen-reading_shaders.html)
describes the mipmap requirement and the opaque scene copy used here. The
material still lacks refraction and multiple internal reflections. Transparent
objects do not appear in the copied screen. No fluid dynamics are simulated.

Final fixed poses are in `artifacts/reservoir-scaled-blur-production`. Native
pixel crops in `artifacts/reservoir-transmission-comparison` show control versus
final production and forward poses. Turn 2 was separately inspected in
`artifacts/reservoir-scaled-blur-turn2`. The result softens the transmitted stripe
and retains the amber character, but is not a matched plastic scan or a complete
solution to the cockpit's approximate geometry.

`preview_cockpit.gd` accepts independent `--reservoir-haze` (0 to 1) and
`--reservoir-blur-m` (0 to 0.01) overrides, rejecting nonfinite values. Captures
record both overrides and effective shader defaults. The early checkpoint's
fixed LOD flag was replaced by the distance and resolution aware control.
Rendered human controls passed. The real 640 by 360 agent camera check passed
five recorded observations, immutable image hashes, frozen ticks, queued reset
and advance serialization, and rejection without a real renderer. Native Mac
performance remains to be verified separately.

## Angular headlamp reconstruction

The prominent oval below the instrument was the rear cap of the old headlamp
body. Hiding that mesh in an isolated capture confirmed its identity. Applying
more detailed polymer to this shape could not reproduce the reference outline.
The original replacement in `headlight_visual.gd` uses a closed angular enclosure,
a red surround, paired lens panels, projectors and running lights. It uses the
existing custom molded polymer shader. Panel depths overlap their supporting
surfaces to avoid detached layers. The assembly is lowered by an estimated
55 mm relative to its initial placement to better follow the lamp upper edge
relative to the fork bridge in manufacturer photographs.

All dimensions and hidden rear details are artistic estimates. Manufacturer
MY25 photographs inform this silhouette; this does not resolve the model year
and instrument differences in the supplied onboard video. No commercial game
geometry or manufacturer texture was imported.

Six rendered views are saved in `artifacts/headlight-angular-lowered` with
builder hashes. `artifacts/headlight-final-comparison` includes the preceding
cockpit render and a separately framed manufacturer comparison. Root and an
independent reviewer preferred the lower placement. The casing remains thick
and faceted, the display support is approximate, and the overall motorcycle
still looks synthetic. Static views do not prove clearance across suspension
travel. This is an intermediate geometry improvement, not visual acceptance.

Mesh tests passed finite normals, winding, positive volume and closed topology
for each custom lamp panel. Rendered human controls passed with zero failures.
Linux Mobile rendering measured 17.30 ms median and 24.67 ms p95 across 265
samples; these are not MacBook performance results. Physical dynamics and
agent action or observation schemas are unchanged.

## Field structure through the sparse shoulder

The shader multiplied mowing detail into the grass color before blending with
soil. Consequently the pattern vanished precisely where cut vegetation became
sparse near the road. The radial region also faded between 83 and 94 metres
from the Turn 2 mowing center, attenuating the inner roadside appearance.
The existing grass coverage generator fades from soil to grass between 0.5 and
3.5 metres from the road boundary; its role remains unchanged.

Mowing now modulates the composed field surface. Explicit bare access corridors
are excluded using the existing clearance weight and grass retention. The
outer regional fade moves to 94 through 104 metres, and the previously optional
fine directional structure is enabled. These are bounded artistic placement
choices, not surveyed mowing dimensions. Texture resolution, physics geometry,
friction and agent observations are unchanged.

The matched apex comparison is `artifacts/mowing-shoulder-comparison/00.png`;
`01.png` compares footage with the new render without color grading. Entry and
midturn captures are in `artifacts/mowing-shoulder-entry` and
`artifacts/mowing-shoulder-midturn`. The roadside field has modestly more
continuous variation. Doubling fine pattern amplitude was separately rejected:
its soft bands looked like furrows rather than irregular flattened straw.
Both root and independent review preferred the original amplitude. The field
still looks too smooth and uniform overall. These fixed views do not establish
motion stability or a photographic match.

## Playable onboard framing

Both riding views previously retained only 22 percent of motorcycle roll in
camera orientation. The onboard anchor was also low enough that its display,
handlebar and damper obscured much of the road in a leaned pose. Frame 40 shows
substantial landscape banking with a less rotated dashboard. An independent
manual image review estimated a terrain boundary angle around 16 to 17 degrees
and dashboard edge around minus 7 degrees. Uneven terrain, perspective and
unknown stabilization prevent treating this as a measured camera calibration.

The onboard preset now uses an estimated 1.37 metre anchor height, 0.30 radian
downward gaze, 90 degree FOV and 0.65 roll coupling. The helmet eye retains its
existing 0.22 coupling and framing. The agent camera uses its own configuration.
Named constants replace duplicated roll literals, including in the pitch
override diagnostic. These are appearance estimates; bike dynamics are unchanged.

`artifacts/onboard-framing-comparison/00.png` compares the old and new actual
runtime camera at the same station, lean and lateral position. Both root and
independent review favour the increased road visibility and banking. The
reference comparison in `01.png` remains visibly different: cockpit placement,
reservoir proportions, shadow and surface appearance need work. This is an
improved playable framing preset, not a recovered camera mount.

The owner subsequently chose current manufacturer cockpit specifications.
The display now follows Ducati's nominal 6.9 inch, 8:3 active area and 1280 by
480 raster, with a newly arranged original functional layout. The molded bezel
and mount remain estimates. `artifacts/current-ducati-display.png` verifies the
rendered hardware and live speed, gear, RPM and lap readings. The earlier
`artifacts/turn2-onboard.mp4` verifies the new camera framing but predates this
screen shape correction. It must not be presented as the final cockpit render.
Human controls and the five real agent camera observation checks passed before
the display adjustment; the functional motorcycle display test passed afterward.

## Road marking aggregate response

Road edge paint previously used a flat normal and revealed concrete color in
worn spots even though its substrate is asphalt. The edge paint branch now adds
original world coordinate aggregate noise with an estimated 8 mm pitch and
0.25 mm relief amplitude. Visibility fades with the projected footprint. Height
is differentiated before applying that visibility, so the fade itself does not
create a surface ridge. Subtle color and roughness variation share the noise.
Wear quantity and marking width remain unchanged; exposed road substrate is
darker, while curb substrate retains its concrete color.

The shared shader already outputs tangent normal maps for concrete. The new
surface gradient is converted into that same convention, using mesh tangents
that the track builder generates. Godot's [spatial shader reference](https://docs.godotengine.org/en/4.7/tutorials/shaders/shader_reference/spatial_shader.html)
documents the view space normal and tangent space normal map outputs. No normal
texture, geometry, collision or friction changes are involved.

`artifacts/paint-aggregate-comparison/00.png` shows native pixel crops of the
same onboard pose. Root and independent review found subtle granular variation
without obvious heavy chips or seams. This is a small material improvement,
not a recovered paint scan. The still does not establish temporal stability.
Rendered controls passed with zero failures. Linux frame intervals were
17.36 ms median and 25.00 ms p95 across 262 samples; these are not Mac timings.
The Mac package exported successfully and remains separately unverified.

## Sky coverage around the apex

The first detailed sky region points toward azimuth 82.5 degrees. The current
apex camera points roughly toward azimuth 194 degrees, where most of the visible
sky still came from the lower resolution 1774 by 887 panorama. Sharpening the
first patch would not fix this coverage gap.

A second original 1536 by 1024 rectilinear source, documented in
`assets/source/sky-studies/README.md`, covers the apex direction. The shader now
accepts two independent fixed world projections, each with its own texture and
edge feather. The second composites after the first before shared sky radiance
and background processing. Direct sunlight and its direction are unchanged.
The setup helper preserves the original default slot and rejects invalid slots.
Projection tests cover independent parameters and cardinal and polar bases.

`artifacts/sky-secondary-comparison/00.png` uses the new secondary off diagnostic
to isolate the additional source. Entry, middle and apex production captures
were inspected. Root and independent review found more natural separated cloud
structure than the soft panorama, without an obvious rectangular boundary in
these stills. The bright cloud arrangement and morphology still differ from
frame 40; the generated source is not a reconstruction of measured weather.
The new `--sky-secondary-off` flag disables only the added region; existing
`--sky-patch-off` disables both. Capture metadata records both enabled states.

The rendered human controls check passed with zero failures. Linux frame
intervals were 17.36 ms median and 20.47 ms p95 across 264 samples.
`artifacts/turn2-detailed-sky.mp4` records the compatible Turn 2 replay with
the current display, camera, paint and sky at 1280 by 800, 30 FPS. Its
23.33 seconds include a short stationary tail. Contact sheet inspection at
two second intervals found no obvious hard patch boundary during the turn;
this sampling does not establish fine temporal stability. Fixed frame rate
movie capture is not a live performance measurement.

Mac export `8068a2089019-3266bd5ec984` completed successfully. Its manifest
records the exact source content and artifact digest. This package has not
been tested on the Mac, which was offline during this pass.

## Restrained broad pavement tone

The authored asphalt applied the same broad noise twice, at 2.7 metre cross
track and 28 metre longitudinal scale. Their combined gain ranged from
0.7728 to 1.1648 independently of the existing surface variation control.
This reinforced soft longitudinal ribbons across the road. A separate
`broad_tone_strength` now attenuates that combined product around its 0.9604
midpoint response. The midpoint is an artistic reference, not a measured mean.
Production uses 0.25 for this strength and for binder mottling, previously 0.85.
Aggregate scale, normal relief, edge treatment, lighting and grip are unchanged.

`artifacts/pavement-tone-comparison/00.png` compares the same apex pose before
and after. The second sheet compares the candidate with frame 40. Root and
independent review preferred the combined reduction over reducing broad tone
alone: soft ribbons are less conspicuous, while foreground aggregate remains
visible. Entry and middle bend captures also retain surface variation. Camera,
exposure and motion blur differ from the footage, so these comparisons do not
establish calibrated material reflectance or an exact visual match.

The existing lighting preview accepts independent bounded broad tone, binder
mottling and surface variation overrides and records them in capture metadata.
The rendered human controls check passed with zero failures. Linux frame
intervals were 17.36 ms median and 25.00 ms p95 across 262 samples, not Mac timings.
Mac package `8806d75b05dc-9d6ee01cf6e2` exported successfully; native Mac
verification remains outstanding.

## Interrupted cut field appearance

Broad mowing strength 0.25 alone barely changed the field mismatch. Disabling
both mowing components isolated additional broad bands from historical color,
and disabling mapped aerial contrast still left the pale field to brown
shoulder transition. That remaining transition belongs to the material mixture
and corridor treatment, so removing historical aerial detail globally would
not address its cause. Production retains the aerial data at full strength.

The mowing shader now uses shorter irregular multiscale streaks instead of
continuous radial sinusoids. Coordinates still join the straight and curved
regions continuously, with footprint filtering for unresolved detail. Pattern
dimensions remain artistic estimates. This changes color only, not vegetation
coverage, contact geometry or grip. The existing lighting preview now supports
a bounded mapped aerial contrast diagnostic and records its effective value.

`artifacts/field-streak-comparison/00.png` compares the identical apex view.
Root and independent review found a small directional improvement toward
broken straw structure, with less contour appearance. The entry view also
retains interrupted structure. Broad smooth material transitions remain and
the scene is not yet a close match to the footage. The reference's camera,
exposure and motion blur differ from these static captures.

Rendered human controls passed with zero failures. Linux frame intervals were
17.32 ms median and 26.83 ms p95 across 266 samples. A compatible replay
completed all 2699 transitions in `artifacts/turn2-field-streaks.mp4`, rendered
at 1280 by 800 and 30 FPS for 700 frames including a short stationary tail.
Fixed frame rate capture is not a live performance measurement. Mac package
`89eb9111f41b-112e8379c9a4` exported successfully. The Mac remained offline
when checked, so this package has not been tested there.
The replay contact sheet was inspected at two second intervals without an
obvious new field seam; this does not establish fine temporal stability.

## Mapped shoulder compositing

The regional straw override replaced the previously perturbed shoulder coverage
with the smooth distance alpha, losing transition irregularity. It now retains
that variation and biases flattened straw coverage through the existing
transition with `1 - (1 - alpha)^2`. Zero and full coverage endpoints remain
unchanged, and explicit bare corridors still apply afterward. This is an
appearance estimate, not a measured shoulder width or vegetation survey.

`artifacts/shoulder-blend-comparison` isolates noise preservation from the
coverage bias. Both changes are small at scene scale; independent review did
not find a convincing removal of the broad brown apron. The entry capture
also remains visually similar. This fixes compositing consistency but does
not establish a substantial realism improvement.

A proposed mask overlap explanation was checked against current runtime data
and rejected: the map metadata specifies 1 metre minimum road clearance and
1.5 metre feathering. At the nearest apex sample, station 1149.572, source
pixels sampled 1.5 and 2 metres beyond the left boundary have detail alpha
127 and field alpha 188 out of 255. At 2.5 metres those values are 236 and
255. These are nearest source pixel samples, not filtered rendered values,
but they demonstrate that the two transitions overlap. Earlier candidate
map measurements must not be substituted for the current source.

The broad field material composition remains unresolved; further small
coverage remaps are not supported as the primary route to the fidelity goal.
Rendered human controls passed with zero failures. Linux frame intervals
were 17.36 ms median and 20.88 ms p95 across 265 samples. Mac package
`e5bee3ee1350-aff68223c957` exported successfully but is not native verified.

## Preserve unresolved paint wear

The road marking shader multiplied exposed substrate by its detail visibility,
making distant paint completely intact. It now blends toward approximate mean
worn coverage when individual chips become unresolved, rather than toward zero.
The correction applies only to edge paint. Curb materials, marking dimensions,
contact geometry and wear strength remain unchanged.

The mean estimate is 0.168 before multiplying by wear amount. Numerical
integration of the thresholded cubic value noise with independent uniform corner
values, uniform cell positions, Python random seed 7391 and 400000 samples gave
0.168025 with estimated standard error 0.000517. This is a filtering model,
not measured Thunderhill paint wear or an exact average of the shader hash.
Resolved detail retains the original expression. Both developer preview tools
support `--paint-pristine-distance` for the preceding behavior and
`--paint-mean-wear` for the corrected default.

Matched apex captures differ by at most two channel levels. Explicit sky,
field, road interior and cockpit crops are identical; the road crop containing
the far painted edge changes. `artifacts/paint-mean-isolation.json` records
the bounds. The native pixel crop comparison is `paint-mean-comparison/00.png`.
The difference is subtle and does not solve the broader material mismatch.

Actual Vulkan compilation and rendering passed, as did rendered human controls,
GDScript formatting and diff checks. The compatible replay completed 600
transitions and 180 movie frames. Nine sampled frames showed no obvious broad
transition; this is not a full aliasing or Mac performance test. The known
ObjectDB exit warning remains. Evidence is `artifacts/paint-mean-motion*`.
Clean Mac export `4396ea10c39c-c78958980dc3` completed successfully. This small
filtering correction has not yet been installed or verified on the physical Mac;
the previously verified broader pavement build remains the native review version.
