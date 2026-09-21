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
