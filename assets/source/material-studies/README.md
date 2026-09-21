# Dry cut grass study

## Original racing asphalt

`racing-asphalt-v1.png` is original generated albedo, not extracted video pixels
or a measured scan. It is copied unchanged to
`godot/assets/materials/racing_asphalt_v1.png`. Actual dimensions are 1254 square,
despite requesting 2048. Both copies have SHA256
`7d412b9ad0c6900e6bf6b184fa2f30a084db3a4534010053027c5a8933ec7f2d`.
Video frames 00:20 and 00:40 guided human inspection; the generator received
the text prompt below without an image reference. Dimensions, relief and
reflectance are artistic estimates. The runtime shader uses a 0.8 metre tile,
linear gain 0.30, contrast 0.70 around linear gray 0.10 and a 0.35 mm
luminance based relief coefficient. This is not a calibrated height map.
`godot/tools/bake_asphalt_material.gd` builds `racing_asphalt_v1.res` with
linear light mip averaging, encoded back to sRGB. The runtime loads that baked
texture with anisotropic filtering. Its JSON manifest pins the source, builder
and output hashes. The PNG remains an unchanged authoring input.

### Exact asphalt prompt

Create an original seamless photorealistic PBR BASE COLOR texture for freshly resurfaced motorcycle racing circuit asphalt, to use in an open source 3D Thunderhill East racing game. Square 2048x2048 if possible. Orthographic top down, flat even diffuse illumination, no perspective or lighting gradient. Dense fine compacted dark warm neutral gray bitumen, tiny tightly embedded angular dark gray mineral aggregate, very occasional slightly lighter stone, extremely fine pores. Physical coverage about 0.8 by 0.8 meters, aggregate mostly 2 to 6 mm, fine smooth rolled surface, not gravel, not concrete. Restrained natural tonal variation, no broad cloudy patches, no large stones, no white speckle overload. Subtle nearly vertical fine compaction striations but no conspicuous lines. Perfectly tileable edges. NO lane paint, no cracks, no tire marks, no oil spills, no objects, no labels, no watermark. This is a game albedo material, not a photograph of a road scene; no baked reflections, no shadows, no vignette. Keep fine details sharp and realistic. Need original artwork not copied from a published texture.

## Flattened straw revision

`dry-cut-grass-v2.png` is the current ground and stubble albedo, copied to
`godot/assets/materials/dry_cut_grass_v2.png`. Runtime uses the sibling `.res`
with mip levels averaged in linear light, built by the existing color mipmap
baker; base level source pixels are preserved exactly. The built in imagegen tool produced
1254 by 1254 pixels despite a requested 2048 square. It used the 00:40 video frame
as appearance reference and v1 as an explicit negative reference. This is an
original generated material, not a scan or extracted video texture. Its connected
straw and soil regions remain more readable than v1 at riding height in the
matched station 400 and 3000 studies. A later scale comparison adopted one
metre rather than two metres to reduce oversized wiry foreground strands.
Runtime scale remains an appearance estimate. The stronger bundles can expose mirrored repetition; no
full lap temporal stability claim is made. Luminance derived relief and roughness
remain artistic estimates. V1 is preserved for comparison.

### Exact revision prompt

Create one original photorealistic game terrain albedo texture, square 2048x2048 pixels. Image 1 is appearance reference only for the dry mown field at Thunderhill East; do not reproduce the scene, viewpoint, motion blur, shadows, pavement or motorcycle. Image 2 is the CURRENT UNSATISFACTORY texture: it resembles wiry curled tangled fibers and evenly distributed speckles. Replace that character with a credible summer California racetrack field surface. Top down orthographic view of an approximately two metre square patch: mostly flattened short dry straw and cut brittle straight grass blades lying in loose irregular overlapping local swaths, interspersed with interconnected compact dusty brown earth openings. Weathered muted beige straw, medium umber dust, a few paler stems, fine irregular plant fragments. Roughly sixty percent straw and forty percent soil, arranged in uneven connected patches, not isolated round tufts. Blade lengths about 2 to 10 centimetres, thin natural widths, clearly resolved against fine soil in the gaps. Broad flattened clumps about 20 to 50 centimetres across with direction varying gradually across the patch. Absolutely no curled curly fibers, moss, hay bales, pebbles, large stones, green growth, roots, weeds, artificial noise, regular stripes or focal object. Flat neutral diffuse lighting suitable for PBR albedo, no directional shadows or highlights, no baked ambient occlusion, no perspective, no horizon. Seamless tileable edge to edge; no text or frame. This is an original artist texture, not a measured scan. Favor photographic surface structure and natural connected patches over uniform fine detail.

## Initial revision

`dry-cut-grass-v1.png` was generated with the built in imagegen tool using the
Ken Moto video frame at 00:40 as visual reference and the existing CC0 Poly Haven
Withered Grass albedo as a technical reference. It is an original generated
material study, not a measured scan or a crop from the video. Reference frames
remain in ignored inspection artifacts. The actual output is 1254 by 1254 pixels,
not the requested 2048 by 2048. No matched normal, height or roughness map exists.
It is installed as `godot/assets/materials/dry_cut_grass_v1.png`. The terrain
shader derives artistic relief and roughness from filtered luminance. These are
appearance estimates, not recovered material measurements.

The paired Godot study compares albedos at the same 2 metre tile size, with
identical camera, geometry, lighting and derived relief shader. The previous
unrelated grass normal and roughness maps are no longer used. Mirrored sampling
uses clamped edges and explicit gradients for mip selection at mirror folds.
Four smoothly blended world space patches rotate and offset the source to
break up its repeated bands. The texture reads more like continuous straw near
the camera, but repetition
and distance detail remain unresolved. It does not establish photographic realism.

## Exact generation prompt

Use case: photorealistic-natural. Asset type: original tileable diffuse/albedo texture for a realistic Godot racing game, dry cut grass beside Thunderhill East. Image 1 is VISUAL REFERENCE ONLY for the dry mown grass texture on the left side, its straw color, horizontal cut swaths and continuous short stubble. Image 2 is a technical texture reference for orthographic scale and detailed individual dry blades, not the desired random clump distribution. Generate an original square 2048x2048 top-down orthographic seamless grass ground ALBEDO texture representing a 4 metre by 4 metre patch: continuous densely cut short sun-bleached golden tan grass, fine broken straw and compressed stubble, subtle broad irregular near-horizontal mowing swaths and sparse exposed dusty brown earth. The texture must look photographic and naturally varied from millimetre blades through decimetre swaths. Neutral flat diffuse light, no baked directional shadows, no highlights, no perspective, no horizon, no motorcycle, no pavement, no trees, no green lush grass, no autumn leaves, no text or borders. Match the material character in the footage, not its motion blur, camera viewpoint or baked illumination. Edge-to-edge tileable, no dominant focal clump or regular striped pattern. This is a source material texture, not a rendered landscape.


## Fine shoulder soil study

`fine-shoulder-v1.png` is an original generated material using the Ken Moto
00:40 frame as visual reference only. Actual output is 1254 by 1254 pixels.
The runtime copy is `godot/assets/materials/fine_shoulder_v1.png`, imported
losslessly with mipmaps. No frame pixels are distributed as a source asset.
The source replaces coarse generic rocky mud beside the pavement. A one metre
scale, 1.5 mm luminance derived relief and roughness between 0.88 and 0.96 are
artistic estimates, not recovered measurements. Soil and grass share smooth
rotated world space patch sampling to suppress obvious tile repetition.
The old unrelated soil normal and roughness maps are no longer sampled.

### Exact shoulder generation prompt

Create an original photorealistic game terrain ALBEDO texture. The attached Thunderhill motorcycle onboard frame is visual reference ONLY for the short dry ochre grass and fine compact earth beside the pavement. Do not reproduce the frame. Output one square 2048 by 2048 seamless tileable top down orthographic diffuse texture representing about one square metre of a dry racetrack shoulder: predominantly finely compacted muted tan brown dust and very small sand grains, mixed with sparse tiny fragments of sun bleached cut straw lying flat and occasional very short stubble. No large stones, no gravel chunks, no cracks, no deep ruts, no leafy plants, no tall stalks, no road, no paint, no horizon. Fine irregular photographic microstructure, restrained natural color variation without prominent swirls or directional bands, no obvious focal features. Flat neutral diffuse illumination, no baked directional shadows, no highlights or ambient occlusion shadows. Edge to edge material texture, no text or border. This is an original artist material guided by the reference, not a measured scan.

## Short stem grass study

`dry-cut-grass-v3.png` is an original built in imagegen candidate. It was
requested as 2048 square; the returned image is 1254 square. No source image
was supplied to the generator. Its compact clumps and shorter straighter
stems are an artistic interpretation, not surveyed plant morphology or a scan.
SHA256: `1f8d534f83398401c9270a904193eabec73582f061b502a0d54447f05fb53eab`.

### Exact v3 generation prompt

Use case: photorealistic-natural. Asset type: original ground albedo for a realistic motorcycle racing game. Generate a square 2048x2048 top down orthographic texture of a one metre square of short mown, completely dry California annual grass. Small dense basal clumps of straight blunt cut straw stems, thin flattened tan blades and fragmented grass litter cover about seventy percent of the ground, with connected irregular patches of fine compact gray brown earth between them. Subdued natural pale beige and dusty brown colors. Millimetre thin stems mostly two to five centimetres long, some longer flattened blades, closely interlocked fine irregular texture. Crucially this is short cut grassy stubble, not tangled roots, curly hair, bundles of twigs, straw ropes or hay windrows. Photographic botanical detail, no regular stripes or repeating clumps, no large stones, leaves, green plants or flowers. Uniform neutral diffuse lighting suitable for an albedo texture, no directional cast shadows or specular highlights, no depth of field, no perspective or landscape, no text, no border. Edge to edge original seamless material study, not a reconstruction or measured scan.

## Broad straw swath study

`dry-straw-swaths-v4.png` is an original built in imagegen material study.
The exact prompt is in `dry-straw-swaths-v4-prompt.txt`. No image inputs were
supplied. It requested a 2048 square view of an eight metre patch; actual output
is 1254 square. Neither the stated scale nor plant morphology is measured.
SHA256: `9b3fbe80620dc8cc0eb44bc961a835927843bd16deeff4fe9fa336c9ec756ee4`.

The image has stronger connected straw and earth regions than v2. Godot tests
at four metre and two metre scales found oversized stems and repeated curved
patterns. Removing derived relief did not remove the pattern. Removing random
patch rotation changed the curves into overly regular straight rows. The source
is retained as a rejected study, not installed as a runtime material. This
isolates orientation and placement as an important next diagnostic; it does not
prove a mipmap bug. There are no measured matching PBR maps.

## Fine straw component study

`fine-straw-v5.png` is an original generated albedo component. Its exact prompt
is in `fine-straw-v5-prompt.txt`. The built in image generator received text only.
It requested a 2048 square image and returned 1254 square. SHA256:
`dbc8fc2939a5582710c97c9a1fdddb1a1d6a92c913e150b9a8109e9229895df7`.
The requested one metre coverage and stem dimensions are artistic estimates.
There are no measured matching height, normal or roughness maps.

The source has straighter shorter fragments and much less broad brightness
variation than v4. Paired Godot renders at stations 400 and 1065 found reduced
repetition but an overly uniform brown middle distance. It is retained as a
candidate fine detail layer, not installed as the production material. A separate
coherent straw coverage layer is needed before adoption; simply increasing
contrast would bring back the source repetition problem.

## Short cut straw v6 study

`short-cut-straw-v6.png` was edited with the built in image generator using
`dry-cut-grass-v2.png` as the target and `fine-straw-v5.png` as a fragment shape
reference. The exact prompt is in `short-cut-straw-v6-prompt.txt`. Requested
2048 square, returned 1254 square. SHA256:
`d2170c4d5a02da05d43823ae74ee8cf7cc7e3a5a9196ff79e20e3ddb7f9a668c`.
Scale, morphology and diffuse lighting are artistic estimates, not measured
material properties. A paired Turn 2 render with linear light mipmaps did not
show a convincing improvement in realism. Retained as a study only.
