# Dry cut grass study

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
