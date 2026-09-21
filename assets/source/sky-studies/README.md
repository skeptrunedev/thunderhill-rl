# Original cirrus sky study

`cirrus-v1.png` is an original image generated with the built in imagegen tool.
It is a candidate LDR panorama, not measured HDR radiance or a photograph of
Thunderhill. The prompt uses cloud morphology described after visual inspection
of the supplied onboard footage. No video frame was supplied to the generator.
The intended dimensions were 3840 by 1920; the actual PNG is 1774 by 887.
The generated file is preserved unchanged. SHA256:
`18211d01c6799b3be44d69d84186509c6dac0f0780d95878b531d4be71894992`.
The original generated asset is distributed under the project's MIT license.
The runtime copy is `godot/assets/sky/cirrus_v1.png`; the `.res` alongside it
contains mipmaps filtered in linear light. The sky shader supplies periodic
edge blending. No solar disk or recovered radiometric calibration is present.

Prompt:

Use case: photorealistic-natural. Asset type: original seamless equirectangular pure sky panorama for a realistic motorcycle racing game at Thunderhill in dry California midday. Create a 3840 by 1920 image, exactly 2:1 longitude latitude spherical projection covering 360 degrees horizontally and 180 vertically. Top row is zenith, middle row is horizon, bottom row is nadir. Upper hemisphere: natural medium blue sky with large blue openings and fine white feathery cirrus filaments, elongated broken sheets of delicate high cloud, some fine mottled cirrocumulus, approximately 45 percent cloud cover. Cloud forms should look photographed, slender and translucent, never rounded puffy cumulus. These visual properties are based on inspected motorcycle onboard footage: bright wispy cloud layer upper left, deeper blue gaps overhead and to the right, hazy pale blue horizon. No warm sunset. No dramatic storm. No terrain, no mountains, no trees, no buildings, no road, no objects, no text or watermarks. No visible solar disk or lens flare; directional sunlight will be supplied by the game. Below the horizon continue a softly uniform pale atmospheric blue, with no objects or clouds. Smooth left/right seam with matching edge color and cloud continuity, no borders. Top polar region must converge smoothly without a pinched radial star. This is a game texture, not a screenshot or landscape picture. Original cloud arrangement, not a copied photographic frame.

## Direct reference v2 study

`cirrus-v2.png` was generated with the built in imagegen tool. Image inputs
were `artifacts/reference/ken-moto/frame-0040.00.jpg` for sky morphology and
`cirrus-v1.png` for panorama projection layout. The exact prompt is saved in
`cirrus-v2-prompt.txt`. Requested 3840 by 1920, returned 1774 by 887. SHA256:
`575c75ef2dd29b9a6647481fe9d102c99d64cf2f3c3f1357862f35ed513e07ba`.
This is an original generated LDR study, not measured weather or HDR radiance.
The video frame is a reference, not included as a runtime texture.

Rejected after the rendered comparison: larger opaque cloud patches were
softer than the reference's fine translucent streaks. It is not installed
as a production asset. See `docs/sky-lighting.md` for the control experiment.

## Rectilinear cirrus detail patch v1

`cirrus-patch-v1.png` is an unchanged original generated image from the built in
imagegen tool. Frame 40 was supplied as a morphology reference. The exact prompt
is in `cirrus-patch-v1-prompt.txt`. The actual image is 1536 by 1024 RGB, SHA256
`5a6022b36601e29483cae5f429b707cc43cb56b9b47ef2a6f40c031775a2a004`.
It is distributed under the project's MIT license as an original asset.

Unlike v1 and v2 panoramas, this is a rectilinear view. Its pixels cover a
limited 100 degree horizontal field instead of all 360 degrees, allowing much
finer visible cloud structure. The prompt's elevation is an artistic composition
request, not recovered image calibration. Runtime placement uses estimated
azimuth 82.5 degrees and elevation 30 degrees, with a 0.15 normalized edge
feather. These are visual layout choices, not captured Thunderhill weather.

The runtime PNG is `godot/assets/sky/cirrus_patch_v1.png`. Its sibling `.res`
contains linear light filtered mip levels built with the existing color bake
tool. `godot/data/sky.json` records production projection and provenance.
The shader blends the patch into the original panorama in fixed world space,
for both visible sky and environment light, while retaining the existing direct
sun direction and lower hemisphere ground radiance. This remains LDR authored
appearance. No photographic radiometry is claimed.
