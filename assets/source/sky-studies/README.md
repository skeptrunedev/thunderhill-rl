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
