# Separate sky appearance and lighting

The scene previously assigned `ambient_light_energy = 0.45` while using full sky contribution. Godot's [Environment reference](https://docs.godotengine.org/en/stable/classes/class_environment.html#class-environment-property-ambient-light-energy) explains that this setting affects the constant ambient component, not full sky illumination. The intended attenuation therefore had no effect.

`sky.gdshader` now applies `lighting_energy = 0.45` during `AT_CUBEMAP_PASS`. This scales the environment contribution to diffuse lighting and reflections together. The visible sky uses a separate background scale, described below. Godot explicitly supports this separation in its [sky shader reference](https://docs.godotengine.org/en/stable/tutorials/shaders/shader_reference/sky_shader.html). The ineffective scene assignment was removed.

This is an artistic lighting balance correction, not calibrated radiometry. The HDR still contains the sun, and the scene also uses a directional sun. Solar energy has not been separated into independent measured components. Material albedo, camera exposure, weather and footage processing remain uncertain. This change does not establish photographic realism.

The actual 1920 by 1200 Vulkan Mobile Cyclone render was inspected before and after the change. The sky rectangle covering the top 350 pixels is byte identical. A foreground road rectangle from x 450 to 1450 and y 800 to 980 changes mean display RGB from (73.91, 73.61, 80.51) to (62.67, 60.61, 61.93), reducing the cool environment cast without changing asphalt material values. These are display pixel comparisons, not reflectance measurements. Render artifacts are `artifacts/grass-clumps-cyclone.png` and `artifacts/sky-lighting-cyclone.png`.

## Background exposure study

The visible sky now uses `background_energy = 0.7`, while the cubemap lighting
remains at 0.45. A fixed camera study in `preview_lighting.gd` compared filmic
and ACES tone mapping, reduced exposure, cooler sunlight, reduced fog and
background only attenuation. ACES exaggerated red paint saturation; cooler
sunlight made the pavement less like the supplied footage. Those alternatives
were not adopted.

The isolated background comparison is in `artifacts/lighting-isolated/`. Across
the top 140 rows, mean displayed RGB changed from (206.11, 214.61, 224.35) to
(189.88, 198.78, 210.34). Road crop (10,520) to (300,570) and tank crop (530,660)
to (760,715) remained byte identical. This verifies the intended separation in
that rendered view, not full radiometric fidelity.

`fetch_sky.py --candidate` downloads hash pinned CC0 sky studies into ignored
artifacts, leaving the production sky unchanged. The inspected
[Aristea Wreck](https://polyhaven.com/a/aristea_wreck_puresky) and
[Kloppenheim 05](https://polyhaven.com/a/kloppenheim_05_puresky) alternatives did
not provide a convincing cloud match. They are not installed in the game. The
current sky still differs from the footage's cloud structure.

## Layered cloud study

The publisher API supplied a verified 2K HDR and MD5 for
[Kloppenheim 03](https://polyhaven.com/a/kloppenheim_03_puresky). The acquisition
CLI now supports this additional candidate. It stays in ignored artifacts;
production still uses Kloofendal. `preview_lighting.gd --camera=1` provides the
rider eye comparison, and validates camera identifiers before loading the game.

Matched rider views are in `artifacts/current-sky-rider/` and
`artifacts/kloppenheim03-rider/`. The candidate replaces rounded clouds with
thinner layers, but puts a broad pale veil where frame 00:20 shows larger blue
openings and fine broken streaks. Reducing background energy to 0.4 darkened
the scene background without solving cloud coverage. This is not adopted.

An optional `background_saturation` shader uniform defaults to 1.0, preserving
production color and environment lighting. The study variant at 1.6 is in
`artifacts/kloppenheim03-chroma/sky_chroma.png`. It improves the blue opening's
color but cannot correct the cloud placement. Saturation operates only outside
the cubemap pass; no lighting or reflection color grading is introduced.
Neither this variant nor the candidate sky is a demonstrated footage match.

## Original cirrus panorama

The current production sky is the original generated
[`cirrus-v1.png`](../assets/source/sky-studies/cirrus-v1.png), with its complete
prompt and provenance in the adjacent README. The built in imagegen tool
received descriptions of cloud morphology observed in the supplied video,
not a video frame as image input. The actual output is 1774 by 887 pixels,
despite the requested 3840 by 1920. It is LDR color, not recovered HDR radiance.

Before authoring it, a new reference search inspected 20 official thumbnails.
The publisher API verified the pinned 2K files for
[Cloud Layers](https://polyhaven.com/a/cloud_layers) and
[Cayley Lookout](https://polyhaven.com/a/cayley_lookout). Cloud Layers was tested
with its solar azimuth aligned to production. Its source trees extend above
the attempted eight degree latitude cutoff, producing vertical smearing in
`artifacts/cloud-layers-rider/production.png`. This candidate was rejected.
Cayley has useful filaments but a low sun and captured landscape. It was not
adopted. The acquisition CLI now stores all HDR studies in ignored artifacts,
including its default historical sky, so it cannot overwrite production sky
metadata merely by acquiring a reference.

The original panorama has elongated feathered bands and broad blue openings.
Root and independent review find its cloud morphology substantially closer
than Kloofendal's bulky cumulus. Reference cloud density, bright left side haze,
deep blue gaps and fine fragments still differ. The generated clouds remain
somewhat soft and painterly. This improves cloud type, not weather reconstruction.

The first LDR radiance scale of three was too bright and blue. Scale one is
adopted for both background and environment, with the existing 0.7 background
and 0.45 lighting multipliers. The direct sun direction is retained from the
previous setup; the generated texture has no solar disk. Shader conversion
decodes sRGB explicitly. The runtime `.res` uses the existing linear color
mipmap builder, extended with explicit asset source/output arguments, and
contains ten mip levels. The raw generated PNG is preserved unchanged.

At station 2000, the opposite view revealed a narrow segmented seam. A four
percent periodic overlap made the source endpoints continuous but did not
remove the artifact. Longitude's wrap was still entering implicit derivatives
as a full texture width. The shader now wraps the longitude derivatives to
their shortest periodic difference and uses `textureGrad`; derivatives also
follow the overlap scale. `artifacts/cirrus-runtime-opposite/production.png`
was inspected by root and independently: the seam is absent without an obvious
replacement stripe. This is a sampled view check, not all angle motion coverage.

The initial lower hemisphere was blue, causing conspicuous blue lower cockpit
reflections. The environment pass now blends into estimated ground radiance
below the horizon, using the shared dry ground linear tint scaled by 0.35.
This is an artistic distant ground approximation, not local reflection probes
or measured bounce light. Background sky shading does not use this blend.
`artifacts/cirrus-ground-cockpit/production.png` shows reduced blue rims on the
reservoirs and hardware. Root inspected the final cockpit render. The top 80
rows differ from the preceding capture by less than 0.04 display byte values
per channel on average; they are not byte identical.

Human controls passed after the final lighting adjustment. Linux timing was
median 17.361 ms and p95 20.854 ms over 264 frames, not a native Mac benchmark.
Physics, agent controls, reward data and track contact surfaces are unchanged.
The game still has substantial geometry, vegetation and lighting fidelity gaps.
Mac export passed as `0da228c4819a-288176de9df8`; native execution and performance
remain unverified for this build.

## Isolated fog correction

The distant pale ridge was partly an atmosphere setting, not a grass texture
resolution problem. The surrounding measured hills share the terrain material.
An isolated comparison reduced fog density from 0.00035 to 0.00008 while holding
camera transform, sky texture, sun direction, exposure and material settings
fixed. These are appearance estimates, not reconstructed weather measurements.

Paired captures are `artifacts/fog-baseline-turn2/` versus
`artifacts/fog-isolated-turn2/`, `fog-baseline-crest/` versus
`fog-isolated-crest/`, and `fog-baseline-wide-view/` versus
`fog-isolated-wide-view/`. They use stations 950, 3000 and 2000 respectively.
All paired metadata was checked for identical camera, sky and sun, and identical
variant parameters other than fog density. Root and independent review accepted
the lower density: the cream colored ridge becomes dry brown, with no obvious
loss of useful depth or excessive saturation. The wider view makes the existing
simple silhouette more visible. It does not add missing landscape layers.
Production now uses the lower density.

`preview_lighting.gd --fog-density` permits isolated finite values from zero to
0.002 and records the effective value in each variant. Without this override,
the production variant reads the runtime environment fog density. Nonfinite,
negative and excessive inputs were rejected. Human controls passed with zero
failures; local Linux timing was median 17.316 ms and p95 26.253 ms over 263
frames. This is a functional check, not a native Mac performance certification.
Mac export completed as `2e1f357b9c31-ec09e30262d5`; native execution of this
build remains unverified.

### Separate horizon registration defect, correction pending

Code inspection found `tools/build_horizon.py` adding EPSG:6339 game origin
coordinates directly into an EPSG:26910 raster without applying the pinned
horizontal operation used by detailed terrain. The same mislabeled coordinates
reach coarse raster reprojection, and boundary clipping mixes these systems.
Both independent audit and root execution of `terrain_transform()` confirmed
an origin adjustment of +0.255285 m east and minus 0.446407 m north, a horizontal
magnitude of 0.514247 m. This is a real registration defect but does not explain
the large color mismatch. Its vertical effect has not been measured.

The builder and horizon data remain unchanged in this fog pass. The next terrain
correction must transform sample coordinates before constructing raster windows,
check transformed pixel bounds explicitly, use the corrected coordinates in the
coarse reprojection, record the pinned registration metadata, and regenerate the
horizon. The existing analytic ramp datum test is the closest verification
pattern. This defect concerns surrounding visual terrain, not driving contact.

### Horizon datum correction applied

The subsequent terrain pass fixes the separate registration defect above.
`build_horizon.py` now uses the pinned horizontal operation for both the raster
coverage envelope and every local grid sample. Coarse source reprojection starts
from those corrected EPSG:26910 coordinates. Both raster resolutions use a shared
window sampler that rejects coordinates outside pixel center support instead of
extending boundary heights. The generated metadata records the operation and
verified grid hashes. No vertical datum conversion or synthetic heights were added.

Regeneration from the two documented USGS URLs retains the 160 by 171 grid,
32 metre spacing and local bounds. Of 27,360 samples, 20,365 use fine lidar and
6,995 use the documented coarse source. Compared with the previous committed
heights, median absolute change is 0.019 m, the 95th percentile is 0.115 m,
and signed extrema are minus 0.618 m and plus 0.612 m. These are changes in
sampled visual terrain, not measurements of improved driving accuracy.

The analytic datum suite passes all four tests, including a nonsquare raster
with unequal pixel dimensions, boundary pixel centers and rejection beyond
all four boundaries. Root inspected `artifacts/horizon-datum-wide/production.png`
against `artifacts/fog-isolated-wide-view/production.png`: no obvious silhouette
or seam regression, and no substantial visual realism improvement is claimed.
Human controls pass with zero failures (Linux median 17.361 ms, p95 18.653 ms,
266 frames). Material uniformity, scenery detail and distant landscape structure
remain visible fidelity gaps.
Mac export passed as `db5b145bdd56-c7bbdf87a84e`. The first packaging process
terminated with signal 15 after the engine export stage, before writing its
manifest. After confirming that process had ended, a fresh packaging invocation
completed the archive checks and provenance manifest. Native Mac execution
remains unverified for this build.

## Photographic 38 degree cloud study

The verified CC0 candidate
[Kloofendal 38d Partly Cloudy Pure Sky](https://polyhaven.com/a/kloofendal_38d_partly_cloudy_puresky)
by Greg Zaal and Jarod Guest is now reproducible through
`uv run tools/fetch_sky.py --candidate kloofendal_38d_partly_cloudy_puresky`.
This is distinct from the previously evaluated 48 degree source. The publisher
API pins the 4096 by 2048 HDR to MD5 `f0e9e19f824767c92f36d2af7ae605b8`;
the downloaded 20,462,635 bytes passed that check and decoded at the expected
resolution. Existing candidates retain their pinned 2K downloads.

`artifacts/sky-38d-rider/production.png` uses the existing station 950 artistic
rider pose (camera 1, lean minus 25 degrees, optical roll minus 14 degrees,
lateral 4 m and yaw 8 degrees). The source is rotated to match production sun
azimuth. This changes ambient radiance and sun elevation, so it is a combined
sky and lighting study, not an isolated background comparison.

The photographic clouds resolve more sharply than the generated panorama,
but the rendered sky has excessive continuous cloud coverage and pale blue
openings compared with reference frame 40. The supplementary
`artifacts/sky-38d-exposure/` study holds fog at production 0.00008. Root
inspected the background energy 0.4 and saturation 1.6 variants as well as the
production settings. Neither restores the required open blue areas and thin
elongated bands. This candidate is not adopted. Sharper source detail alone
is insufficient evidence of a closer reference match.

Both Godot study runs exited successfully with nonblack captures. The fetcher
passed Python compilation and diff checks. No production game assets, shaders,
physics or camera protocol changed; no new native build is claimed.

## Direct video reference panorama study

A second original panorama now uses frame 40 directly as an image reference,
with v1 supplied only as the projection layout reference. Its exact prompt
and unchanged generated PNG are retained in `assets/source/sky-studies`.
The generator again returned 1774 by 887 despite the requested 3840 by 1920.
That supplies only about 4.93 source pixels per degree before seam overlap.
It does not supply high resolution detail simply because the prompt asks for it.

`artifacts/cirrus-v2-baseline/production.png` and
`artifacts/cirrus-v2-rider/production.png` use the same station 950 rider pose.
Camera transform, FOV, sunlight direction, exposure, fog, panorama radiance
scale, seam overlap and ground radiance match exactly in their metadata.
The texture changes both background and environment illumination. Root and
independent visual review reject v2: it introduces broad soft opaque white
patches rather than the reference's finer translucent filaments. Increased
coverage is not evidence of improved photographic realism.

The preview previously generated ordinary encoded mipmaps for LDR sky inputs,
unlike the production linear light builder. It now uses the same existing
builder for sRGB inputs and checks generation failures for both encodings.
To validate the preview against runtime loading, the original v1 PNG was
loaded through the candidate path with matching source metadata. The complete
1280 by 720 capture in `artifacts/cirrus-v1-linear-control/production.png`
is pixel identical to the production baseline. This verifies that this
preview reproduces the production texture path for the tested pose. The v2
cloud enlargement is therefore not a candidate versus production mapping
difference. Source morphology and limited angular detail remain the next
authoring constraints; exact video registration is still unproven.

All three Godot rendering runs completed successfully with nonblack captures.
Formatting and diff checks passed. No production asset or shader was changed
in this study, and no new native build is claimed.

## Rectilinear cloud detail, 21 September 2026

Selective cloud radiance and 180 degree panorama rotation studies are retained
as diagnostic controls but not adopted. Gain three alone barely changes the
sparse viewing sector; rotating the panorama reveals broad smooth ribbons, and
the gain washes out their bright regions. Captures are `cloud-radiance-control`,
`cloud-radiance-three`, `cloud-yaw180` and `cloud-yaw180-neutral` in artifacts.
The production radiance gain remains one and panorama yaw remains zero.

The accepted new `cirrus-patch-v1` source allocates 1536 by 1024 pixels to one
rectilinear sky region. It resolves fragmented cloud structure that the 1774
pixel full panorama did not supply at riding view scale. A shared projection
helper maps it into world directions and derives vertical field of view from
image aspect. The shader fades all four edges and rejects the back hemisphere.
Derivatives are computed before spatial coverage branching. It uses the same
linear light mip construction and manual sRGB decode as the original sky.
The base panorama supplies every direction outside the patch.

At station 950, `cirrus-patch-v1-render/production.png` is visibly more detailed
than `cloud-radiance-control/production.png`. The 1.5 cloud gain variant loses
fine white shading and lightens the road, so it is not adopted. The production
resource capture `cirrus-patch-v1-production/production.png` is pixel identical
to the accepted loose source preview. This verifies the runtime texture path
for that pose, not exact matching to the video.

Additional views at station 400 (`cirrus-patch-v1-straight-repeat`) and station
1450 (`cirrus-patch-v1-exit`) show no obvious hard blend boundary. The first
station 400 process exited with signal 15 before saving; after its terminal
status was confirmed, a fresh invocation completed in the separate repeat
directory. No cause for that external termination is established.

The arrangement, projection and lighting remain artistic estimates. Other sky
directions still use the lower detail panorama. Finite angular view checks are
not a full moving lap seam or performance acceptance test. The cardinal and
polar projection test verifies handedness, aspect and a finite orthonormal basis.
`preview_lighting.gd --sky-patch-off` gives a control without the production
patch; `--sky-patch=/absolute/manifest.json` provides a validated alternative.
Study metadata records enabled state, production metadata hash, helper hash and
any override's source and projection. Physics and observation dimensions are
unchanged.

Human control checks passed with zero failures. The rendered agent camera suite
passed all five recorded observations, including PNG hashes, immutable artifacts,
frozen ticks, queued advance/reset ordering and headless capture rejection.
These are compatibility checks, not an RL training result or visual approval.
Comparison sheets are in `artifacts/cirrus-patch-comparison/`.

## Reference guided solar azimuth

The inherited Kloofendal light direction had geographic azimuth 145.81 degrees
and elevation 47.90 degrees. The dense reference sequence now supplies a more
useful direction constraint: frames 30 and 32 show forward illumination and
pavement glare while riding the mapped Turn 1 exit straight, whose heading is
approximately 106 to 110 degrees. The southbound main straight at frame 20 has
illumination toward the left, providing a separate consistency check.

The controlled 110 degree trial centers the reflection more like the reference;
the inherited light places it distinctly to the right. Root and independent
review support 110 degrees as a working estimate. It is now the default in
`godot/data/sky.json`. Elevation, intensity, sky texture, environment radiance,
materials and camera settings remain unchanged. Elevation is explicitly
uncalibrated; clouds, clipping, gaze and lens distortion prevent exact recovery.

`preview_lighting.gd --sun-azimuth-deg` independently overrides direct sunlight
azimuth while preserving elevation. It validates the finite [0, 360) interval
and records the override and effective direction. This does not rotate the
separately authored cloud panorama. Evidence is in `artifacts/solar-110-straight`,
`solar-baseline-straight`, `solar-direction-comparison` and the default apex
capture `solar-corrected-apex`.

The preceding station inference from shadow onset is weakened: an incorrect
solar direction can make a shadow enter the view too early. Reevaluate turn
phase under the corrected estimate before treating station 1150 as later than
video frame 40. This adjustment does not establish full photographic realism.

Paired capture metadata confirms identical camera transform, FOV, material
shader hashes, panorama source and energy, cloud gain and solar elevation.
Human controls passed with zero failures; local Linux frame intervals were
17.269 ms median and 18.315 ms p95 over 270 samples. Five agent camera observations
passed the existing capture checks in `artifacts/solar-corrected-camera-qa`.
Mac export `74bd8a5fb095-af94ae03c423` completed; native laptop testing is pending.

## Circumsolar brightness experiment

The optional `preview_lighting.gd --solar-haze=0..8` adds an authored angular
brightness lobe to both the visible sky and its environment radiance. This is
an appearance experiment, not measured atmospheric scattering. Its shader
default is zero and the playable game does not enable it.

Strength three made little visible difference at stations 850 and 1150.
Strength eight whitened the upper edge of the straight capture, but failed to
reproduce the reference frame 30 brightness extending toward the horizon.
Root and independent review rejected promotion. Evidence is in
`artifacts/solar-haze3-straight`, `solar-haze3-apex`, and `solar-haze8-straight`.
The reference is `artifacts/reference/ken-moto-turn2-avc/frame-0030.00.jpg`.

The preview now records the direct sun direction projected into the camera.
At station 850, camera two, the projection is (682.89, -472.21) pixels for a
1280 by 720 capture. It is in front of the camera but above the image. This
explains why raising this lobe's strength mainly affects the upper edge.
It does not recover the true solar elevation from the footage: camera pitch,
lens, cloud placement, exposure and bloom remain uncalibrated. The next lighting
comparison must resolve that registration rather than simply increase gain.

The projection describes a direction, not actual raster visibility or occlusion.
Editor import and the real Vulkan preview completed successfully. Formatter
and whitespace checks passed. This experiment does not change physics, agent
observations, or the default sky appearance.
