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
