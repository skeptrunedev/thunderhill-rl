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
