# Separate sky appearance and lighting

The scene previously assigned `ambient_light_energy = 0.45` while using full sky contribution. Godot's [Environment reference](https://docs.godotengine.org/en/stable/classes/class_environment.html#class-environment-property-ambient-light-energy) explains that this setting affects the constant ambient component, not full sky illumination. The intended attenuation therefore had no effect.

`sky.gdshader` now applies `lighting_energy = 0.45` during `AT_CUBEMAP_PASS`. This scales the environment contribution to diffuse lighting and reflections together. The visible sky retains the original HDR values. Godot explicitly supports this separation in its [sky shader reference](https://docs.godotengine.org/en/stable/tutorials/shaders/shader_reference/sky_shader.html). The ineffective scene assignment was removed.

This is an artistic lighting balance correction, not calibrated radiometry. The HDR still contains the sun, and the scene also uses a directional sun. Solar energy has not been separated into independent measured components. Material albedo, camera exposure, weather and footage processing remain uncertain. This change does not establish photographic realism.

The actual 1920 by 1200 Vulkan Mobile Cyclone render was inspected before and after the change. The sky rectangle covering the top 350 pixels is byte identical. A foreground road rectangle from x 450 to 1450 and y 800 to 980 changes mean display RGB from (73.91, 73.61, 80.51) to (62.67, 60.61, 61.93), reducing the cool environment cast without changing asphalt material values. These are display pixel comparisons, not reflectance measurements. Render artifacts are `artifacts/grass-clumps-cyclone.png` and `artifacts/sky-lighting-cyclone.png`.
