# Scene startup measurements

The game accepts `--startup-profile=/absolute/path/profile.jsonl` after Godot's `--` separator. It writes and flushes a JSON line after each completed loading stage, so a slow subsequent stage can be distinguished from a stalled process. A requested profile path that cannot be opened fails explicitly. Ordinary gameplay performs no profile file writes.

Timing begins in the main scene's `_ready` method. It excludes earlier engine startup, executable loading and imports. `first_frame_post_draw` means RenderingServer finished a viewport update, not that the operating system presented that frame on the physical display. Headless runs end their profile at scene readiness.

A local rendered baseline measured 43.171 seconds through the first viewport update. Its completed stages identified landmark construction as the dominant cost at 31.136 seconds, compared with 2.218 seconds for the track, 0.854 seconds for the horizon and 8.599 seconds for scenery. The trace is `artifacts/startup-local.jsonl`.

Mapped paving is recursively subdivided into triangles. Adjacent triangles repeatedly query ground at identical vertices. Landmark construction now caches each exact sampled x/z position during its synchronous build, stores unlifted height and applies the requested lift separately. It clears the cache before and after construction. This preserves sampling and geometry rather than approximating terrain or reducing pavement detail.

The independent landmark build diagnostic measured 31.403 seconds before and 6.064 seconds after. Both complete geometry digests were `38e9d7a41d5998bdad20b5b359fffa102b876562c3908f2363db9eeb387a9ae4`. The diagnostic hashes every mesh surface array, node transform and MultiMesh instance transform/color. It does not hash materials, which the optimization does not modify. Run it with `godot --headless --path godot --script tests/check_landmark_geometry.gd`.

A subsequent actual rendered control diagnostic completed scene startup through the first viewport update in 19.302 seconds, with zero control failures. Its landmark stage measured 7.328 seconds and scenery 8.509 seconds. The editor import ran concurrently during part of this second check, so it is not a controlled microbenchmark; the separate geometry diagnostics establish the isolated comparison. The rendered trace is `artifacts/startup-cached.jsonl`. Startup is still too slow for the desired experience, especially before native measurements establish the remaining costs.
