# Native circuit benchmark

Pass `-- --benchmark-recording=/absolute/path/episode.jsonl` to the game executable. Use a complete successful episode from `tools/check_lap.py` with the same track, physics version, parameters and timestep.

This diagnostic feeds recorded controls through live physics and renders the resulting motion. It measures actual wall clock frame intervals, records a new episode, compares positions with the source trajectory, and exits with a `BENCHMARK_RESULT`. A successful result requires a completed lap without a crash, simulation error or off track tick. Performance must be assessed separately from that functional result.

Use a rendered native build at the intended resolution, without `--fixed-fps`. Headless or accelerated runs verify mechanics only. Startup is excluded from frame statistics. Report the resolution, build provenance, wall time, simulated time, frame percentiles and trajectory error. Other running game instances may affect performance.

The input fixture is a privileged diagnostic controller run, not an LLM policy or training result. State playback remains a separate `--replay` mode.

Local validation on 2026-09-20 completed 40,313 physics ticks, one lap and 335.941667 simulated seconds with zero off track ticks and zero position difference from the input fixture. This was an accelerated headless contract check, not a Mac performance measurement.

## Native Mac result, 20 September 2026

Build `1709b61aad29-c6bac0bbda50`, clean source commit `1709b61aad2989898047ea121afd735b45dd38e1`, ran through Metal 4 on the owner's M3 Pro. It completed one valid lap with 40,313 live physics ticks, zero off track ticks, no crash and zero position difference from the Linux fixture. Simulation time was 335.941667 seconds and measured wall time was 335.904157 seconds.

Across 40,181 process frame intervals, median was 8.314 ms, p95 9.954 ms, p99 10.871 ms and maximum 86.977 ms. These are actual wall clock callback intervals, not GPU timings or proof of displayed frames. Startup was excluded. Resident memory samples during the run were approximately 373 to 437 MiB, not a measured peak. Another review instance remained open.

The launch requested a 1920 by 1200 window; the benchmark reported a 1600 by 1000 logical viewport. Window size and render target pixel size were not independently measured. macOS reported the display online, but screen capture failed and foreground presentation could not be established. Therefore this validates native circuit execution and process pacing, not visible full circuit performance acceptance. Physical human controls and visual presentation still require review.

The source fixture SHA256 is `bdc94c8c880b16ba0e083192b9cb5f41190503b07a9b85ba3e7e13ed8c15eb2b`. Local raw evidence is `artifacts/mac-full-circuit-benchmark.log`; the Mac log is in `~/Downloads/Thunderhill-full-circuit-benchmark.log`. Installed app: `~/Applications/ThunderhillReview/1709b61/Thunderhill.app`.

Clock mode must be established from the actual launch command. Godot consumes engine flags before exposing script arguments, so the original automatic `clock_mode` label was unreliable and has been removed. The native run above used no fixed FPS flag; the local accelerated regression did. Always compare simulated and wall seconds as well.

The subsequent terrain build and full native rider view run are documented in [conforming ground](conforming-ground.md), including an unresolved 582 ms maximum frame interval. Do not use the earlier benchmark alone to certify the later build.

## Capture attribution

Build `b1205bc` adds a common monotonic clock for screenshot stages and process frame intervals. Reports retain every interval of at least 50 ms, with its index, start, end and duration. Capture events separately measure waiting for the render completion signal, GPU image readback, and PNG encoding plus file writing. The wait includes normal asynchronous rendering; its duration is not itself evidence of a blocked main thread. Readback and PNG saving execute synchronously. Compare their timestamp ranges to long frame intervals before attributing a stall. These measurements remain process callback intervals, not display presentation or GPU timestamp queries.

The instrumented M3 Pro run of `b1205bc91e3c-a249e005a559` completed one legal lap with 40,313 ticks, zero trajectory error, no off track ticks and no crash. Its screenshot readback took 15.676 ms and synchronous PNG saving took 614.651 ms. Both overlap frame interval 14 (645.280 ms), proving that screenshot saving blocked this frame. A separate first interval lasted 677.956 ms before screenshot capture began; another interval lasted 83.405 ms at 246.293 seconds. Their causes remain unresolved. The prior build's 582 ms maximum cannot be retrospectively attributed with certainty.

Median was 10.138 ms, p95 14.739 ms and p99 16.979 ms over 32,230 intervals. These are process callbacks rather than display presentation. Other review instances remained open, so this is not an isolated device benchmark. Raw evidence is `artifacts/mac-timing-benchmark.log` and its retrieved 1920 by 1200 PNG. The benchmark label now explicitly includes render warmup rather than calling all startup excluded.

Screenshot saving now transfers the captured CPU image to one worker thread. GPU readback remains on the main thread. The worker does not access the scene, renderer or simulation; completion is joined before reporting the save or exiting preview mode. Shutdown joins any pending worker. This follows Godot's [thread lifecycle](https://docs.godotengine.org/en/stable/tutorials/performance/using_multiple_threads.html) and [resource ownership rules](https://docs.godotengine.org/en/stable/tutorials/performance/thread_safe_apis.html). This affects review screenshots only; the agent camera's exact tick capture contract remains unchanged.

## Worker verification on the Mac

Clean build `ff55d1fc3a64-59d626e0b262` completed the same native full circuit with the screenshot request. All 40,313 ticks matched the reference trajectory exactly, with one legal lap, zero off track ticks and no crash. PNG saving took 538.540 ms on its worker from 228.970 to 767.510 ms after benchmark start; no process interval of at least 50 ms overlapped capture or saving. GPU readback remained synchronous at 16.910 ms. The retrieved PNG was visually inspected at 1920 by 1200.

Across 31,951 process intervals, median was 10.213 ms, p95 14.938 ms, p99 17.356 ms and maximum 60.695 ms. Wall time was 335.902994 seconds. The only interval above the diagnostic threshold began at 246.358645 seconds. This is close to the previous instrumented run's late stall at 246.293160 seconds, and warrants separate investigation of location dependent work. No cause is established yet. The large initial interval from the previous run did not recur; that alone does not establish a startup fix.

Evidence is `artifacts/mac-worker-benchmark.log` and `artifacts/mac-worker-benchmark.png`. The installed build is `~/Applications/ThunderhillReview/ff55d1f/Thunderhill.app`. Existing human review windows remained open. This run demonstrates removal of the synchronous PNG saving stall, not certified display pacing, final physical realism or human control acceptance.

## Continuous wall contact build

The universal app from clean commit `271a52caadc3fb4ad57e25b19fa6b9edc3064517`, build `271a52caadc3-dff12e1d81cf`, was installed on the owner's M3 Pro. The received archive SHA256 matched `36652227ffd441d798fba8c774a386f173cebcf9c01a270039eff18217fa05df`, and macOS code signature verification passed.

The shared native control check exited successfully with `HUMAN_CONTROLS_CHECK failures=0`. It exercised injected throttle, steering, both brakes, camera changes, pause and reset through the exported game. The 72 process frame intervals had median 66.667 ms and p95 74.723 ms. The screen was locked and six older review instances were open. Startup took several minutes; a process sample captured a wait inside Metal compilation during texture upload, while a subsequent sample showed progress beyond that wait. This does not isolate the full startup delay or establish acceptable foreground performance.

Evidence is `artifacts/Thunderhill-controls-271a52c.log` and `artifacts/Thunderhill-startup-sample-271a52c.txt`. Installed app: `~/Applications/ThunderhillReview/271a52c/Thunderhill.app`. These checks do not include a native wall impact benchmark or physical keyboard delivery.

## Imported pavement source packaging failure

Native review of clean build `97903d85ffde-c13d8d829fb8` found an actual startup
failure despite a successful export. The received archive matched SHA256
`41627771f2d77508df060fcb145d238b11e04daf039ca02c5d20d9f96d2cdede`.
Metal initialized, then the game exited with status two and
`Historical pavement tone source mismatch`.

The integrity check hashed the original `pavement_tone.png`, which exists in the
editor project but is replaced by an imported texture in native packages.
[Godot's import documentation](https://docs.godotengine.org/en/4.7/tutorials/assets_pipeline/import_process.html)
explicitly distinguishes resource loading from raw access to imported sources.
Export success alone did not establish that this game could launch.

The fix bakes the numerical map into an explicit `ImageTexture` resource with a
separate runtime manifest. The bake uses linear numerical mip averaging, not
color conversion, and checks that saving and loading preserve the base pixel
bytes. Runtime verifies the shipped resource hash, its source identity, and the
track hash. Packaging additionally verifies original PNG, runtime resource,
builder and track against their manifests. Integrity checks remain enabled.

The package validation suite includes changed source pixels, changed runtime
bytes, changed generator and track, and missing or malformed manifests. Nine
package tests passed. Optional scanned ground study tools still rely on loose
authoring images and are not supported by the ordinary native game entry point.

The corrected candidate `97903d85ffde-eb3c3db27afe` transferred with matching
archive SHA256 `5eebbf60212d24ca73b697552c9ae397552f056957c4d2e4a680f385e23b72ae`.
The actual exported app launched on the M3 Pro using Metal 4, reached
`THUNDERHILL_READY`, captured the onboard straight, and exited successfully.
The retrieved 1152 by 720 image was visually inspected. Evidence is in
`artifacts/mac-eb3c3db-straight.png`, its JSON sidecar and log. macOS constrained
the requested 1280 by 720 window; the recorded dimensions are authoritative.

Native controls passed with zero failures, and code signature verification
passed. The 551 process intervals report median and p95 of 8.333 ms, but the
screen was locked and the window unfocused. These values are not foreground
presentation or sustained lap performance certification. The local rendered
controls also passed; rebuilt scenery and landmark geometry hashes are unchanged.
