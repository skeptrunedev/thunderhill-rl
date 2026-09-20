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
