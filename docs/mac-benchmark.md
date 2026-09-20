# Native circuit benchmark

Pass `-- --benchmark-recording=/absolute/path/episode.jsonl` to the game executable. Use a complete successful episode from `tools/check_lap.py` with the same track, physics version, parameters and timestep.

This diagnostic feeds recorded controls through live physics and renders the resulting motion. It measures actual wall clock frame intervals, records a new episode, compares positions with the source trajectory, and exits with a `BENCHMARK_RESULT`. A successful result requires a completed lap without a crash, simulation error or off track tick. Performance must be assessed separately from that functional result.

Use a rendered native build at the intended resolution, without `--fixed-fps`. Headless or accelerated runs verify mechanics only. Startup is excluded from frame statistics. Report the resolution, build provenance, wall time, simulated time, frame percentiles and trajectory error. Other running game instances may affect performance.

The input fixture is a privileged diagnostic controller run, not an LLM policy or training result. State playback remains a separate `--replay` mode.

Local validation on 2026-09-20 completed 40,313 physics ticks, one lap and 335.941667 simulated seconds with zero off track ticks and zero position difference from the input fixture. This was an accelerated headless contract check, not a Mac performance measurement.
