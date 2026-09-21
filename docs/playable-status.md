# Playable implementation status

September 20, 2026. The requested end state remains an attractive, hyperrealistic Thunderhill East motorcycle game reviewed on the owner's MacBook. The first implementation is a working prototype, not acceptance of that goal. A small Gemma 3 launch probe has now verified GPU adapter updates from game rewards; no racing policy training or E4B training has started. See [local training validation](../training/README.md).

## September 21 closeout checkpoint

The owner reports the game is mostly playable and wants to move toward Gemma
fine tuning. Further visual experiments are deferred. This is acceptance of
playability for the next phase, not a claim of photographic or physical realism.
The detailed motorcycle asset remains deferred at the owner's request.

Fresh local checks passed against the current game:

* `tools/check_agent.py`: frozen observations, fixed tick advancement, duplicate
  and stale request handling, invalid controls, repeatable reset, differing action
  outcomes, checkpoint attribution, 36 recorded transitions and 12 replayed states.
* `tools/check_camera.py`: five rendered 640 by 360 PNG observations, matching
  hashes and immutable artifacts, frozen capture ticks, changed pixels after an
  action, queued reset and advance serialization, and headless rejection.

Evidence: `artifacts/agent-closeout-check.json` and
`artifacts/qa/closeout-camera/summary.json`. These are short integration checks,
not proof of long episode stability or parallel isolation. No model training ran.

Before a learning pilot, finish versioned reward configuration, invalid action
and trainer budgets, longer parallel isolation checks,
and the trainer's observation, action and token trace joins. The current game
records raw progress, elapsed time and crash components; it is not yet a complete
training harness. Harbor integration and a verified adapter update remain open.
The full requirements remain in [the RL contract](rl-game-contract.md).

An optional server owned simulation tick budget and explicit terminal reasons are
now implemented and verified through real socket requests. See the episode
duration section of the RL contract. The default human game is unchanged.

A subsequent two process check passed with `tools/check_parallel.py`. Both workers
started from equal physical states, advanced concurrently with different controls
and identical action identifiers, and produced distinct outcomes. Resetting either
worker left the other observation unchanged. Advancing one left the other frozen;
a foreign episode identifier was rejected. Separate user data directories held
exactly 24 and 12 transitions with only their own policy identifiers. Evidence:
`artifacts/parallel-closeout-check.json`. This is short telemetry isolation evidence,
not a camera concurrency, long episode, GPU memory or throughput benchmark. A
future rollout launcher must preserve the separate process user data directories.

The optional Turn 2 edge hypothesis is preserved in development tools only.
`build_turn2_exit_envelope.py` requires its original track hash and sampled
historical profiles; `build_track.py --post-envelope` applies it only when
explicitly requested. Eight analytic tests pass. A default full reconstruction
produced byte identical track, terrain and surface files. An isolated candidate
build also succeeded, with maximum horizontal anchor residuals about 0.098 metres
on either side after refitting. Those residuals measure agreement with provisional
anchors, not real track accuracy. The candidate has not been installed or accepted;
its source imagery and taper assumptions remain uncertain. Production geometry,
textures and the playable package are unchanged by these development tools.

## Implemented

* Road clipped terrain with source elevations restored outside a provisional shoulder transition, shared off road render/contact triangles, and terrain provenance hashes. See [conforming ground](conforming-ground.md).
* Full closed East circuit from 1,536 measured geometry samples, historical lidar road elevation and cross slope, and explicitly provisional interpolated pavement widths.
* Original Streetfighter inspired motorcycle and rider meshes, measured wheelbase and nominal tire sizes, dry ground and asphalt CC0 material maps, roadside vegetation, and provisional curb geometry.
* Keyboard and gamepad riding paths, chase and rider cameras, pause, restart, Cyclone start location, speed, gears, lap clock, and course map.
* Explicit 120 Hz simulation, an assisted reduced order motorcycle model, configurable estimated parameters, and recorded requested and applied inputs. Version 3 solves current step longitudinal axle force balance, supports signed rollback, and balances stationary braking without fictional deceleration. Version 5 distributes lateral tire force to satisfy steady yaw balance while respecting each axle’s remaining friction budget. Its explicitly labeled assisted mode reserves needed lateral grip by reducing longitudinal forces within both friction circles; direct mode retains longitudinal priority. Independent analytical tests cover these changes; this remains a reduced model without yaw inertia or tire slip.
* Local TCP reset, observe and advance interface. Each action advances 12 ticks, while observation and inference delays leave simulation time unchanged. Episode and action identifiers prevent stale or duplicated execution.
* A dedicated cockpit camera provides 640 by 360 PNG observations with exact episode and tick attribution, image hashes, camera pose and intrinsics. Capture serializes with reset and advance. Saved bytes match the delivered image. Headless instances explicitly reject image capture.
* Mapped paddock buildings, paved areas, fences and aerial positioned trees; original detailed cockpit and live instruments; CC0 HDR sky with directional lighting aligned to its brightest pixel. Building heights and motorcycle body geometry remain approximations.
* Per tick JSONL state and transition recording, checkpoint attribution, track hash, physics settings and version, and authoritative state playback. Replay rejects an incompatible track hash.

## Verification so far

Godot 4.7.2 imports the project. Actual scene renders were inspected on the local RTX 2080 Ti using the Mobile renderer. Tests execute the motorcycle model, actual keyboard input path, actual TCP service, recording files and replay reader. The agent test launches an isolated process with a temporary user data directory and checks results, rather than mocking the environment.

A privileged, conservative pure pursuit test driver completed a full legal lap through the actual agent API: all 32 ordered gates, no offroad ticks or crashes, 40,313 physics ticks and 335.942 seconds of simulated riding. This validates the circuit and recording loop, not model learning or real world lap performance. The trace is generated by `tools/drive_lap.py`.

Independent banked surface tests verify gravity projection, normal load, mirrored banking, and equilibrium. Additional tests cover 60 axle force cases and assisted full throttle and hard braking on mirrored banks. The real rendered control diagnostic caught a straight launch regression in the intermediate axle allocation; the corrected combined allocation passes that unchanged input sequence, with an added explicit check against falling during launch. Camera tests exercise queued capture, advance and reset, immutable image artifacts, and explicit headless rejection. GPU rerenders are not asserted to be bit identical.

The keyboard test verifies throttle movement, braking, camera changes, pause freezing and restart through the scene. This does not replace a person's assessment of steering feel. Headless frame timings are not GPU benchmarks. A short rendered keyboard test at 1920 by 1200 on the local RTX 2080 Ti measured median frame time 17.27 ms and 95th percentile 18.53 ms. This was a brief control test, not sustained whole circuit performance acceptance. The universal application from commit `216417b` now launches on the owner’s M3 Pro MacBook using Metal 4.0 and renders the Cyclone cockpit view at 1920 by 1200. A 30 second stationary rendered check at that resolution completed 3,600 physics ticks and recorded 3,578 frames, with reported median and 95th percentile frame intervals of 8.33 ms. This is not a whole circuit benchmark or a GPU time measurement. Physical gamepad behavior and sustained whole circuit performance still require verification. The release executable does not support the editor standalone script test flag. A shared control diagnostic can now run through the game entry point with `-- --qa-controls`; it injects Godot input events and checks throttle, both brakes, steering, camera, pause and reset. The exported Mac application from commit `b6787c3` passed that shared diagnostic with zero failures on the owner’s M3 Pro using the Metal renderer at 1920 by 1200. It ran in a separate process while the first review build remained open. The short diagnostic recorded 535 frame intervals with reported median and 95th percentile values of 8.33 ms; this is not a sustained whole circuit benchmark. The exported grass JSON and texture loaded without resource errors. Physical keyboard delivery and controller ergonomics still require human review.

## Remaining acceptance work

1. Complete visual refinement against matched onboard viewpoints, including scenery placement, curb appearance, lighting, cockpit detail, audio, and camera motion. Current procedural artwork remains approximate.
2. Improve and validate motorcycle dynamics. The prototype does not yet model suspension, airborne motion, wheelies, tire slip dynamics, tire temperature, or physical impact dynamics. The historical pit wall now has continuous motorcycle contact detection and terminal recording, described in [wall contact](continuous-wall-contact.md). Engineering estimates need calibration. See the dynamics implementation document.
3. Verify full circuit riding, lap validity, surface transitions and failures. Expand adversarial gate and boundary testing before rewards are used for learning.
4. Complete native Mac acceptance beyond the successful launch and inspected screenshot: actual human riding, physical controller inputs, whole circuit frame pacing and memory. A full native circuit diagnostic now passes with exact Linux trajectory agreement and measured process frame pacing; visible presentation and physical human controls remain unverified. See [Mac benchmark evidence](mac-benchmark.md).
5. Finish the RL contract beyond the current telemetry interface: bounded experiment configuration, full reset snapshots, termination and truncation budgets, reward versioning, concurrent environment isolation tests, and trainer trace joins. Harbor and model updates remain after game review.
6. Review the playable, polished result with the owner. Passing software tests alone does not satisfy this gate or establish physical realism.

## Run and verify

Open `godot/project.godot` in Godot 4.7.2 and run the main scene, or run `godot --path godot` from the repository root. Assets and generated game geometry are included. Source reconstruction tools require the acquired reference data, but normal gameplay does not.

Controls: W or up for throttle, S or down for front brake, space for rear brake, A and D or left and right for steering, C to cycle chase, rider and onboard cameras, R for restart, Escape for pause, F11 for full screen. Gamepad triggers control throttle and front brake; the left stick steers. Rider assistance and automatic shifting are enabled in the human prototype and labeled in recordings.

Run `godot --headless --path godot --script tests/test_motorcycle.gd`, `godot --headless --path godot --script tests/test_human.gd`, and `uv run tools/check_agent.py --godot /path/to/godot` for the implemented checks.

Agent mode starts with `godot --headless --path godot -- --agent-port=PORT`. Bind is localhost only. The process owns the fixed action duration. Send newline separated JSON requests; see `tools/check_agent.py` for an exercised client. This is a development telemetry interface, not the complete Harbor training harness.

Recordings live in Godot's user data directory under `runs/RUN_ID/EPISODE_ID.jsonl`. Replay with `godot --path godot -- --replay=/absolute/path/episode.jsonl`. Playback uses recorded states and does not claim identical resimulation. Native packages carry source commit and content hashes in build manifests and episode recordings. Development packages explicitly label a dirty source tree. Unbundled runs identify their script hashes.

## Native packages

Current closeout export: `4b6929eece72-8bf7fb8bfdc8`, built from clean commit
`4b6929eece72fa6ca96a555b9c0f54086275e203`. The universal Mac archive is
`artifacts/builds/4b6929eece72-8bf7fb8bfdc8/macos/Thunderhill-macos.zip`.
Its SHA256 is
`2531614dc2ae78c37f96d63166b19e2d9bb2b6891287d78b8d6b9a5be4207e29`.
Export validates the baked inputs, universal executable, bundle identity and
unchanged source content. Nine package tool tests pass. This version includes the
optional agent episode budget and current paint filtering. After the owner brought
the Mac online, this exact archive was copied to Downloads as
`Thunderhill-closeout.zip`, its SHA256 matched, and the application was installed
under `~/Applications/ThunderhillReview/4b6929eece72-8bf7fb8bfdc8/`.
Strict deep signature verification passed. The application launched with Metal on
the Apple M3 Pro and captured a nonblack 1280 by 800 onboard image at station 1150,
then exited successfully. The image was visually inspected. Evidence:
`artifacts/Thunderhill-closeout-native.png`, its JSON sidecar, and native log.
The window was visible and drawable but not focused during this brief capture;
this verifies native launch and rendering, not sustained gameplay performance or
human acceptance. The normal game was then opened for owner review. The archive's
original export manifest remains unchanged; this paragraph records subsequent
runtime evidence.


Install the official Godot 4.7.2 export templates. From a clean committed tree, run `python3 tools/package_game.py --godot /path/to/godot --platform macos` (or `--platform linux`). Packages and manifests are written under `artifacts/builds/`. The Mac archive contains a universal application with builtin ad hoc signing. No Apple notarization or Mac runtime verification is implied by a successful export. `--allow-dirty` is only for explicitly marked development candidates.

Run `uv run tools/check_camera.py --godot /path/to/godot --display :1` on a machine with a working graphical display for actual rendered camera checks.

## Model failure handling

Unsupported dynamics are separate from crashes and lap completion. The game records an `environment_failure`, marks the rollout invalid and truncated, excludes that failed step from reward transitions, and requires reset. Successful ticks earlier in the same action remain available for diagnosis. Retrying the same failed action returns its cached result. Human play pauses with an explanation and restart prompt. This prevents an unimplemented contact regime from becoming a successful or rewarded training outcome.

Static decorative scenery is generated before packaging. If you change its
placement code, ground data, landmark exclusions or source grass assets, run
`godot --path godot --script res://tools/bake_scenery.gd` using Godot 4.7.2 with a
real renderer. Headless baking is rejected. Commit the regenerated scene and
`data/scenery-bake.json` together with the source change. The package tool rejects
stale inputs. See [the verification details](surface-rendering.md#static-scenery-bake).

## Short recorded review clips

`tools/clip_replay.py` extracts a time interval for authoritative state playback.
It preserves original transition rows and tick numbers, makes the preceding
recorded state the initial state, and retains the source policy and geometry
identity. A `playback_clip` manifest entry records the source SHA256, requested
interval, actual tick boundaries and duration. It rejects unavailable time
ranges, discontinuous ticks and existing output files. It does not create
training samples or a new control benchmark.

```sh
python3 tools/clip_replay.py /absolute/episode.jsonl --start 57.5 --end 80 --output /absolute/clip.jsonl
godot --path godot --write-movie /absolute/clip.avi --fixed-fps 30 --quit-after 700 -- --replay=/absolute/clip.jsonl --preview-camera=1
```

The frame limit includes a short stationary tail after playback finishes. Movie
writing fixes frame timing; its reported frame intervals are not live gameplay
performance. The project currently records at its 1280 by 800 window override.
Startup lighting may settle during initial frames, so inspect those separately
from steady playback.

The current Turn 2 review is `artifacts/turn2-motion.mp4`, with original AVI and
sampled contact sheet in `artifacts/turn2-motion-review`. Godot consumed all
2,699 selected transitions, preserving ticks 6901 through 9599 from the existing
privileged QA driver recording. This is not a learned policy. The 700 frame,
30 FPS movie covers approximately 22.49 seconds of recorded movement plus its
stationary tail. Source geometry hashes match current track, terrain and surface
files. The updated extraction tool reproduced identical selected states after
range validation was tightened.

Sampled frames confirm the field pattern persists around the bend. They also
show that the field remains too smooth, the cockpit too approximate and the
camera too upright compared with the supplied footage. Sampling every two
seconds cannot establish absence of fine temporal shimmer. Native Mac playback
and human visual acceptance remain outstanding.

### Model decision feed in recordings

Agent recordings now include a `model_decision` event before each accepted
control call, with the simulation tick, elapsed time, action identifier and
actual `control_bike` arguments. Retries do not duplicate events. The HUD shows
the model name, generation, current steering direction and percentage, throttle
and brake bars, and two recent calls in the upper left during live agent control
and replay. Repeated identical commands still appear separately.
The feed shows issued controls, not generated explanations or inferred reasoning.

For older evaluation recordings, join the saved model completions before clipping:

```sh
python3 tools/clip_replay.py /absolute/episode.jsonl --decisions /absolute/decisions.jsonl --start 222 --end 240.78 --output /absolute/decision-clip.jsonl
```

This validates episode identity, contiguous ticks and every applied control
against the original episode. Decision end ticks are mapped to the first
transition controlled by that call. A clip carries the call active at its start,
without displaying future calls early. Native events need no sidecar and survive
subsequent clipping. Render the enriched clip with the same movie command above.

Verified with real headless agent transport and replay checks, seven clipping
tests, and a rendered cockpit clip inspected for legible text. The example clip
is the failed baseline corner attempt, not a completed lap.

The readable overlay uses explicit `policy_display` manifest metadata. Supply
`--model-name 'GEMMA 3  270M' --generation 0` to the clip tool, or
`--model-name='GEMMA 3  270M' --generation=0` as Godot user arguments
when starting a live agent recording. These are presentation labels supplied by
the caller, never guessed from filenames. Missing labels are shown as unknown.
For the corner experiment, generation zero means its starting adapter; each
optimizer update advances the generation by one. Four sampled rollouts contribute
to each update. Model checkpoint hashes remain the authoritative identity.

Steering percentage is the normalized tool input magnitude, not steering angle
or physical lean. Negative input requests left, positive requests right. The
control window is at most 0.10 simulated seconds and may end early on termination.
Throttle bars are green, brake bars are orange, and each call pulses the panel
using simulation time so movie exports remain synchronized.
