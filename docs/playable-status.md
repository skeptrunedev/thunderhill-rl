# Playable game and operational checks

The Godot game supports human controls, isolated agent stepping, camera observations, physical state recordings and state replay. It uses historical Thunderhill East geometry and a Ducati Streetfighter V4 S reference. The game is playable, but its authored assets and reduced order motorcycle dynamics do not establish a hyperrealistic or physically calibrated reconstruction.

The current RL integration is [NVIDIA AlpaGym and Cosmos RL](../training/README.md). It trains the Alpamayo trajectory expert with the vision language backbone frozen. Previous Gemma, Qwen and custom Alpamayo training runbooks are obsolete. Actual GPU training through the replacement stack remains a separate validation requirement.

## Run the game

Open `godot/project.godot` in Godot 4.7.2, or run:

```sh
godot --path godot
```

Normal gameplay uses the included generated assets. Source reconstruction requires the separately acquired reference datasets.

Controls: W or up for throttle, S or down for front brake, space for rear brake, A and D or left and right for steering. C cycles cameras, R restarts, Escape pauses, and F11 toggles full screen. Gamepad triggers control throttle and front brake; the left stick steers. Rider assistance and automatic shifting are enabled and recorded.

The human camera is separate from the policy observation camera. Presentation changes must not silently change what the policy sees.

## Verify real workflows

```sh
godot --headless --path godot --script tests/test_human.gd
uv run tools/check_agent.py --godot /path/to/godot
uv run tools/check_camera.py --godot /path/to/godot --display :1
```

Camera checks require a working renderer. Headless telemetry checks are not graphics performance measurements. The game also supports `-- --qa-controls` for checking human input through an exported application. Native Mac performance and visual acceptance require the actual native build and human review, not old timing reports from another build.

See [training verification](../training/README.md#verification) for the actual Godot to NVIDIA protocol workflow. Those checks use synthetic inference fixtures and do not prove model learning.

## Record and replay

Agent mode uses `godot --headless --path godot -- --agent-port=PORT` for telemetry. Camera based RL uses a rendered instance. The server binds to localhost and owns action duration. `tools/check_agent.py` contains an exercised client.

Recordings live under Godot's user data directory at `runs/RUN_ID/EPISODE_ID.jsonl`. Replay recorded states with:

```sh
godot --path godot -- --replay=/absolute/path/episode.jsonl
```

State playback does not claim identical resimulation. Episodes identify the source, physics and geometry. Unsupported dynamics or readback failures are recorded as infrastructure failures, stop progression and require reset.

Every native AlpaGym attempt queues a video job. The launcher renders jobs serially after workers stop, including recordings from failed runs. `video_status.json` reports rendering separately from training status. `tools/render_video_queue.py` can recover jobs manually. The overlay identifies model, policy version and rollout; controls shown are the fixed controller's commands, not generated language or hidden reasoning. A streaming rollout total is not invented when it is unavailable. Evaluation attempts are labeled separately.

`tools/clip_replay.py` extracts intervals without generating new training data. It retains source identity and tick numbers. For example:

```sh
python3 tools/clip_replay.py /absolute/episode.jsonl --start 10 --end 20 --output /absolute/clip.jsonl
```

## Package and update assets

Install the official Godot 4.7.2 export templates. From a clean committed tree:

```sh
python3 tools/package_game.py --godot /path/to/godot --platform macos
```

Use `--platform linux` for Linux. Packages and manifests appear under `artifacts/builds/`. A successful export is not native runtime acceptance or Apple notarization. Dirty development candidates require the explicit `--allow-dirty` option.

After modifying scenery inputs, regenerate the baked scene with a real renderer:

```sh
godot --path godot --script res://tools/bake_scenery.gd
```

Commit the generated scene and `data/scenery-bake.json` with its inputs. Packaging rejects stale baked data. See [surface rendering](surface-rendering.md#static-scenery-bake), [source geometry](geometry-sources.md), [dimensions](dimension-register.md), and [motorcycle assumptions](dynamics-implementation.md).
