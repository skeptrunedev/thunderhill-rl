# NVIDIA AlpaGym with the Thunderhill simulator

Training delegates to the pinned NVIDIA AlpaGym and Cosmos RL implementation,
with a reviewed patch that carries native navigation text through inference and replay.
The previous custom Alpamayo optimizer, prompt builder, episode collector, adapter
loader and Modal launcher have been removed. Historical recordings remain in
their artifact archives. Obsolete runbooks and result reports were removed.
This migration does not establish improved driving.

NVIDIA owns model loading, observation preprocessing, diffusion sampling, replay
packing, GRPO optimization, checkpoints, weight synchronization and W&B logging.
Our code supplies Godot through NVIDIA's AlpaSim gRPC protocol. The native
configuration trains the trajectory expert, with the vision language backbone
frozen. It does not optimize language tool tokens or update the whole model.
There is no supervised imitation, ground truth trajectory reward or teacher.

## Components

* `nvidia_alpagym.py` prepares NVIDIA configuration and launches its worker stack.
* `alpagym_bridge.py` supplies measured images, full poses and route geometry,
  executes returned trajectories, reports metrics and records attempts.
* `alpagym_worker.py` delegates to NVIDIA's worker and rollout backend, preserves
  policy version identity, and records separate gameplay diagnostics.
* `driving_trajectory.py` converts trajectories into motorcycle controls. It has
  no optimizer, learned policy, racing line or demonstration labels.

The pinned source is [NVlabs/alpagym at 972d160](https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879).
Its own `uv.lock` and dependency revisions are authoritative. We no longer maintain
a different slim training environment. The former local environment may remain on
developer machines only as an ignored testing artifact.

## Install and prepare

Use a Linux CUDA host matching NVIDIA's [onboarding prerequisites](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md).
Our simulator replaces AlpaSim Wizard and scene downloads, so it does not require
NuRec datasets or AlpaSim Docker services. Cosmos still requires Redis and its
CUDA dependencies. Godot requires a working renderer and display or Xvfb.

```sh
python3 tools/setup_alpagym.py --checkout /path/to/alpagym
```

This installs the actual upstream workspace. For source inspection alone, add
`--source-only`. Download the authorized Alpamayo checkpoint and convert it using
NVIDIA's checkpoint conversion command from their onboarding guide. The model
argument below must point to that converted checkpoint, not an old LoRA adapter.

Navigation conditioning requires the reviewed Alpamayo 1.5 release. After
conversion, verify it with an actual captured `native_model_input.pt` from the
real game protocol check. This uses NVIDIA's tokenizer and compares rollout and
replay tensors; it does not optimize weights:

```sh
/path/to/alpagym/.venv/bin/python -m tools.check_native_navigation \
  --model-input /path/to/native_model_input.pt \
  --checkpoint /path/to/converted/checkpoint \
  --release-config /path/to/Alpamayo-1.5-10B/config.json \
  --output artifacts/navigation-verification.json
```

The check records source configuration provenance in the converted checkpoint.
The launcher rejects missing or changed provenance. It also verifies the pinned
source plus the exact navigation patch before any GPU workers start.

From this repository, using the upstream environment:

```sh
/path/to/alpagym/.venv/bin/python -m training.nvidia_alpagym prepare \
  --source /path/to/alpagym \
  --model /path/to/converted/checkpoint \
  --godot /path/to/godot \
  --ffmpeg /path/to/ffmpeg \
  --max-steps 1 --rollouts 2 --episode-seconds 30

/path/to/alpagym/.venv/bin/python -m training.nvidia_alpagym run /path/to/prepared/run
```

Preparation does not allocate compute or run training. `run` explicitly starts
the prepared experiment, bounded by `--max-wall-seconds` (default 3600).
The topology uses one training GPU and one rollout GPU, with NVIDIA's NCCL
transport. NVIDIA recommends at least 40 GB per GPU for this setup. Our local
RTX 2080 Ti with 11 GB cannot validate this native two GPU topology. Installation
on this machine also stopped at the missing `redis-server` prerequisite; it could
not be installed without administrator access. Full GPU training remains unverified.

`--rollouts` is the number of sibling attempts per scene (one GRPO group).
`--rollouts` defaults to NVIDIA's Alpamayo 1.5 closed-loop RL experiment (6).
`--learning-rate` and `--warmup-steps` default to NVIDIA's Alpamayo 1.5 RL
post-training recipe (2e-6, no warmup), not that experiment's smoke test value
(1e-4) or AlpaGym's generic host defaults (1e-6 with a 20 step warmup from
zero). `--max-steps` bounds NVIDIA training steps. Each step accumulates the
group's replay rows as `mini_batch=1` forwards into one optimizer step; it is
not the old one update per generation loop. Video generations identify the actual
Cosmos policy weight version, not an invented batch counter.

## Deliberate simulator differences

NVIDIA owns sampling, replay loss, optimization and checkpointing. A small versioned
patch enables the checkpoint's native navigation text input in both rollout and
training tokenization. The installer and launcher verify the exact patched file
hashes; arbitrary upstream edits are rejected. The environment is our
Godot motorcycle game, not NVIDIA's car simulator. The adapter exposes four real
camera views, matching the selected native policy camera identities and ordering:
left, front wide, right and front telephoto. Each view has 512 by 320 pixels,
with 120 degree horizontal fields for wide views and 30 degrees for telephoto.
Four frames per view and sixteen measured poses supply native temporal context.
All cameras capture the same frozen physics tick. The side cameras use declared
motorcycle mounts at 60 degrees left and right from the rider eye. These are not
recovered NVIDIA car extrinsics. Actual motorcycle roll and pitch remain visible.
Camera calibration is recorded but native model inputs do not include calibration
tensors. The domain difference therefore remains even with matching inputs.

Track centerline geometry is expressed as a simple upcoming road direction and
distance instruction using NVIDIA's native navigation tokens. This is route
context, not an optimized racing line or control target. The exact route tensor
is preserved in replay and produces identical navigation text for training.
No additional camera views or recorded expert motion are fabricated.
The control period is 0.2 seconds, sensor cadence 0.1 seconds. Rewards use actual
progress, collision and offroad metrics, excluding warmup progress. NVIDIA's
car controller is replaced by our motorcycle controller and physics. Sensor
protocol compatibility is not complete simulator or driving distribution parity.
The controller follows native world XYZ using measured current position and
forward direction, horizontal pure pursuit steering and 3D waypoint speed. Below 4 m/s it requests
wheel angle; above that threshold it requests lean, matching the game actuator.
Recorded tracking residuals distinguish execution error from policy plans. Model
motion history retains full measured roll and pitch.

Every attempt has session identity, policy version, sampled trajectories, executed
controls, observation receipts, simulator recordings and an immutable video job,
including failed attempts. After training workers stop, the launcher renders
queued recordings serially, including recordings from failed runs. FFmpeg is
checked before launch; `--ffmpeg` defaults to `ffmpeg`. `video_status.json` reports
rendering separately from `run_status.json`, so a training failure is not hidden
by video work. `tools/render_video_queue.py` also supports manual recovery.
Rendering is playback of recorded states, not a second gameplay rollout.

## Verification

Verification uses actual simulator workflows rather than standalone unit tests.
With NVIDIA host/protocol packages, Torch, a Godot executable and a display:

```sh
PYTHONPATH=.:training:tools THUNDERHILL_GODOT=/path/to/godot python -m unittest \
  training.test_alpagym_bridge training.test_driving_trajectory \
  training.test_lap_episode
```

Set `THUNDERHILL_GODOT` to an executable and provide a display to include real
Godot protocol tests. Their synthetic driver is strictly a test fixture; no
fixture actions enter training. Simulator checks do not verify a
GPU update. A training validation requires real NVIDIA workers, nonzero optimizer
steps, checkpoint output and evaluated gameplay recordings.

## Reward contract

All reward terms share one unit: metres divided by `20 m/s * episode_seconds`,
the distance a rider averaging 20 m/s covers in the fixed horizon (warmup
excluded). `progress` is credited signed legal distance in that unit, so it is a
progress rate over a fixed horizon. Horizon time not executed because the
attempt stopped early (stall, crash, track limits) earns zero progress. A safe
completed lap is credited at its own mean lap speed for the whole horizon, so a
faster lap always scores higher. Raw metres, credited metres and the horizon
progress rate remain in each summary.
The existing centerline measures route position; proximity to it earns no reward.
No optimized racing line has been selected. Preview the reference with:

```sh
uv run tools/plot_track_reference.py --output artifacts/track-reference.png
```

Progress is the only reward term (v5). Obstacle collisions, offroad excursions
and falls end the episode, which forfeits the rest of the horizon; their kinetic
energy as metres of 0.25 g braking (`collision_cost`, `offroad_cost`,
`fall_cost`) is reported as a diagnostic, not rewarded, because charging it on
top of the forfeited horizon ranked stalling above attempting a corner.
Events are accumulated across every executed physics tick, including brief
excursions between observations. An invalid lap alone is not an offroad event.
The 0/1 flags `collision_any`, `offroad` and `fall_without_collision` are still
reported under NVIDIA's names but are not reward terms.

This is an adaptation, not exact reward parity. NVIDIA projects position onto a
recorded scene trajectory and tests vehicle footprint geometry. We use legal
circuit progress and the game's on_track point test. NVIDIA's optional reference
trajectory distance penalty is omitted; there is no recorded expert trajectory or
imitation target. New summaries and prepared runs identify the reward version;
stale prepared configurations are rejected. Historical results are unchanged.

The stall monitor still ends an attempt with under 1 m of legal progress in a
5 s window after 5 s grace. Because unexecuted horizon earns nothing, the stalled
attempt scores what a bike that stayed stopped would score over the full horizon.
See [reward details](../docs/reward-references.md).
