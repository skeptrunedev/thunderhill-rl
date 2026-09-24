# NVIDIA AlpaGym with the Thunderhill simulator

Training delegates to the pinned NVIDIA AlpaGym and Cosmos RL implementation.
The previous custom Alpamayo optimizer, prompt builder, episode collector, adapter
loader and Modal launcher have been removed. Historical recordings and result
reports remain intact. This migration does not establish improved driving.

NVIDIA owns model loading, observation preprocessing, diffusion sampling, replay
packing, GRPO optimization, checkpoints, weight synchronization and W&B logging.
Our code supplies Godot through NVIDIA's AlpaSim gRPC protocol.
There is no supervised imitation, ground truth trajectory reward or teacher.

## Components

* `nvidia_alpagym.py` prepares NVIDIA configuration and launches its worker stack.
* `alpagym_bridge.py` supplies measured images, full poses and route geometry,
  executes returned trajectories, reports metrics and records attempts.
* `alpagym_worker.py` delegates to NVIDIA's worker and rollout backend. Its sole
  rollout override preserves Cosmos's actual weight version for videos.
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

From this repository, using the upstream environment:

```sh
/path/to/alpagym/.venv/bin/python -m training.nvidia_alpagym prepare \
  --source /path/to/alpagym \
  --model /path/to/converted/checkpoint \
  --godot /path/to/godot \
  --max-steps 1 --rollouts 2 --episode-seconds 30

/path/to/alpagym/.venv/bin/python -m training.nvidia_alpagym run /path/to/prepared/run
```

Preparation does not allocate compute or run training. `run` explicitly starts
the prepared experiment, bounded by `--max-wall-seconds` (default 3600).
The topology uses one training GPU and one rollout GPU, with NVIDIA's NCCL
transport. NVIDIA recommends at least 40 GB per GPU for this setup. Our local
11 GB GPU cannot validate full model optimization.

`--rollouts` is the number of sibling attempts per scene. `--max-steps` bounds
NVIDIA training steps. A step can contain several optimizer minibatches; it is
not the old one update per generation loop. Video generations identify the actual
Cosmos policy weight version, not an invented batch counter.

## Deliberate simulator differences

The official training and model code remains unchanged. The environment is our
Godot motorcycle game, not NVIDIA's car simulator. The adapter exposes one real
forward camera, measured stationary warmup and track centerline route geometry.
It does not fabricate additional camera views or recorded expert motion.
The control period is 0.2 seconds, sensor cadence 0.1 seconds. Rewards use actual
progress, collision and offroad metrics, excluding warmup progress.

Every attempt has session identity, policy version, sampled trajectories, executed
controls, observation receipts, simulator recordings and an immutable video job,
including failed attempts. Render them using `tools/render_video_queue.py`.
Rendering is playback of recorded states, not a second gameplay rollout.

## Verification

With NVIDIA host/protocol packages and Torch installed:

```sh
PYTHONPATH=.:training:tools python -m unittest \
  training.test_nvidia_alpagym training.test_alpagym_worker \
  training.test_alpagym_bridge
```

Set `THUNDERHILL_GODOT` to an executable and provide a display to include real
Godot protocol tests. Their synthetic driver is strictly a test fixture; no
fixture actions enter training. CPU config and simulator checks do not verify a
GPU update. A training validation requires real NVIDIA workers, nonzero optimizer
steps, checkpoint output and evaluated gameplay recordings.
