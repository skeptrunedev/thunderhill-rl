# Alpamayo gameplay reinforcement learning

The active training pipeline uses NVIDIA Alpamayo 1.5 to sample future driving
trajectories from simulator camera frames and measured motion history. A fixed
motorcycle controller executes those trajectories. Actual Godot progress,
completion, crashes, track limits and stalls determine episode rewards.

There is no supervised imitation, demonstration dataset, scripted teacher or
language tool formatting migration in this pipeline. The sampled action is a
trajectory. Serialized `control_bike` messages are controller outputs used to
execute and audit the action, not sampled model tokens.

## Environment and entry points

The isolated environment is defined in [alpamayo/pyproject.toml](alpamayo/pyproject.toml).
The previous Gemma and Qwen launchers, training environments and imitation code
have been removed. Historical result reports remain in [results](results).

* [modal_alpamayo_rl.py](modal_alpamayo_rl.py) provisions the bounded cloud validation.
* [train_alpamayo_rl.py](train_alpamayo_rl.py) collects game episodes, performs one
  generation of policy updates and evaluates the resulting checkpoint.
* [alpamayo_policy.py](alpamayo_policy.py) loads the driving policy and implements
  likelihood evaluation and RL updates using the upstream action distribution.
* [driving_episode.py](driving_episode.py) captures images and motion history and
  maintains the existing audited episode and video lifecycle.
* [driving_trajectory.py](driving_trajectory.py) converts predicted trajectories
  to motorcycle controls without reading track geometry.

Inspect each entry point's arguments before launching paid compute. A successful
build or local unit test does not establish that a cloud generation completed.
Completion requires audited gameplay, a verified optimizer update, saved model
state and its evaluation and recordings.

## Local verification

The generic unit tests require no model weights or paid compute:

```sh
PYTHONPATH=training:tools uv run --project training/alpamayo python -m unittest \
  test_agent_harness test_driving_trajectory test_lap_audit test_lap_episode \
  test_lap_policy test_lap_rollout test_video_jobs test_experiment_tracking
```

Set `THUNDERHILL_GODOT` to a Godot executable to include actual simulator
lifecycle tests. The policy distribution tests in `test_alpamayo_policy.py` also
require the pinned NVIDIA source checkout via `ALPAMAYO_SOURCE_ROOT`. Small
random test heads validate arithmetic, not driving competence.

Every actual rollout keeps its model identity, generation, rollout number,
trajectory decisions, executed controls and authoritative simulator recording.
Each trajectory also retains its tokenized inputs, diffusion samples, timesteps
and behavior density in a hashed replay artifact for reproducing the RL update.
Video jobs are queued from those recordings. W&B metrics distinguish collection,
optimizer updates and evaluation. See [HARNESS.md](HARNESS.md) for the interface.
