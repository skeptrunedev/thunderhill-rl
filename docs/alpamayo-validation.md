# Alpamayo gameplay RL validation

The first experiment is one bounded generation using the published Alpamayo 1.5
driving checkpoint. It must learn from its own Thunderhill simulator outcomes.
No supervised imitation, teacher actions, demonstration dataset, or distillation
is part of this experiment.

The policy predicts a trajectory from camera frames and motion history. A fixed
motorcycle controller executes that trajectory through the simulator controls.
These controller commands are not native language model tool calls. The policy
gradient must use the actual sampled trajectory probability, never the text
serialization of the controller commands.

The integration reference is NVIDIA AlpaGym's continuous action expert replay:
https://github.com/NVlabs/alpagym

The initial cost boundary is one RTX PRO 6000, at most 25 minutes of child GPU
execution, no automatic retries, and one generation. Start with two training
rollouts and matching evaluations before and after the update, each capped at
12 simulated seconds. Record every attempt and queue videos for local rendering.
Download model weights using CPU resources before allocating the GPU.

Completion requires audited gameplay rewards, an actual nonzero optimizer update,
a saved reloadable checkpoint, evaluation of that checkpoint, complete video
coverage, and confirmation that the cloud job has terminated. One generation is
a pipeline validation, not statistical evidence of improved lap performance.

Local validation now covers real rendered camera capture, trajectory execution,
reward auditing, recording queues and persisted replay artifacts. CPU tests of
NVIDIA's stochastic sampler verify replay density and policy gradient arithmetic
using a tiny test head, not a driving checkpoint.

The actual Cosmos Reason2 processor test passes in the pinned Alpamayo
environment, including four camera frames and sixteen motion history poses.
Run these checks with `training/alpamayo/.venv/bin/python`; the previous training
environment contains incompatible Transformers configuration initialization.
The full model subsequently completed the bounded cloud validation described
below. Local arithmetic tests alone do not establish full model trainability.

The first GPU load exposed a serialization difference: the released model marks
Fourier frequencies and action normalization statistics as nonpersistent, while
the pinned RL classes expect persistent buffers. The loader reconstructs only
those seven constants from the published configuration using official module
constructors. Missing learned weights remain fatal. See NVIDIA's
[release Fourier encoder](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/action_in_proj.py)
and [action normalization](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/action_space/unicycle_accel_curvature.py).

The next GPU attempt loaded the full model and sampled trajectories, but exposed
two integration problems before any optimizer update. The legacy integer control
codec rounded small continuous controller outputs to zero. CPU software rendering
also dominated rollout time. That attempt was stopped and both attempted episodes
were archived with complete videos. It is not evidence of learning.

Trajectory execution now uses continuous JSON control values. Modal uses Vulkan
Mobile with explicit offscreen observation draws, and validates actual hardware
PNG capture before loading the model. The same offscreen path passed the local
rendered gameplay, reward audit and recording tests on the RTX 2080 Ti. Root
viewport presentation is disabled while camera viewports remain active, so Xvfb
does not need a Vulkan presentation surface. Renderer validation and training
share the original 25 minute child execution budget.

Cloud hardware capture succeeded, but an exact pixel equality assertion rejected
stationary images with only 0.346% of pixels changed, at most 3 levels out of 255
per channel. Pose and simulator state were identical. Camera QA therefore checks
bounded pixel stability separately from exact state and calibration invariance.
The underlying cause of this small hardware raster variation is unconfirmed.

## Completed cloud generation

Run `alpamayo-validation-20260924T022439Z` completed on one RTX PRO 6000.
The child execution took 171.76 seconds, including hardware camera preflight,
baseline evaluation, two training attempts, a real optimizer update, checkpoint
save and reload, and final evaluation. All four attempts passed recording audits.
Four complete videos were rendered locally, each 330 frames, including model,
generation and rollout identity. The Modal app is stopped with zero tasks and
no containers remaining. The existing HF_TOKEN environment was never modified.

This validates the RL plumbing, not racing performance. Every attempt stalled
after 10 simulated seconds. Baseline legal progress was negative 0.142863 metres;
the updated evaluation reached negative 0.142576 metres. That microscopic change
is not convincing evidence of improved driving. Do not scale to long training
based on this result; first investigate the stationary trajectory predictions
and the driving model's behavior in this motorcycle camera setting.

The update had a nonzero gradient norm of 0.00055014 and changed adapter weights.
Loading the initial adapter restored its original hash; loading the saved updated
adapter restored the new hash before evaluation. No supervised training occurred.

See [machine readable evidence](alpamayo-validation-result.json) and
[W&B run](https://wandb.ai/skeptrune-org/thunderhill-rl/runs/7lru9l4b).
The local archive is under
`artifacts/modal-alpamayo-validation-20260924T022439Z/`.
