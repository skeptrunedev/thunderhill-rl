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
The full Alpamayo model has not completed a gameplay or optimizer validation.
Neither local integration tests nor the CPU arithmetic tests establish that the
complete model trains successfully on the planned GPU.

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
