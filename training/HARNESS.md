# NVIDIA simulator boundary

`alpagym_bridge.py` implements AlpaSim's `RuntimeService`. NVIDIA requests a
simulation with a session UUID and an `EgodriverService` address. The bridge creates
an isolated Godot process, then calls the unchanged NVIDIA driver:

1. `start_session` with actual camera calibration and a reproducible episode seed.
2. `submit_image_observation` and `submit_egomotion_observation` with recorded
   measurements at 10 Hz. A measured stationary warmup supplies initial history.
3. `submit_route` with track geometry. This is navigation input, not a
   demonstration of controls or an optimal racing line.
4. `drive` every 0.2 simulated seconds. NVIDIA assembles observation buffers,
   constructs model inputs and samples its native trajectory distribution.
5. Execute the returned trajectory with the fixed motorcycle controller.
6. `close_session`, return actual progress and failure metrics, flush recordings
   and queue the attempt for video rendering.

Rig axes are X forward, Y left and Z up. Sensor poses preserve measured pitch and
roll. Driver responses are timestamped world poses; the bridge converts future
poses into the current rig frame before execution. The controller uses trajectory,
speed and lean, without reading route geometry.

The policy action is a trajectory. `control_bike` records are controller outputs,
not language tool tokens. NVIDIA records native action replay and performs its own
GRPO update. No local likelihood or advantage implementation is substituted.

Simulation pauses during inference. Each control advances twelve physics ticks,
or 0.1 seconds. Stale receipts cannot advance simulation. Driving failure is valid
RL experience; protocol or renderer failure is an infrastructure error.
No prerecorded driving or teacher controls are used.

## Recording identity

NVIDIA discards the `current_weight_version` supplied by Cosmos. A small subclass
in `alpagym_worker.py` publishes that value and batch identity while delegating the
entire rollout to NVIDIA. The simulator snapshots the identity when it starts.
Missing or inactive identity fails rather than guessing a generation. Prefetch
is disabled so episodes cannot start outside this versioned call. Training,
sampling, inference batching and NCCL transport are unchanged.

Simulator recordings are immutable. Camera receipts and hashes bind images to
ticks; trajectory logs bind predictions to controls. Video work items are consumed
by `tools/render_video_queue.py`. Historic runs retain their original provenance
and are not relabeled as runs from the replacement stack.

Reward metrics use full circuit normalized legal progress, excluding warmup.
Every executed physics tick contributes collision and offroad events. Motorcycle
falls without obstacle contact have a separate penalty. Invalid lap status alone
is not an offroad event. See [the reward contract](README.md#reward-contract) for
coefficients and the remaining differences from NVIDIA. No reward targets an
optimized racing line or closeness to the centerline.
