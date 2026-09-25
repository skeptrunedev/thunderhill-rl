# NVIDIA simulator boundary

`alpagym_bridge.py` implements AlpaSim's `RuntimeService`. NVIDIA requests a
simulation with a session UUID and an `EgodriverService` address. The bridge creates
an isolated Godot process, then calls the unchanged NVIDIA driver:

1. `start_session` with actual camera calibration and a reproducible episode seed.
2. `submit_image_observation` and `submit_egomotion_observation` with recorded
   measurements at 10 Hz. A measured stationary warmup supplies initial history.
3. `submit_route` with track geometry to satisfy the native policy buffer
   contract. Our reviewed native source patch turns it into navigation text using
   NVIDIA's existing navigation prompt builder and preserves it for training replay.
   It is not an optimal racing line.
4. `drive` every 0.2 simulated seconds. NVIDIA assembles observation buffers,
   constructs model inputs and samples its native trajectory distribution.
5. Execute the returned trajectory with the fixed motorcycle controller.
6. `close_session`, return actual progress and failure metrics, flush recordings
   and queue the attempt for video rendering.

Rig axes are X forward, Y left and Z up. Sensor poses preserve measured pitch and
roll. Driver responses are timestamped world poses. The controller follows native
world XYZ using measured current position and forward direction. Steering uses
horizontal pure pursuit and target speed uses 3D waypoint distance. Sensor history
retains the measured banked rig pose independently. The controller reads no route
geometry. The policy receives four genuine synchronized cameras at 512 by 320 pixels:
left, front wide, right (120 degree horizontal fields) and front telephoto
(30 degrees). Native preprocessing packs four measured frames per camera.
Motorcycle mounts differ from NVIDIA car mounts. Below 4 m/s the controller
requests wheel angle; above it, lean. Tracking residuals are recorded.

The policy action is a trajectory. `control_bike` records are controller outputs,
not language tool tokens. The reviewed navigation patch carries route context through both input paths.
NVIDIA records native action replay and performs its own
GRPO update of its trajectory expert. The vision language backbone is frozen.
No local likelihood or advantage implementation is substituted. Supplying
correct camera calibration does not remove the visual domain difference: native
model inputs do not include camera calibration tensors.

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
The same wrapper persists immutable batch gameplay metrics and writes a separate
W&B diagnostics run per rollout worker, grouped by prepared run. NVIDIA's trainer
logger and reward computation remain unchanged.

Simulator recordings are immutable. Camera receipts and hashes bind images to
ticks; trajectory logs bind predictions to controls. Video work items are consumed
by `tools/render_video_queue.py` automatically and serially after workers stop,
including after a failed training attempt. Rendering status is separate in
`video_status.json`. Historic runs retain their original provenance
and are not relabeled as runs from the replacement stack.

Reward metrics use legal progress over the fixed episode horizon, excluding
warmup, with unexecuted horizon earning zero. Every executed physics tick
contributes collision, offroad and fall events, each costing the rider's kinetic
energy at onset. Safe completed laps are credited at their own lap pace. Invalid
lap status alone is not an offroad event. See [the reward contract](README.md#reward-contract) for
coefficients and the remaining differences from NVIDIA. No reward targets an
optimized racing line or closeness to the centerline.
