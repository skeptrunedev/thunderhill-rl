# Camera and trajectory racing harness

Alpamayo receives four camera frames and sixteen measured ego poses. The policy
predicts future positions in metres in its current ego frame, with X forward,
Y left and Z up, at 0.1 second intervals. At a stationary reset, missing history
is padded with the stationary initial observation. Image hashes and simulator
observation identity bind camera inputs to the recorded physics state.

`DrivingEpisode` executes each sampled plan through `TrajectoryTracker`. The
controller uses only the predicted trajectory, current speed and current lean.
It does not read the centerline, choose an optimal racing line or supply training
labels. Motion between replans is approximated from speed and lean; banking and
slip can make that estimate inaccurate. Godot steer and lean are positive right,
so the conversion reverses the model's leftward turn sign.

The model action is the sampled trajectory. The resulting steering, throttle
and brake commands are deterministic controller outputs, not native language
tool calls. Their likelihood must not be substituted for the trajectory's
sampling probability during RL.

## Simulator execution

`ThunderhillEnv` retains the audited control transport:

| Operation | Purpose |
| --- | --- |
| `observe()` | Return current state and sequencing receipt without advancing time |
| `control_bike(...)` | Apply bounded controls using the latest receipt and return the resulting state |

Each control advances at most twelve physics ticks, or 0.1 simulated seconds.
Stale receipts and malformed controls execute no physics. The trainer owns reset,
episode duration, policy identity and reward collection. The policy cannot change
its rewards or silently reset an unsuccessful attempt. Assistance and automatic
shifting remain fixed by the environment.

Road telemetry remains available to the shared lifecycle and diagnostic records.
The Alpamayo policy input and trajectory controller do not use that privileged
track geometry. Reward audits independently inspect the recorded simulator
transitions and confirm that submitted controls actually executed.

## Learning and recording

The trainer groups gameplay attempts by generation and updates the policy from
those attempts' simulator rewards. Stalls, crashes and track limit failures are
valid outcomes when the recording audit passes. Infrastructure failures are
reported separately and must not be presented as successful training.

Each attempt retains camera observation identities, model trajectory decisions,
executed controls, reward components, generation and rollout identity, plus an
immutable simulator recording and queued video job. Evaluation uses identified
checkpoints separately from training collection. W&B tracking provides reward
and driving outcome graphs; a higher reward in one generation alone is not proof
of a lasting improvement.

No supervised imitation or scripted driving demonstrations are part of this
training workflow.
