# Simulator contract for gameplay RL

The current integration uses NVIDIA AlpaGym and Cosmos RL with the Godot Thunderhill motorcycle simulator. Alpamayo predicts trajectories from camera observations and measured motion history. Native RL trains its trajectory expert while freezing the vision language backbone. There is no supervised imitation, demonstration collection, teacher action or reward for matching an expert line.

This describes the implemented boundary and remaining validation requirements. It does not claim the replacement stack has completed a real GPU training update. See [training instructions](../training/README.md) and [the harness](../training/HARNESS.md).

## Simulation lifecycle

Each rollout owns a Godot process, episode identity, controller state and recording directory. Reset initializes an episode. Observe and camera capture do not advance physics. Each accepted control advances up to twelve physics ticks, or 0.1 seconds. Model inference pauses simulation time. The bridge asks NVIDIA for a new trajectory every 0.2 simulated seconds.

Requests identify the episode, expected tick and action. Stale requests are rejected. Repeating an accepted action identifier returns its recorded response instead of applying the action twice. Termination, truncation and infrastructure failure must remain distinguishable. Driving failures are valid RL experience; renderer, protocol and unsupported dynamics failures invalidate the attempt.

Current episodes begin from a standing start. A measured 1.5 second stationary warmup supplies history and is excluded from progress reward. Episode duration is configured before launch. The default 30 second validation horizon is not sufficient evidence of a full lap.

## Observation and action boundary

The bridge supplies a real forward camera, exact image receipts and full measured poses at 10 Hz. NVIDIA preprocessing retains four camera frames and sixteen historical poses. Raw PNG captures and the exact JPEG bytes sent to the driver are retained. Camera pose, timestamps and image hashes identify what was observed.

The policy camera renders 512 by 320 pixels with a 120 degree horizontal field of view, matching the native input aspect ratio. It remains a single motorcycle view rather than the upstream multiple camera setup.

The simulator also supplies twenty centerline route points covering eighty metres. They satisfy the policy buffer contract, but the pinned Alpamayo R1 adapter does not pass route data into the model. Do not claim that this checkpoint follows road telemetry through that field. Camera calibration is also metadata rather than a model input tensor. One synthetic motorcycle view remains a different observation distribution from NVIDIA's multiple vehicle cameras.

The sampled policy action is a trajectory, not a language tool call. A fixed motorcycle controller converts that trajectory into throttle, steering, front brake and rear brake commands. Requested and applied controls are recorded. The controller and the motorcycle's balance assistance and automatic shifting are explicit simulator components, separate from the learned trajectory expert.

The controller consumes native world XYZ and measured current position and forward direction. It uses horizontal pure pursuit steering and 3D waypoint distances for target speed. Full roll and pitch remain in model observations; the controller does not interpret that banked rig as a level plane.

## Reward and lap validity

The current [reward contract](../training/README.md#reward-contract) normalizes signed legal progress by full circuit length and accumulates actual collision, offroad and motorcycle fall events across executed ticks. Warmup is excluded. Raw measurements accompany the aggregate metrics. Progress measurement uses existing track geometry and does not reward staying near the centerline. No optimized racing line has been chosen.

Ordered gates and lap validity determine circuit completion. Incomplete trajectories, brief offroad events, reversals and crashes must not be reported as successful laps. Stall detection ends an attempt that makes insufficient legal progress. A completed lap terminates the current episode; continuous multiple lap episodes are not implemented.

This reward is an explicit motorcycle adaptation of NVIDIA's metric reward, not full parity with its recorded trajectory and vehicle footprint scoring. It has no separate lap time bonus. Faster progress within a fixed horizon can score higher; equally complete legal laps receive the same progress term.

## Required records

| Record | Required contents |
| --- | --- |
| Run | Source revisions, resolved configuration, model checkpoint, geometry and physics identities, reward definition, observation settings |
| Episode | Session UUID, scenario, seed, actual policy weight version, rollout identity, final state and stop reason |
| Observation | Exact sent image bytes, capture tick, timestamp, camera metadata and measured pose |
| Decision | Sampled trajectory, execution interval, issued controls and controller diagnostics |
| Transition | Physical state, legal progress, contacts, ordered gate state, termination and truncation fields |
| Training | Native replay payload, sampling settings, group membership, log probabilities, advantages, optimizer and checkpoint records |
| Video | Immutable state recording, policy identity, generation, rollout label and render job |

Unavailable physical quantities must remain unsupported rather than invented. The trainer owns replay likelihoods, group advantages and optimization. The simulator owns state transitions and measured outcomes. The video generation label comes from Cosmos's actual policy weight version, not an inferred update counter. Failed attempts retain recordings too.

## Verification before scaling

Use checks that run the actual game or the complete workflow. Verify request identity, capture timing, recording and replay, control response, termination and independent rollout state. Synthetic inference fixtures verify transport and preprocessing only; their actions never become training data.

A real GPU validation must demonstrate native model inference, completed optimizer steps, changed expert weights, checkpoint save and reload, synchronized rollout weights, and evaluated gameplay recordings. Confirm that the frozen backbone remains frozen. Compare held out gameplay outcomes under identical conditions rather than using loss alone as evidence of driving improvement.

Human playability, photographic appearance and physically calibrated handling remain separate acceptance criteria. Protocol correctness and an optimizer update do not establish those properties.
