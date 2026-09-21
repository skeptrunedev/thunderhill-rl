# Game design contract for language model RL

This specification is required for game implementation. The prototype implements a subset; see [implementation status](playable-status.md) for tested behavior and outstanding work. This document is not a claim that the full API or trainer is implemented. Target: Godot motorcycle simulation of Thunderhill East, Gemma 4 E4B adapter training, Harbor integration, batched rollouts on one RTX PRO 6000 96GB, and recordings of checkpoint progression.

## Simulation lifecycle

Reset creates an isolated episode from a versioned scenario, seed, and initial state. Observe reads the current state without advancing time. Advance validates an action, applies it for a configured number of physics ticks, and returns the next observation and transition record. Stop early on terminal events. Inference, camera capture, logging, and video rendering must not advance physics.

The server owns action duration. The policy cannot choose extra ticks, skip resets, change physics, set rewards, or access another episode. Each request carries an episode identifier, expected tick, and unique action identifier. Reject stale and out of order actions. A retried action identifier returns the original result rather than applying the action twice. Invalid commands receive a structured error and recorded invalid action event; do not silently substitute controls. A bounded invalid action budget prevents stalled rollouts.

Return separate terminated and truncated flags with reasons. A crash or completed evaluation episode differs from a time or token budget cutoff. Infrastructure failures invalidate a rollout instead of becoming a driving penalty. Trainers must receive these distinctions.

## Information captured

| Record | Required contents | Purpose |
| --- | --- | --- |
| Run manifest | Source commit, engine and dynamics versions, dependency versions, track asset hashes and provenance, units and coordinates, physics timestep, action cadence, scenario configuration, reward version, observation version | Identify exactly what was trained and make results comparable |
| Episode | Run, environment and episode identifiers, scenario and seed, reset snapshot identifier, starting tick, policy checkpoint and adapter identifier | Isolate parallel sessions and match candidate groups |
| Policy observation | Exact camera image bytes or immutable artifact reference, camera calibration and pose, image processing settings, capture tick, configured telemetry, observation history presented to the model | Reconstruct what the policy actually saw |
| Policy output | Exact generated text and parsed command, validation result, requested and applied controls, expected and actual tick, action duration | Connect model decisions to executed behavior |
| Transition | Previous and next observation identifiers, tick interval, simulation duration, reward components and total, terminal and truncation reasons, event list | Assign outcomes to the action that caused them |
| Physical state | Pose, linear and angular velocities, steering and lean, wheel speeds, tire forces and slip, wheel normal loads, suspension state, drivetrain state, rider state, contacts and surface properties where modeled | Validate dynamics, diagnose failures, and reconstruct motion |
| Track state | Ordered gate crossings, legal signed progress, lateral position, boundary excursions, lap validity, sector and lap times | Score racing without rewarding shortcuts or wrong way loops |
| Training trace | Prompt and chat template, processor and tokenizer versions, input and generated token identifiers, assistant token masks, rollout policy version, sampling settings, old policy log probabilities where required by the trainer, group identifier and reset key | Preserve the actual policy gradient training inputs |
| Performance | Wall time for physics, observation rendering, generation and updates, peak GPU memory, environment throughput and queue time | Choose concurrency from measurements |

Unavailable physical quantities must be marked unsupported, never invented. Checkpoint and artifact references must resolve to immutable content. Training trace fields belong to the trainer, while physical state belongs to the simulator; stable identifiers join both records.

## Policy information boundary

Road telemetry is the selected standard policy input. The lap policy receives speed, lean, lateral position, angle and distance to a lookahead centerline point, and upcoming centerline curvature. These are explicitly simulator supplied observations. Camera perception is optional future work, not a requirement for the current training goal.

Maintain an explicit allowlist of observations for each experiment. Camera input and telemetry input are separately versioned modes. A camera policy must not accidentally receive privileged tire friction, perfect future curvature, ideal racing lines, or reward internals. Telemetry experiments may expose selected simulator quantities, but their results must be labeled accordingly. Both modes may record privileged physical state for verification without putting it in the model prompt.

Use bounded observation history rather than allowing an entire lap to grow the context indefinitely. Record the exact retained history. Observation normalization, camera exposure, image resolution, and history length are part of the configuration, not hidden preprocessing.

## Motorcycle action semantics

Define actions in terms of physically available rider inputs: throttle, front and rear braking, steering actuation, gear selection, and rider movement as supported by the selected dynamics model. Specify units, bounds, rate limits, and actuator response from that model. Requested and applied values are both recorded when actuator dynamics differ.

Do not equate commanded lean angle with a physical steering input. Any balance assistance, steering controller, traction control, ABS, or gear automation must be explicit and versioned. A coarse action tool and a direct control policy are different experiments. No arbitrary numerical control limits are fixed until dynamics validation establishes them.

## Reward and lap validity

Compute rewards from simulator transitions after actions. Preserve raw measured components so alternate reward formulas can be audited. Ordered track gates and boundary rules determine valid progress and completed laps; Euclidean proximity to the finish is insufficient. Handle crossing the start line, reverse motion, leaving and reentering the course, teleportation, and incomplete laps explicitly.

Dense legal progress can provide early learning signal. Completion and elapsed simulation time determine racing performance. Crash, invalid lap, and control validity events remain separately visible. Reward coefficients require versioned experiment configuration; do not choose them implicitly in game scripts. Test whether repeating a section, oscillating across a gate, cutting corners, stopping indefinitely, or consuming the action budget increases reward unfairly.

Reward terminal events exactly once. Preserve episode component totals even if a subsequent observation is requested. Do not use text verbosity or tool syntax compliance as a substitute for racing outcome rewards. A valid command and a successful lap are separate metrics.

## Parallel rollouts and trainer integration

Each rollout owns simulator state, random generator state, observation history, and event counters. Within a GRPO group, use the same initial snapshot and scenario randomness where practical; log remaining nondeterminism. Snapshot scope includes rider controllers, actuator state, contacts and solver state where supported, track progress, and reward accumulators. A seed alone is not a full reset guarantee.

Reuse the Gemma CARLA example's action and observation cycle, the Catch example's batched generation pattern, and the E4B reference's adapter training setup. Preserve all assistant actions across the episode for training and mask simulator supplied observations from the policy loss. Pin the rollout policy version for an entire episode and synchronize new adapters before starting the next batch. The trainer, not the game, computes token log probabilities and advantages.

Expose the same simulation service through Harbor tasks and verifiers. The verifier derives outcomes from simulator owned records. The policy must not be able to edit scores or verifier inputs through its sandbox. OpenEnv reference code informs the control interface but does not replace the requested Harbor integration.

Headless telemetry training may omit rendering. Camera based training must render every observation the policy consumes with the same camera semantics used for evaluation. Separate replay rendering can produce higher resolution presentation footage without pretending those frames were policy inputs.

## Replay and evaluation

Save initial snapshots, action sequences, and periodic full snapshots, plus sufficient per tick motion state to render authoritative trajectories. Resimulation must be checked against recorded state within declared tolerances; fixed timestep does not promise bit identical physics. Visual replay from recorded states remains possible when resimulation diverges.

Evaluate checkpoints on a fixed held out scenario set, using identical physics and action cadence. Report completion rate, legal lap times, crashes, invalid laps, and action failures. Keep training reward separate from evaluation results. Record checkpoint, scenario, lap validity, simulation time, and camera view in video metadata and overlays. Use consistent replay cameras across generations and retain failed episodes as well as successful laps.

## Required verification before scaling

1. Observation and inference delays do not alter the simulated trajectory.
2. Parallel episodes cannot affect each other's state, reward, or reset.
3. Repeated reset and action sequences match within documented tolerances.
4. Different actions produce correctly attributed subsequent observations and reward.
5. A complete rollout trace reaches the trainer, produces nonzero adapter updates when advantages differ, and updates the next rollout policy.
6. Gate, boundary, termination, truncation, invalid action, and duplicate request behavior pass adversarial checks.
7. A saved checkpoint evaluation can be replayed and matched to its recorded lap metrics.
8. Measure E4B GPU memory and sustained rollout throughput before selecting environment counts or training batch sizes.

Physical realism remains a separate acceptance requirement. Correct RL plumbing cannot validate inaccurate motorcycle dynamics or track geometry.

## Implemented readback failure handling

Agent capture validates image presence, expected dimensions and nonblack RGB
bytes before encoding or recording an observation. The black image check shares
the human screenshot validator and ignores alpha. It targets the observed black
readback failure in this daylight environment; a nonblack result alone cannot
prove a complete or correct rendered frame. Camera pose and intrinsics are
unchanged. PNG encoding and artifact write failures also return infrastructure
errors. Artifact writes are flushed and checked before a response is returned.

An actual capture failure records one `environment_failure`, marks the rollout
invalid and prevents further captures or advances until reset. It does not
advance physics, fabricate an observation, grant reward or automatically retry
the GPU readback. Repeated capture requests return the recorded failure. Asking
a headless telemetry instance for an unsupported camera remains a capability
error, without attempting a readback.

Reset now flushes the complete episode manifest before acknowledging the new
episode. The end to end camera diagnostic exposed truncated final manifest
lines when a process stopped immediately after reset; deferring the flush until
the first observation was insufficient. This guarantees the buffered line has
been written before acknowledgement, not durability against a power failure.

Verification passed the shared screenshot validator, injected black/empty/wrong
size readbacks through the rendered capture path, existing wall failure atomicity,
and the real socket camera diagnostic. Five valid camera observations were
recorded with verified PNG hashes and unchanged simulation ticks during capture;
queued advance and reset requests remained serialized. Evidence is in
`artifacts/camera-failure-check.log`, `artifacts/capture-wall-check.log` and
`artifacts/qa/capture-validation-durable/summary.json`. No training was performed.

## Optional server owned episode duration

Start an agent worker with `--agent-max-episode-ticks=N`, where N is a positive
integer. The limit counts successful physics ticks since reset, including resets
from a supplied initial state. It is fixed at process startup and cannot be
changed through reset or action requests. Omission preserves unlimited prototype
behavior; human mode rejects this option. Recordings include `episode_limits`
with version `tick-budget-v1` and `max_physics_ticks` (zero means unlimited).

The final permitted tick retains its normal transition and reward components,
sets `truncated=true`, and records `truncation_reason=episode_tick_limit` and one
truncation event. The worker stops even partway through its usual 12 tick action.
Further actions require reset. Retrying the last action returns the cached result.
Observation and final camera capture remain available. A timeout remains a valid
rollout, whereas an infrastructure failure invalidates it. Crashes and completed
laps use `terminated=true` and `termination_reason` of `crash` or `lap_completed`;
task termination takes precedence if it occurs on the last permitted tick.
Lap completion does not imply lap validity; consumers must inspect `lap_valid`.

This is a simulation duration budget, not a wall clock watchdog. Idle connections,
repeated invalid commands, inference deadlines and token budgets still require
trainer supervision and further implementation. No reward coefficients or model
updates are introduced by this feature.

`tools/check_episode_limit.py --godot /path/to/godot` exercises real socket
boundaries at 1, 12 and 13 ticks, retry behavior, post limit rejection, reset,
recorded transition equality, immutable server configuration and invalid startup
arguments. Current evidence is `artifacts/episode-limit-check.json`. Existing
agent transport, authoritative replay, environment failure and rendered human
control checks also pass after the change.

## Initial tool harness validation

The [bike agent harness](../training/HARNESS.md) now exposes `observe` and
`control_bike` through TRL's `environment_factory`. The control tool includes
throttle, steering, both brakes and gear changes. Fresh observation receipts
prevent queued actions from reusing unseen state. Trainer lifecycle methods own
reset and rewards. Infrastructure faults abort reward collection even when the
trainer catches tool exceptions.

A local Qwen3 0.6B GRPO run completed eight two action telemetry rollouts across
four isolated Godot workers, with nonzero adapter updates and audited observation
masking and episode attribution. This is a harness integration result, not a
Gemma 4, camera, Harbor or racing competence result. See the linked reproduction
instructions and explicit remaining work.
