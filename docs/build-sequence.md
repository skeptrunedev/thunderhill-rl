# Build sequence

## 1. Validate references and tooling

Run the public motorcycle dynamics model with its supported dependency versions. Record source commits, solver configuration, actions, initial state, and outputs. Evaluate physical and numerical limits. Install and pin the free Godot editor before claiming compilation or gameplay verification.

## 2. Establish the simulation contract

Implement against the [RL game design contract](rl-game-contract.md), including observation boundaries, transition records, action semantics, reset state, trainer traces, and replay verification.

Use explicit reset, observe, and advance operations. Each advance applies bounded controls for a fixed number of physics ticks. Validate bounds against the selected motorcycle model rather than inventing arbitrary limits. Record simulation time separately from wall time. Every response identifies the episode, tick, terminal state, and configuration.

Keep the validated dynamics implementation separate from Godot presentation. Physics forces, controllers, observations, and reward calculations must participate in one explicitly ordered loop. Test consistency across different rendering frame rates and inference delays. A fixed step alone does not establish deterministic replay. Godot's default vehicle body is not a validated motorcycle model.

## 3. Build and review a playable section

Follow the [reference readiness report](reference-readiness.md). Build the Cyclone approach, crest, descent, and exit using acquired terrain, footage observations, and the selected Ducati Streetfighter V4S baseline. Implement manual controls, camera modes, pause and restart. Validate appearance, riding behavior, and actual performance on the M3 Pro target. No RL training begins during this phase.

## 4. Reconstruct and polish Thunderhill East

Establish scale and coordinate system from measured sources. Build elevation and camber before detailed scenery. Confirm the course configuration from footage. Validate recognizable views around the lap. Track uncertainties and reference provenance. Do not label a flat trace of the PDF as a realistic reconstruction.

Expand the accepted section into a complete manually playable circuit. Finish controls, cameras, presentation, lap timing and reset flows. Review the game with the owner before proceeding to model training.

## 5. Validate riding physics

Compare acceleration, braking, steady cornering, steering response, lean transitions, and suspension behavior against reference results or telemetry. Exercise wheel lift, tire slip, contact loss, and collisions. Reject artificial grip or balance forces that conceal missing dynamics. Any modeled rider controller has to use physically available actuation and be disclosed in the action contract.

## 6. Prove the training cycle after game review

Only after the game is playable, visually reviewed, and approved for this phase, connect Harbor to the actual trainable language model. Demonstrate observation delivery, generated controls, independently computed racing reward, gradient updates, changed model weights, and repeatable evaluation. A mock environment or a conventional RL policy is not acceptance evidence for this step.

Compare initial and trained checkpoints on held out initial conditions. Keep behavior cloning as an explicit possible initialization strategy, not an undocumented substitution for RL.

## 7. Scale and record

Measure simulation and inference throughput before choosing environment concurrency or training budgets. Keep renderer and physical fidelity independently configurable without changing the evaluated dynamics. Preserve model checkpoints, model observations, actions, full replay states, seeds, physics settings, and rewards. Render comparable laps with checkpoint, lap time, and crash overlays. Compare against human or optimized model baselines without claiming mathematical optimality.
