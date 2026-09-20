# Thunderhill motorcycle RL

A new Unity motorcycle racing environment for training a language model on Thunderhill East after the 2026 repave.

## Acceptance criteria

1. Motorcycle dynamics are evaluated against published models and measured behavior. Steering, tire forces, lean, braking, suspension, and rider movement must have explicit physical assumptions.
2. The track uses documented geometry, elevation, camber, and current surface references. Unknown measurements remain marked as unknown.
3. Harbor exposes the simulation to a trainer that updates the language model itself from racing rewards.
4. Simulation time advances explicitly in fixed steps. Model inference time does not advance the world.
5. Evaluation compares legal laps, completion rate, crashes, and lap times under identical conditions.
6. Model checkpoints and state trajectories support recordings of training progress from consistent camera views.

Current status: reference evaluation and development environment preparation. No playable game or trained racing model yet.

See [reference evaluation](docs/reference-evaluation.md) and [build sequence](docs/build-sequence.md).
