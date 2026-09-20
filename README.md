# Thunderhill motorcycle RL

A new Godot motorcycle racing environment for training a language model on Thunderhill East after the 2026 repave. No paid engine license is required.

## Acceptance criteria

1. Motorcycle dynamics are evaluated against published models and measured behavior. Steering, tire forces, lean, braking, suspension, and rider movement must have explicit physical assumptions.
2. The track uses documented geometry, elevation, camber, and current surface references. Unknown measurements remain marked as unknown.
3. Harbor exposes the simulation to a trainer that updates the language model itself from racing rewards.
4. Simulation time advances explicitly in fixed steps. Model inference time does not advance the world.
5. Evaluation compares legal laps, completion rate, crashes, and lap times under identical conditions.
6. Model checkpoints and state trajectories support recordings of training progress from consistent camera views.

Current status: reference evaluation and development environment preparation. No playable game or trained racing model yet.

Current priority: build and review the playable game before starting any RL. The first bike is the 2026 Ducati Streetfighter V4S. The [reference readiness report](docs/reference-readiness.md) records inspected footage, acquired terrain and materials, the M3 Pro Mac target, and remaining realism gaps.

See [reference evaluation](docs/reference-evaluation.md) and [build sequence](docs/build-sequence.md).

The [RL game design contract](docs/rl-game-contract.md) defines the information the simulator and trainer must capture, action and reward semantics, isolated rollouts, and checkpoint replay requirements.

The [small language model reference audit](docs/small-model-references.md) covers RobotxR1, MindDrive, and the limits of existing racing examples.

The [Gemma 4 RL audit](docs/gemma4-rl-references.md) identifies exact E4B training code and the closest game and environment integration references.

## License

Original project code and documentation use the [MIT license](LICENSE). Referenced projects, models, footage, maps, and assets retain their own licenses and are not bundled here. Godot is [MIT licensed](https://godotengine.org/license/).
